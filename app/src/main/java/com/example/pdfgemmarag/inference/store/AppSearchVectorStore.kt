package com.example.pdfgemmarag.inference.store

import android.content.Context
import android.util.Log
import androidx.appsearch.app.AppSearchSession
import androidx.appsearch.app.EmbeddingVector
import androidx.appsearch.app.Features
import androidx.appsearch.app.GenericDocument
import androidx.appsearch.app.GetByDocumentIdRequest
import androidx.appsearch.app.PutDocumentsRequest
import androidx.appsearch.app.SearchSpec
import androidx.appsearch.app.SetSchemaRequest
import androidx.appsearch.localstorage.LocalStorage
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.DocumentInfo
import com.example.pdfgemmarag.core.process.ProcessGuard
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import kotlinx.coroutines.guava.await
import java.io.Closeable
import java.util.concurrent.Executors

/**
 * AppSearch `LocalStorage` as the vector + keyword index. Exactly one instance lives in the
 * :inference process; `LocalStorage` is a per-process embedded Icing engine and opening the same
 * database from two processes corrupts it.
 *
 * Search is hybrid: exact cosine over 8-bit quantised embeddings (`semanticSearch`, brute force, no
 * ANN) OR'ed with a prefix keyword match, ranked by summed semantic score plus a small BM25 term.
 */
class AppSearchVectorStore private constructor(private val session: AppSearchSession) : Closeable {

    data class FeatureReport(val supported: Map<String, Boolean>) {
        val hybridOk: Boolean get() = REQUIRED.all { supported[it] == true }
        override fun toString() = supported.entries.joinToString("\n") { "${it.key}=${it.value}" }
    }

    val features: FeatureReport by lazy {
        val f: Features = session.features
        FeatureReport(ALL_CHECKED.associateWith { f.isFeatureSupported(it) })
    }

    suspend fun setSchema() {
        val compatible = SetSchemaRequest.Builder()
            .addDocumentClasses(PdfChunkDocument::class.java)
            .build()
        try {
            session.setSchemaAsync(compatible).await()
            Log.i(TAG, "schema set (compatible, index preserved); features:\n$features")
        } catch (t: Throwable) {
            // Incompatible change only: forceOverride deletes every document. Never do this on
            // a matching schema — that is what wiped the index after :inference restarted.
            Log.w(TAG, "schema incompatible (${t.message}); force-override (index will be empty)")
            val forced = SetSchemaRequest.Builder()
                .addDocumentClasses(PdfChunkDocument::class.java)
                .setForceOverride(true)
                .build()
            session.setSchemaAsync(forced).await()
            Log.i(TAG, "schema set (forced); features:\n$features")
        }
    }

    /** Batched put; every [flushEvery] documents a flush is requested so a crash loses at most that many. */
    suspend fun putChunks(docs: List<PdfChunkDocument>, batchSize: Int = 100, flushEvery: Int = 500) {
        var sinceFlush = 0
        for (batch in docs.chunked(batchSize)) {
            val result = session.putAsync(PutDocumentsRequest.Builder().addDocuments(batch).build()).await()
            if (!result.isSuccess) {
                val first = result.failures.entries.first()
                throw IllegalStateException("AppSearch put failed for ${first.key}: ${first.value.errorMessage}")
            }
            sinceFlush += batch.size
            if (sinceFlush >= flushEvery) { session.requestFlushAsync().await(); sinceFlush = 0 }
        }
        session.requestFlushAsync().await()
    }

    suspend fun putChunk(doc: PdfChunkDocument) = putChunks(listOf(doc), flushEvery = Int.MAX_VALUE)

    suspend fun flush() { session.requestFlushAsync().await() }

    /**
     * Hybrid query for one document. [queryVec] is the L2-normalised query embedding.
     * @param similarityFloor cosine below which vector hits are ignored (tunable; 0.3 default).
     * @param keywordWeight weight of the BM25-ish relevance score in the final rank (tunable).
     */
    suspend fun search(
        docHash: String,
        queryText: String,
        queryVec: FloatArray,
        topK: Int,
        similarityFloor: Double = 0.3,
        keywordWeight: Double = 0.05,
    ): List<Citation> {
        val terms = HybridQuery.keywordTerms(queryText)
        var hits = executeSearch(docHash, HybridQuery.build(terms, similarityFloor, topK), terms, queryVec, topK, keywordWeight)
        Log.i(TAG, "search ns=${docHash.take(8)} q='${queryText.take(80)}' terms=$terms floor=$similarityFloor -> ${hits.size} hits " +
            hits.take(5).joinToString { "p${it.pageNumber}@${"%.3f".format(it.score)}" })
        if (hits.isEmpty()) {
            Log.w(TAG, "hybrid empty; retrying semantic-only floor=0")
            hits = executeSearch(docHash, HybridQuery.build(emptyList(), 0.0, topK), emptyList(), queryVec, topK, keywordWeight)
            Log.i(TAG, "semantic-only fallback -> ${hits.size} hits " +
                hits.take(5).joinToString { "p${it.pageNumber}@${"%.3f".format(it.score)}" })
        }
        hits.forEachIndexed { i, c ->
            if (i < 5) Log.i(TAG, "  #$i p${c.pageNumber} c${c.chunkIndex} score=${"%.3f".format(c.score)} '${c.text.take(120).replace('\n', ' ')}'")
        }
        return hits
    }

    private suspend fun executeSearch(
        docHash: String,
        query: String,
        terms: List<String>,
        queryVec: FloatArray,
        topK: Int,
        keywordWeight: Double,
    ): List<Citation> {
        val spec = SearchSpec.Builder()
            .setListFilterQueryLanguageEnabled(true)
            .apply { if (terms.isNotEmpty()) addSearchStringParameters(terms) }
            .addEmbeddingParameters(listOf(EmbeddingVector(queryVec, EmbeddingGemmaEmbedder.MODEL_SIGNATURE)))
            .setDefaultEmbeddingSearchMetricType(SearchSpec.EMBEDDING_SEARCH_METRIC_TYPE_COSINE)
            .setRankingStrategy(
                "sum(this.matchedSemanticScores(getEmbeddingParameter(0))) + $keywordWeight * this.relevanceScore()",
            )
            .addFilterNamespaces(docHash)
            .addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE)
            .setResultCountPerPage(topK)
            .addProjection(PdfChunkDocument.SCHEMA_TYPE, listOf("text", "pageNumber", "chunkIndex"))
            .build()
        Log.d(TAG, "query=$query")
        val results = session.search(query, spec)
        try {
            val page = results.nextPageAsync.await()
            return page.map { r ->
                val g: GenericDocument = r.genericDocument
                Citation(
                    chunkId = g.id,
                    docHash = g.namespace,
                    pageNumber = g.getPropertyLong("pageNumber").toInt(),
                    chunkIndex = g.getPropertyLong("chunkIndex").toInt(),
                    score = r.rankingSignal,
                    text = g.getPropertyString("text") ?: "",
                )
            }
        } finally {
            results.close()
        }
    }

    /** A few stored chunks, no ranking — used to prove the namespace is not empty. */
    suspend fun sampleChunks(docHash: String, limit: Int = 3): List<Citation> {
        val spec = SearchSpec.Builder()
            .addFilterNamespaces(docHash)
            .addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE)
            .setRankingStrategy(SearchSpec.RANKING_STRATEGY_CREATION_TIMESTAMP)
            .setResultCountPerPage(limit)
            .addProjection(PdfChunkDocument.SCHEMA_TYPE, listOf("text", "pageNumber", "chunkIndex"))
            .build()
        val results = session.search("", spec)
        try {
            return results.nextPageAsync.await().map { r ->
                val g = r.genericDocument
                Citation(
                    chunkId = g.id, docHash = g.namespace,
                    pageNumber = g.getPropertyLong("pageNumber").toInt(),
                    chunkIndex = g.getPropertyLong("chunkIndex").toInt(),
                    score = 0.0, text = g.getPropertyString("text") ?: "",
                )
            }
        } finally {
            results.close()
        }
    }

    suspend fun getCitation(chunkId: String): Citation? {
        val docHash = chunkId.substringBefore(':')
        val result = session.getByDocumentIdAsync(
            GetByDocumentIdRequest.Builder(docHash).addIds(chunkId).build(),
        ).await()
        val g = result.successes[chunkId] ?: return null
        return Citation(
            chunkId = g.id, docHash = g.namespace,
            pageNumber = g.getPropertyLong("pageNumber").toInt(),
            chunkIndex = g.getPropertyLong("chunkIndex").toInt(),
            score = 0.0, text = g.getPropertyString("text") ?: "",
        )
    }

    /** Deletes every chunk of one document (used for delete and for rolling back a cancelled index). */
    suspend fun removeDocument(docHash: String) {
        val spec = SearchSpec.Builder().addFilterNamespaces(docHash).addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE).build()
        session.removeAsync("", spec).await()
        session.requestFlushAsync().await()
    }

    /** Lists indexed documents by reading one chunk per namespace. */
    suspend fun listDocuments(): List<DocumentInfo> {
        val namespaces = session.namespacesAsync.await()
        val out = ArrayList<DocumentInfo>()
        for (ns in namespaces) {
            val spec = SearchSpec.Builder()
                .addFilterNamespaces(ns).addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE)
                .setRankingStrategy(SearchSpec.RANKING_STRATEGY_CREATION_TIMESTAMP)
                .setResultCountPerPage(1)
                .addProjection(PdfChunkDocument.SCHEMA_TYPE, listOf("docName", "script", "pageCount"))
                .build()
            val results = session.search("", spec)
            try {
                val first = results.nextPageAsync.await().firstOrNull() ?: continue
                val g = first.genericDocument
                out += DocumentInfo(
                    docHash = ns,
                    displayName = g.getPropertyString("docName") ?: ns,
                    pageCount = g.getPropertyLong("pageCount").toInt(),
                    chunkCount = countChunks(ns),
                    script = g.getPropertyString("script") ?: "",
                )
            } finally { results.close() }
        }
        return out
    }

    private suspend fun countChunks(ns: String): Int {
        val spec = SearchSpec.Builder().addFilterNamespaces(ns).addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE)
            .setResultCountPerPage(500).addProjection(PdfChunkDocument.SCHEMA_TYPE, emptyList()).build()
        val results = session.search("", spec)
        var n = 0
        try {
            while (true) {
                val page = results.nextPageAsync.await()
                if (page.isEmpty()) break
                n += page.size
            }
        } finally { results.close() }
        return n
    }

    override fun close() = session.close()

    companion object {
        private const val TAG = "AppSearchVectorStore"
        const val DB_NAME = "rag_chunks"

        val REQUIRED = listOf(
            Features.SCHEMA_EMBEDDING_PROPERTY_CONFIG,
            Features.SEARCH_SPEC_SEARCH_STRING_PARAMETERS,
            Features.SEARCH_SPEC_ADVANCED_RANKING_EXPRESSION,
            Features.LIST_FILTER_QUERY_LANGUAGE,
        )
        val ALL_CHECKED = REQUIRED + listOf(
            Features.SCHEMA_EMBEDDING_QUANTIZATION,
            Features.NUMERIC_SEARCH,
            Features.VERBATIM_SEARCH,
        )

        private val executor by lazy { Executors.newSingleThreadExecutor { r -> Thread(r, "appsearch-worker") } }

        suspend fun open(context: Context): AppSearchVectorStore = open(context, DB_NAME, enforceProcess = true)

        /**
         * [enforceProcess] = false and a private [dbName] are for instrumented tests, which run in the
         * default process. Production code must use [open] with the single [DB_NAME].
         */
        suspend fun open(context: Context, dbName: String, enforceProcess: Boolean): AppSearchVectorStore {
            if (enforceProcess) ProcessGuard.requireInferenceProcess(context)
            val ctx = LocalStorage.SearchContext.Builder(context, dbName)
                .setWorkerExecutor(executor)
                .build()
            val session = LocalStorage.createSearchSessionAsync(ctx).await()
            val store = AppSearchVectorStore(session)
            store.setSchema()
            return store
        }
    }
}
