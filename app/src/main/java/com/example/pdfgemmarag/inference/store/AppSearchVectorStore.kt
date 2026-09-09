package com.example.pdfgemmarag.inference.store

import android.content.Context
import android.util.Log
import androidx.appsearch.app.AppSearchSession
import androidx.appsearch.app.ExperimentalAppSearchApi
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
@androidx.annotation.OptIn(markerClass = [ExperimentalAppSearchApi::class])
class AppSearchVectorStore private constructor(
    private val session: AppSearchSession,
    private val manifests: DocumentStructureManifestStore,
) : Closeable {

    data class FeatureReport(val supported: Map<String, Boolean>) {
        val hybridOk: Boolean get() = REQUIRED.all { supported[it] == true }
        override fun toString() = supported.entries.joinToString("\n") { "${it.key}=${it.value}" }
    }

    val features: FeatureReport by lazy {
        val f: Features = session.features
        FeatureReport(ALL_CHECKED.associateWith { f.isFeatureSupported(it) })
    }

    suspend fun setSchema() {
        check(features.hybridOk) {
            "This device's AppSearch implementation is missing required RAG features:\n$features"
        }
        val compatible = SetSchemaRequest.Builder()
            .addDocumentClasses(PdfChunkDocument::class.java)
            .build()
        // Never force-override here. A catch-all override turns transient storage failures into
        // silent data loss. Explicit schema migrations must be versioned and user-visible.
        session.setSchemaAsync(compatible).await()
        Log.i(TAG, "schema set (compatible, index preserved); features:\n$features")
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
        sectionId: String? = null,
        specificationNumber: String? = null,
        /** Trusted published namespace supplied by the UI; permits FACT recovery if the manifest is corrupt. */
        indexNamespace: String? = null,
    ): List<Citation> {
        val namespace = indexNamespace
            ?.takeIf { it == docHash || it.startsWith("$docHash:") }
            ?: activeNamespace(docHash)
        val terms = HybridQuery.keywordTerms(queryText)
        // Over-fetch so the local term-coverage re-rank can promote chunks whose wording differs
        // from the question; the vector function is given the same limit so more candidates
        // carry a semantic score instead of only a small keyword score.
        val requested = if (sectionId == null && specificationNumber == null) topK * CANDIDATE_FACTOR else topK * 6
        val requiredPropertyTerm = specificationNumber?.let { "specificationNumber" to it }
        fun List<Citation>.withinRequestedScope(): List<Citation> = asSequence()
            .filter { sectionId == null || it.sectionId == sectionId }
            .filter { specificationNumber == null || it.specificationNumber.equals(specificationNumber, true) }
            .toList()
        val raw = try {
            executeSearch(
                docHash, namespace,
                HybridQuery.build(terms, similarityFloor, requested, requiredPropertyTerm),
                terms, queryVec, requested, keywordWeight,
            )
        } catch (t: Exception) {
            // Prefix operators are a query-language feature; fall back to exact terms if the
            // installed AppSearch rejects them so retrieval never fails outright.
            Log.w(TAG, "prefix query rejected (${t.message}); retrying with exact terms")
            executeSearch(
                docHash, namespace,
                HybridQuery.build(terms, similarityFloor, requested, requiredPropertyTerm, withPrefixes = false),
                terms, queryVec, requested, keywordWeight,
            )
        }
        var hits = HybridQuery.rerank(
            raw.withinRequestedScope(), terms, TERM_COVERAGE_WEIGHT,
            score = { it.score }, text = { it.sectionPath + " " + it.text },
            anchors = HybridQuery.anchorTerms(queryText),
        ).map { (citation, score) -> citation.copy(score = score) }.take(topK)
        Log.i(TAG, "search ns=${namespace.takeLast(18)} q='${queryText.take(80)}' terms=$terms floor=$similarityFloor -> ${hits.size} hits " +
            hits.take(5).joinToString { "p${it.pageNumber}@${"%.3f".format(it.score)}" })
        hits.forEachIndexed { i, c ->
            if (i < 5) Log.i(TAG, "  #$i p${c.pageNumber} c${c.chunkIndex} score=${"%.3f".format(c.score)} '${c.text.take(120).replace('\n', ' ')}'")
        }
        return hits
    }

    private suspend fun executeSearch(
        docHash: String,
        namespace: String,
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
            .addFilterNamespaces(namespace)
            .addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE)
            .setResultCountPerPage(topK)
            .addProjection(PdfChunkDocument.SCHEMA_TYPE, PROJECTION)
            .build()
        Log.d(TAG, "query=$query")
        val results = session.search(query, spec)
        try {
            val page = results.nextPageAsync.await()
            return page.map { r ->
                val g: GenericDocument = r.genericDocument
                citation(g, docHash, r.rankingSignal)
            }
        } finally {
            results.close()
        }
    }

    /** A few stored chunks, no ranking — used to prove the namespace is not empty. */
    suspend fun sampleChunks(docHash: String, limit: Int = 3): List<Citation> {
        val spec = SearchSpec.Builder()
            .addFilterNamespaces(activeNamespace(docHash))
            .addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE)
            .setRankingStrategy(SearchSpec.RANKING_STRATEGY_CREATION_TIMESTAMP)
            .setResultCountPerPage(limit)
            .addProjection(PdfChunkDocument.SCHEMA_TYPE, PROJECTION)
            .build()
        val results = session.search("", spec)
        try {
            return results.nextPageAsync.await().map { r ->
                val g = r.genericDocument
                citation(g, docHash, 0.0)
            }
        } finally {
            results.close()
        }
    }

    suspend fun getCitation(indexNamespace: String, chunkId: String, docHash: String = ""): Citation? {
        val result = session.getByDocumentIdAsync(
            GetByDocumentIdRequest.Builder(indexNamespace).addIds(chunkId).build(),
        ).await()
        val g = result.successes[chunkId] ?: return null
        return citation(g, docHash.ifBlank { g.getPropertyString("docHash") ?: g.namespace }, 0.0)
    }

    /** Legacy helper for V1 tests/indices only. Production V2.1 callers pass the namespace. */
    suspend fun getCitation(chunkId: String): Citation? {
        val legacyNamespace = chunkId.substringBefore(':')
        return getCitation(legacyNamespace, chunkId, legacyNamespace)
    }

    suspend fun getChunks(manifest: DocumentStructureManifest, ids: List<String>): List<Citation> {
        manifest.validate()
        val found = LinkedHashMap<String, Citation>()
        for (batch in ids.chunked(GET_BATCH)) {
            val result = try {
                session.getByDocumentIdAsync(
                    GetByDocumentIdRequest.Builder(manifest.indexNamespace).addIds(batch).build(),
                ).await()
            } catch (t: Throwable) {
                throw ManifestIntegrityError("Direct section fetch failed", t)
            }
            result.successes.forEach { (id, document) ->
                found[id] = citation(document, manifest.documentHash, 0.0)
            }
        }
        val missing = ids.filterNot(found::containsKey)
        if (missing.isNotEmpty()) {
            throw ManifestIntegrityError("Manifest references ${missing.size} missing chunks")
        }
        return ids.map { found.getValue(it) }
    }

    fun loadManifest(docHash: String): DocumentStructureManifest? = manifests.load(docHash)

    fun publishManifest(manifest: DocumentStructureManifest) = manifests.publish(manifest)

    fun activeNamespace(docHash: String): String = manifests.load(docHash)?.indexNamespace ?: docHash

    /** Deletes every chunk of one document (used for delete and for rolling back a cancelled index). */
    suspend fun removeDocument(docHash: String) {
        val namespaces = session.namespacesAsync.await().filter { it == docHash || it.startsWith("$docHash:") }
        for (namespace in namespaces) removeNamespace(namespace, flush = false)
        manifests.delete(docHash)
        session.requestFlushAsync().await()
    }

    /**
     * Removes every namespace of [docHash] except [keep]. Used after publishing a new index so an
     * unreadable legacy manifest (older index version) cannot leave orphaned chunks behind.
     */
    suspend fun removeStaleNamespaces(docHash: String, keep: String): List<String> {
        val stale = session.namespacesAsync.await()
            .filter { (it == docHash || it.startsWith("$docHash:")) && it != keep }
        for (namespace in stale) removeNamespace(namespace, flush = false)
        if (stale.isNotEmpty()) session.requestFlushAsync().await()
        return stale
    }

    suspend fun removeNamespace(namespace: String, flush: Boolean = true) {
        val spec = SearchSpec.Builder().addFilterNamespaces(namespace).addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE).build()
        session.removeAsync("", spec).await()
        if (flush) session.requestFlushAsync().await()
    }

    /** Lists indexed documents by reading one chunk per namespace. */
    suspend fun listDocuments(): List<DocumentInfo> {
        val active = manifests.list().associateBy { it.indexNamespace }
        val namespaces = session.namespacesAsync.await().filter { namespace ->
            namespace in active || ':' !in namespace
        }
        val out = ArrayList<DocumentInfo>()
        for (ns in namespaces) {
            val spec = SearchSpec.Builder()
                .addFilterNamespaces(ns).addFilterSchemas(PdfChunkDocument.SCHEMA_TYPE)
                .setRankingStrategy(SearchSpec.RANKING_STRATEGY_CREATION_TIMESTAMP)
                .setResultCountPerPage(1)
                .addProjection(
                    PdfChunkDocument.SCHEMA_TYPE,
                    listOf("docHash", "docName", "script", "pageCount", "indexVersion"),
                )
                .build()
            val results = session.search("", spec)
            try {
                val first = results.nextPageAsync.await().firstOrNull() ?: continue
                val g = first.genericDocument
                out += DocumentInfo(
                    docHash = g.getPropertyString("docHash")?.ifBlank { ns } ?: ns,
                    displayName = g.getPropertyString("docName") ?: ns,
                    pageCount = g.getPropertyLong("pageCount").toInt(),
                    chunkCount = countChunks(ns),
                    script = g.getPropertyString("script") ?: "",
                    indexVersion = active[ns]?.indexVersion ?: g.getPropertyLong("indexVersion").toInt().coerceAtLeast(1),
                    activeIndexNamespace = ns,
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

    private fun citation(g: GenericDocument, fallbackDocHash: String, score: Double): Citation = Citation(
        chunkId = g.id,
        docHash = g.getPropertyString("docHash")?.ifBlank { fallbackDocHash } ?: fallbackDocHash,
        pageNumber = g.getPropertyLong("pageNumber").toInt(),
        chunkIndex = g.getPropertyLong("chunkIndex").toInt(),
        score = score,
        text = g.getPropertyString("bodyText") ?: g.getPropertyString("text") ?: "",
        indexNamespace = g.namespace,
        sectionId = g.getPropertyString("sectionId") ?: "",
        specificationNumber = g.getPropertyString("specificationNumber") ?: "",
        sectionNumber = g.getPropertyString("sectionNumber") ?: "",
        sectionTitle = g.getPropertyString("sectionTitle") ?: "",
        sectionPath = g.getPropertyString("sectionPath") ?: "",
        contentKind = g.getPropertyString("contentKind") ?: if (g.getPropertyBoolean("isTable")) "TABLE" else "PARAGRAPH",
        continuesFromChunkIndex = g.getPropertyLong("continuesFromChunkIndex").toInt(),
        continuesToChunkIndex = g.getPropertyLong("continuesToChunkIndex").toInt(),
    )

    override fun close() = session.close()

    companion object {
        private const val TAG = "AppSearchVectorStore"
        private const val CANDIDATE_FACTOR = 4
        /** Bonus for full query-term coverage; semantic cosine sums stay dominant (typically 0.6-1.5). */
        private const val TERM_COVERAGE_WEIGHT = 0.35
        const val DB_NAME = "rag_chunks"
        private const val GET_BATCH = 100
        private val PROJECTION = listOf(
            "text", "bodyText", "pageNumber", "chunkIndex", "docHash", "sectionId",
            "specificationNumber", "sectionNumber", "sectionTitle", "sectionPath", "contentKind",
            "continuesFromChunkIndex", "continuesToChunkIndex", "isTable",
        )

        val REQUIRED = listOf(
            Features.SCHEMA_EMBEDDING_PROPERTY_CONFIG,
            Features.SCHEMA_EMBEDDING_QUANTIZATION,
            Features.SEARCH_SPEC_SEARCH_STRING_PARAMETERS,
            Features.SEARCH_SPEC_ADVANCED_RANKING_EXPRESSION,
            Features.LIST_FILTER_QUERY_LANGUAGE,
        )
        val ALL_CHECKED = REQUIRED + listOf(
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
            val store = AppSearchVectorStore(session, DocumentStructureManifestStore(context))
            store.setSchema()
            return store
        }
    }
}
