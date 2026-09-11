package com.example.pdfgemmarag.eval

import android.os.PowerManager
import android.util.Log
import androidx.appsearch.app.EmbeddingVector
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import com.example.pdfgemmarag.inference.service.AiInferenceService
import com.example.pdfgemmarag.inference.service.ThermalMonitor
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.PdfChunkDocument
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.random.Random

/**
 * Indexes the eval corpus with synthetic embeddings (keyword + ranking path) and reports page-hit@5.
 * When EmbeddingGemma is installed, [com.example.pdfgemmarag.inference.service.SelfTest] runs the
 * same dataset with real query vectors.
 */
@RunWith(AndroidJUnit4::class)
class RetrievalEvalDeviceTest {

    private lateinit var store: AppSearchVectorStore
    private val ns = RetrievalEvalDataset.DOC_HASH + "-${System.currentTimeMillis()}"

    @Before
    fun open(): Unit = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        store = AppSearchVectorStore.open(ctx, "eval_harness_db", enforceProcess = false)
        val now = System.currentTimeMillis()
        val docs = RetrievalEvalDataset.chunks().map { c ->
            PdfChunkDocument().apply {
                namespace = ns
                id = "$ns:${c.chunkIndex}"
                creationTimestampMillis = now
                text = c.text
                pageNumber = c.page
                chunkIndex = c.chunkIndex
                docName = RetrievalEvalDataset.DOC_NAME
                script = c.lang
                pageCount = 250
                embedding = EmbeddingVector(rand(c.chunkIndex), EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
            }
        }
        store.putChunks(docs)
    }

    @After
    fun close(): Unit = runBlocking {
        runCatching { store.removeDocument(ns) }
        store.close()
    }

    @Test
    fun pageHitAt5OverFullDataset(): Unit = runBlocking {
        val report = RetrievalEval.run(
            store = store,
            docHash = ns,
            embed = { rand(it.hashCode()) },
            topK = 5,
            similarityFloor = 0.0,
            keywordWeight = 0.2,
        )
        Log.i("EVAL", report.toString())
        assertEquals(210, report.results.size)
        RetrievalEvalDataset.LANGS.forEach { lang ->
            val row = report.byLanguage.first { it.lang == lang }
            assertEquals(30, row.n)
            assertTrue("$lang hit@5 too low: $row", row.hitAt5 >= 0.7)
        }
        assertTrue("overall hit@5 too low: $report", report.overallHitAt5 >= 0.75)
    }

    /** Frozen 210-question comparison; records measured quality and latency before tuning defaults. */
    @Test
    fun compareFrozenCandidateStrategies(): Unit = runBlocking {
        data class Strategy(
            val name: String,
            val semanticWeight: Double,
            val keywordWeight: Double,
            val localRerankWeight: Double,
        )
        val strategies = listOf(
            Strategy("dense_only", 1.0, 0.0, 0.0),
            Strategy("lexical_only", 0.0, 1.0, 0.35),
            Strategy("hybrid_default", 1.0, 0.05, 0.35),
            Strategy("hybrid_balanced", 1.0, 0.10, 0.35),
            Strategy("hybrid_keyword_strong", 1.0, 0.20, 0.35),
        )
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        check(ModelPaths.embeddingReady(ctx)) { "EmbeddingGemma is required for the measured comparison" }
        val measuredNs = "$ns-real"
        val questions = RetrievalEvalDataset.questions()
        val embedder = EmbeddingGemmaEmbedder(
            ctx, ModelPaths.embeddingModel(ctx), ModelPaths.embeddingTokenizer(ctx), preferGpu = true,
        )
        var tokenAudit = JSONArray()
        var denseBruteForceAudit = JSONObject()
        val (rows, thermalStart) = try {
            val now = System.currentTimeMillis()
            val corpusChunks = RetrievalEvalDataset.chunks()
            val documentVectors = LinkedHashMap<Int, FloatArray>(corpusChunks.size)
            val docs = corpusChunks.map { chunk ->
                val vector = embedder.embedDocument(chunk.text)
                documentVectors[chunk.chunkIndex] = vector
                PdfChunkDocument().apply {
                    namespace = measuredNs
                    id = "$measuredNs:${chunk.chunkIndex}"
                    creationTimestampMillis = now
                    text = chunk.text
                    pageNumber = chunk.page
                    chunkIndex = chunk.chunkIndex
                    docName = RetrievalEvalDataset.DOC_NAME
                    script = chunk.lang
                    pageCount = 250
                    embedding = EmbeddingVector(vector, EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
                }
            }
            store.putChunks(docs)
            val queryVectors = questions.associate { it.question to embedder.embedQuery(it.question) }
            val oracleResults = questions.map { question ->
                val queryVector = queryVectors.getValue(question.question)
                val pages = corpusChunks
                    .sortedByDescending { chunk ->
                        EmbeddingGemmaEmbedder.cosine(queryVector, documentVectors.getValue(chunk.chunkIndex))
                    }
                    .take(5)
                    .map { it.page }
                RetrievalEval.QuestionResult(
                    id = question.id,
                    lang = question.lang,
                    question = question.question,
                    expectedPages = question.expectedPages,
                    retrievedPages = pages,
                    hitAt1 = RetrievalEval.pageHitAtK(pages, question.expectedPages, 1),
                    hitAt5 = RetrievalEval.pageHitAtK(pages, question.expectedPages, 5),
                    reciprocalRank = RetrievalEval.reciprocalRank(pages, question.expectedPages),
                )
            }
            val oracle = RetrievalEval.summarise(oracleResults, 0)
            denseBruteForceAudit = JSONObject()
                .put("questions", oracle.results.size)
                .put("hitAt1", oracle.overallHitAt1)
                .put("hitAt5", oracle.overallHitAt5)
                .put("mrr", oracle.overallMrr)
                .put("perLanguage", JSONArray(oracle.byLanguage.map { lang ->
                    JSONObject().put("language", lang.lang).put("questions", lang.n)
                        .put("hitAt1", lang.hitAt1).put("hitAt5", lang.hitAt5).put("mrr", lang.mrr)
                }))
            tokenAudit = JSONArray(RetrievalEvalDataset.LANGS.map { language ->
                val languageQuestions = questions.filter { it.lang == language }
                val languageChunks = RetrievalEvalDataset.chunks().filter { it.lang == language }
                val questionTokens = languageQuestions.map { embedder.tokenCount("", it.question) }
                val documentTokens = languageChunks.map { embedder.tokenCount("", it.text) }
                JSONObject().put("language", language)
                    .put("meanQuestionTokens", questionTokens.average())
                    .put("maxQuestionTokens", questionTokens.maxOrNull() ?: 0)
                    .put("meanDocumentTokens", documentTokens.average())
                    .put("maxDocumentTokens", documentTokens.maxOrNull() ?: 0)
            })
            // Warm every strategy, then rotate order across three passes. A single sequential pass
            // makes the first strategy pay AppSearch/JIT cache warm-up and produces misleading latency.
            strategies.forEach { strategy ->
                RetrievalEval.run(
                    store, measuredNs, { queryVectors.getValue(it) }, 5, 0.0,
                    strategy.keywordWeight, questions, strategy.semanticWeight, strategy.localRerankWeight,
                )
            }
            val thermal = ctx.getSystemService(PowerManager::class.java).currentThermalStatus
            val reports = strategies.associate { it.name to ArrayList<RetrievalEval.EvalReport>() }
            repeat(3) { pass ->
                val rotated = strategies.drop(pass) + strategies.take(pass)
                rotated.forEach { strategy ->
                    reports.getValue(strategy.name) += RetrievalEval.run(
                        store, measuredNs, { queryVectors.getValue(it) }, 5, 0.0,
                        strategy.keywordWeight, questions, strategy.semanticWeight, strategy.localRerankWeight,
                    )
                }
            }
            strategies.map { strategy ->
                val samples = reports.getValue(strategy.name)
                val report = samples.first()
                val elapsedSamples = samples.map { it.elapsedMs }.sorted()
                val queryLatencies = samples.flatMap { it.results }.map { it.latencyMs }.sorted()
                Log.i("EVAL", "${strategy.name} latency=$elapsedSamples\n$report")
                JSONObject()
                    .put("name", strategy.name)
                    .put("semanticWeight", strategy.semanticWeight)
                    .put("keywordWeight", strategy.keywordWeight)
                    .put("localRerankWeight", strategy.localRerankWeight)
                    .put("questions", report.results.size)
                    .put("hitAt1", report.overallHitAt1)
                    .put("hitAt5", report.overallHitAt5)
                    .put("mrr", report.overallMrr)
                    .put("elapsedMsSamples", JSONArray(elapsedSamples))
                    .put("medianElapsedMs", elapsedSamples[elapsedSamples.size / 2])
                    .put("medianQueryLatencyMs", percentile(queryLatencies, 0.50))
                    .put("p95QueryLatencyMs", percentile(queryLatencies, 0.95))
                    .put("perLanguage", JSONArray(report.byLanguage.map { lang ->
                        JSONObject().put("language", lang.lang).put("questions", lang.n)
                            .put("hitAt1", lang.hitAt1).put("hitAt5", lang.hitAt5).put("mrr", lang.mrr)
                    }))
            } to thermal
        } finally {
            embedder.close()
            runCatching { store.removeDocument(measuredNs) }
        }
        val output = File(
            requireNotNull(InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null)),
            "retrieval_strategy_comparison.json",
        )
        val packageInfo = ctx.packageManager.getPackageInfo(ctx.packageName, 0)
        output.writeText(JSONObject()
            .put("dataset", "acme-210-v1")
            .put("device", android.os.Build.MODEL)
            .put("androidRelease", android.os.Build.VERSION.RELEASE)
            .put("appVersion", packageInfo.versionName ?: "")
            .put("appVersionCode", packageInfo.longVersionCode)
            .put("embeddingSignature", EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
            .put("embeddingMode", "real EmbeddingGemma document and query vectors")
            .put("denseBruteForceAudit", denseBruteForceAudit)
            .put("tokenAudit", tokenAudit)
            .put("thermalStatusStart", thermalStart)
            .put("thermalStatusEnd", ctx.getSystemService(PowerManager::class.java).currentThermalStatus)
            .put("thermalOverride", File(ctx.filesDir, ThermalMonitor.GATE_DISABLED_MARKER).exists())
            .put("speculativeDecodingOverride", File(ctx.filesDir, AiInferenceService.MTP_MARKER).exists())
            .put("strategies", JSONArray(rows))
            .toString(2))
        assertTrue("incomplete strategy comparison; report=$output", rows.all { it.getInt("questions") == 210 })
    }

    @Test
    fun captureEmbeddingTensorContract() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        check(ModelPaths.embeddingReady(ctx)) { "EmbeddingGemma is required" }
        val embedder = EmbeddingGemmaEmbedder(
            ctx, ModelPaths.embeddingModel(ctx), ModelPaths.embeddingTokenizer(ctx), preferGpu = true,
        )
        val report = try {
            JSONObject()
                .put("modelFile", ModelPaths.embeddingModel(ctx).name)
                .put("modelBytes", ModelPaths.embeddingModel(ctx).length())
                .put("backend", embedder.backend)
                .put("sequenceLength", embedder.sequenceLength)
                .put("requestedOutputDimensions", embedder.outputDim)
                .put("outputValueCounts", JSONArray(embedder.outputValueCounts))
        } finally {
            embedder.close()
        }
        val output = File(requireNotNull(ctx.getExternalFilesDir(null)), "embedding_tensor_contract.json")
        output.writeText(report.toString(2))
        Log.i("EVAL", "embedding tensor contract=$report")
        assertTrue("embedding export has no output", report.getJSONArray("outputValueCounts").length() > 0)
    }

    @Test
    fun compareEmbeddingBackends() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        check(ModelPaths.embeddingReady(ctx)) { "EmbeddingGemma is required" }
        fun measure(preferGpu: Boolean): JSONObject {
            val embedder = EmbeddingGemmaEmbedder(
                ctx, ModelPaths.embeddingModel(ctx), ModelPaths.embeddingTokenizer(ctx), preferGpu = preferGpu,
            )
            return try {
                val query = embedder.embedQuery("Which planet is known as the Red Planet?")
                val mars = embedder.embedDocument("Mars is known as the Red Planet because of its reddish appearance.")
                val venus = embedder.embedDocument("Venus is Earth's hot and cloudy neighboring planet.")
                val dogs = embedder.embedDocument("Dogs are domesticated animals that often live with people.")
                JSONObject()
                    .put("backend", embedder.backend)
                    .put("queryMars", EmbeddingGemmaEmbedder.cosine(query, mars).toDouble())
                    .put("queryVenus", EmbeddingGemmaEmbedder.cosine(query, venus).toDouble())
                    .put("queryDogs", EmbeddingGemmaEmbedder.cosine(query, dogs).toDouble())
                    .put("marsVenus", EmbeddingGemmaEmbedder.cosine(mars, venus).toDouble())
                    .put("marsDogs", EmbeddingGemmaEmbedder.cosine(mars, dogs).toDouble())
            } finally {
                embedder.close()
            }
        }
        val report = JSONObject().put("preferredGpu", measure(true)).put("cpu", measure(false))
        val output = File(requireNotNull(ctx.getExternalFilesDir(null)), "embedding_backend_comparison.json")
        output.writeText(report.toString(2))
        Log.i("EVAL", "embedding backend comparison=$report")
    }

    private fun percentile(sorted: List<Long>, fraction: Double): Long {
        if (sorted.isEmpty()) return 0
        return sorted[((sorted.size - 1) * fraction).toInt().coerceIn(sorted.indices)]
    }

    private fun rand(seed: Int): FloatArray {
        val r = Random(seed)
        val v = FloatArray(512) { r.nextFloat() - 0.5f }
        EmbeddingGemmaEmbedder.l2Normalize(v)
        return v
    }
}
