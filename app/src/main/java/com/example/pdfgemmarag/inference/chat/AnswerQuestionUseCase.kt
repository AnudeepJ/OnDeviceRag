package com.example.pdfgemmarag.inference.chat

import android.os.SystemClock
import android.util.Log
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.GenerationStats
import com.example.pdfgemmarag.core.model.QaPair
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import com.example.pdfgemmarag.inference.llm.ContextAssembler
import com.example.pdfgemmarag.inference.llm.GemmaEngine
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.HybridQuery

/** Retrieve -> assemble -> generate for one question against one document. */
class AnswerQuestionUseCase(
    private val embedder: EmbeddingGemmaEmbedder,
    private val store: AppSearchVectorStore,
    private val engine: GemmaEngine,
    private val assembler: ContextAssembler = ContextAssembler(),
    private val topK: Int = 12,
    private val similarityFloor: Double = 0.3,
    private val keywordWeight: Double = 0.05,
) {

    interface Listener {
        fun onRetrieved(citations: List<Citation>)
        fun onToken(text: String)
        fun onDone(stats: GenerationStats)
        fun onError(message: String)
    }

    /** Runs retrieval synchronously and starts generation; returns the handle used for cancellation. */
    suspend fun start(generationId: Long, docHash: String, question: String, history: List<QaPair>, listener: Listener): GemmaEngine.Generation? {
        val t0 = SystemClock.elapsedRealtime()
        val queryVec = embedder.embedQuery(question)
        val terms = HybridQuery.keywordTerms(question)
        Log.i(TAG, "ask hash=${docHash.take(12)} q='${question.take(120)}' terms=$terms dim=${queryVec.size}")
        val ranked = store.search(docHash, question, queryVec, topK, similarityFloor, keywordWeight)
        val assembled = assembler.assemble(question, ranked)
        Log.i(TAG, "retrieved ${ranked.size} -> ${assembled.citations.size} chunks (~${assembled.approxTokens} tokens) in ${SystemClock.elapsedRealtime() - t0} ms")
        assembled.citations.forEachIndexed { i, c ->
            Log.i(TAG, "  cite[$i] p${c.pageNumber} c${c.chunkIndex} score=${"%.3f".format(c.score)} '${c.text.take(80).replace('\n', ' ')}'")
        }
        // Citations cross Binder without text; the UI fetches text on demand by chunkId.
        listener.onRetrieved(assembled.citations.map { it.copy(text = "") })

        if (assembled.citations.isEmpty()) {
            Log.w(TAG, "EMPTY retrieval — returning canned refusal (index empty or query matched nothing)")
            val msg = "The document does not appear to contain information about that."
            listener.onToken(msg)
            listener.onDone(
                GenerationStats(generationId, 0, SystemClock.elapsedRealtime() - t0, msg.length, 0.0, 0, 0, engine.backendName, false),
            )
            return null
        }

        var firstToken = -1L
        var chars = 0
        if (history.isNotEmpty()) Log.i(TAG, "ignoring ${history.size} history turns for grounded ask")
        // Do not replay prior Q&A into the KV cache: a wrong turn poisons the next one, and
        // extra history plus 4k excerpts is what stalled GPU prefill so Stop never returned.
        return engine.generate(
            systemInstruction = GemmaEngine.SYSTEM_INSTRUCTION,
            history = emptyList(),
            userMessage = assembled.prompt,
            sink = object : GemmaEngine.TokenSink {
                override fun onToken(text: String) {
                    if (firstToken < 0) firstToken = SystemClock.elapsedRealtime()
                    chars += text.length
                    listener.onToken(text)
                }

                override fun onDone(cancelled: Boolean) {
                    val end = SystemClock.elapsedRealtime()
                    val genMs = if (firstToken > 0) (end - firstToken).coerceAtLeast(1) else 1
                    listener.onDone(
                        GenerationStats(
                            generationId = generationId,
                            timeToFirstTokenMs = if (firstToken > 0) firstToken - t0 else -1,
                            totalMs = end - t0,
                            outputChars = chars,
                            approxTokensPerSecond = ContextAssembler.estimateTokens("x".repeat(chars)) * 1000.0 / genMs,
                            retrievedChunks = assembled.citations.size,
                            contextTokensApprox = assembled.approxTokens,
                            backend = engine.backendName,
                            cancelled = cancelled,
                        ),
                    )
                }

                override fun onError(t: Throwable) {
                    Log.e(TAG, "generation failed", t)
                    listener.onError(t.message ?: t.javaClass.simpleName)
                }
            },
        )
    }

    companion object { private const val TAG = "AnswerQuestionUseCase" }
}
