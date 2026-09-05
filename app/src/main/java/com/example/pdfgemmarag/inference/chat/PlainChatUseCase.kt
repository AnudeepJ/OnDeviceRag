package com.example.pdfgemmarag.inference.chat

import android.os.SystemClock
import android.util.Log
import com.example.pdfgemmarag.core.model.GenerationStats
import com.example.pdfgemmarag.core.model.QaPair
import com.example.pdfgemmarag.inference.llm.ContextAssembler
import com.example.pdfgemmarag.inference.llm.GemmaEngine

/**
 * Generation without retrieval (`docHash == ""`): lets the user talk to Gemma before any document
 * or embedding model is installed, and is the Phase 2 acceptance path for streaming + cancel.
 */
class PlainChatUseCase(private val engine: GemmaEngine) {

    fun start(generationId: Long, question: String, history: List<QaPair>, listener: AnswerQuestionUseCase.Listener): GemmaEngine.Generation {
        val t0 = SystemClock.elapsedRealtime()
        listener.onRetrieved(emptyList())
        var firstToken = -1L
        var chars = 0
        return engine.generate(
            systemInstruction = SYSTEM_INSTRUCTION,
            history = history,
            userMessage = question,
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
                            retrievedChunks = 0,
                            contextTokensApprox = ContextAssembler.estimateTokens(question),
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

    companion object {
        private const val TAG = "PlainChatUseCase"
        /** Reserved docHash for retrieval-free chat; Room stores its transcript under this key. */
        const val PLAIN_CHAT_DOC_HASH = ""
        val SYSTEM_INSTRUCTION = "You are a helpful, concise assistant running fully on the user's phone. Answer in the language of the question."
    }
}
