package com.example.pdfgemmarag.inference.llm

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.example.pdfgemmarag.core.model.GpuMarker
import com.example.pdfgemmarag.core.model.QaPair
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import java.io.Closeable
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Gemma 4 on LiteRT-LM.
 *
 * Backend selection: GPU unless `filesDir/gpu_disabled.marker` exists. The marker is written by the
 * :ui watchdog when this process dies during GPU initialisation, so a driver crash converges on CPU
 * after one restart instead of looping. Every question gets a fresh [Conversation] seeded with the
 * system instruction and a short, chunk-free history, so the KV cache never accumulates stale
 * retrieved context across turns.
 */
class GemmaEngine(
    private val context: Context,
    val modelPath: String,
    private val allowGpu: Boolean,
    private val maxNumTokens: Int = 8192,
) : Closeable {

    val backendName: String
    private val engine: Engine
    private val active = AtomicReference<Conversation?>(null)

    init {
        val gpuDisabled = gpuMarker(context).exists()
        val useGpu = allowGpu && !gpuDisabled
        backendName = if (useGpu) "GPU" else "CPU"
        val cacheDir = File(context.cacheDir, "litertlm").apply { mkdirs() }
        val backend: Backend = if (useGpu) Backend.GPU() else Backend.CPU()
        engine = Engine(
            EngineConfig(
                modelPath = modelPath,
                backend = backend,
                maxNumTokens = maxNumTokens,
                cacheDir = cacheDir.absolutePath,
            ),
        )
        Log.i(TAG, "GemmaEngine created: backend=$backendName gpuMarker=$gpuDisabled model=$modelPath")
    }

    /** Blocking; 10-120 s on first GPU init (shader compile) then a few seconds with a warm cache. */
    fun initialize() {
        val t0 = SystemClock.elapsedRealtime()
        engine.initialize()
        Log.i(TAG, "engine initialised on $backendName in ${SystemClock.elapsedRealtime() - t0} ms")
    }

    val isInitialized: Boolean get() = engine.isInitialized()

    interface TokenSink {
        fun onToken(text: String)
        fun onDone(cancelled: Boolean)
        fun onError(t: Throwable)
    }

    /** Handle for a running generation. */
    inner class Generation internal constructor(private val conversation: Conversation) {
        @Volatile var cancelled = false
            private set

        fun cancel() {
            cancelled = true
            runCatching { conversation.cancelProcess() }.onFailure { Log.w(TAG, "cancelProcess: ${it.message}") }
            runCatching { conversation.close() }.onFailure { Log.w(TAG, "close after cancel: ${it.message}") }
        }
    }

    /**
     * Starts a streaming turn. Only one generation runs at a time; a second call cancels the first.
     * The returned handle can cancel; [sink] callbacks arrive on a LiteRT-LM worker thread.
     */
    fun generate(
        systemInstruction: String,
        history: List<QaPair>,
        userMessage: String,
        sink: TokenSink,
        temperature: Double = 0.2,
    ): Generation {
        check(engine.isInitialized()) { "engine not initialised" }
        active.getAndSet(null)?.let { old -> runCatching { old.cancelProcess() }; runCatching { old.close() } }

        val initial = ArrayList<Message>()
        for (qa in history.takeLast(MAX_HISTORY_TURNS)) {
            initial += Message.user(qa.question)
            initial += Message.model(qa.answer)
        }
        val config = ConversationConfig(
            systemInstruction = Contents.of(systemInstruction),
            initialMessages = initial,
            samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = temperature, seed = 0),
            thinkingConfig = ThinkingConfig(false, 0),
        )
        val conversation = engine.createConversation(config)
        active.set(conversation)
        val generation = Generation(conversation)
        val done = CountDownLatch(1)
        Log.i(TAG, "generate start backend=$backendName promptChars=${userMessage.length} history=${history.size}")

        val watchdog = java.util.concurrent.Executors.newSingleThreadScheduledExecutor { r ->
            Thread(r, "gemma-watchdog").apply { isDaemon = true }
        }
        val firstTokenMs = java.util.concurrent.atomic.AtomicLong(-1)
        watchdog.schedule({
            if (firstTokenMs.get() < 0 && !generation.cancelled && done.count > 0) {
                Log.w(TAG, "no first token after ${FIRST_TOKEN_TIMEOUT_SEC}s — cancelling")
                generation.cancel()
                sink.onError(IllegalStateException("Generation timed out waiting for the first token. Tap Stop and try a shorter question."))
                done.countDown()
            }
        }, FIRST_TOKEN_TIMEOUT_SEC, TimeUnit.SECONDS)

        conversation.sendMessageAsync(
            Message.user(userMessage),
            object : MessageCallback {
                override fun onMessage(message: Message) {
                    val text = message.contents.toString()
                    if (text.isNotEmpty() && !generation.cancelled) {
                        firstTokenMs.compareAndSet(-1, SystemClock.elapsedRealtime())
                        sink.onToken(text)
                    }
                }

                override fun onDone() {
                    watchdog.shutdownNow()
                    finish(conversation)
                    if (done.count > 0) {
                        done.countDown()
                        sink.onDone(generation.cancelled)
                    }
                }

                override fun onError(throwable: Throwable) {
                    watchdog.shutdownNow()
                    finish(conversation)
                    if (done.count == 0L) return
                    done.countDown()
                    if (throwable is CancellationException || generation.cancelled) sink.onDone(cancelled = true)
                    else sink.onError(throwable)
                }
            },
        )
        return generation
    }

    private fun finish(conversation: Conversation) {
        active.compareAndSet(conversation, null)
        runCatching { conversation.close() }
    }

    fun cancelActive() {
        active.getAndSet(null)?.let { conv ->
            runCatching { conv.cancelProcess() }
            runCatching { conv.close() }
        }
    }

    override fun close() {
        active.getAndSet(null)?.let { runCatching { it.cancelProcess() }; runCatching { it.close() } }
        runCatching { if (engine.isInitialized()) engine.close() }
    }

    companion object {
        private const val TAG = "GemmaEngine"
        const val MAX_HISTORY_TURNS = 4
        const val FIRST_TOKEN_TIMEOUT_SEC = 60L
        fun gpuMarker(context: Context): File = GpuMarker.file(context)

        /** Warm-cache detection: LiteRT-LM writes compiled GPU kernels under cacheDir. */
        fun hasWarmGpuCache(context: Context): Boolean =
            File(context.cacheDir, "litertlm").listFiles()?.isNotEmpty() == true

        val SYSTEM_INSTRUCTION = """
            You are a precise assistant that answers questions about a PDF document using only the excerpts supplied in each message.
            Rules:
            - Use only the excerpts. Never rely on outside knowledge.
            - Cite every fact with its page in square brackets, e.g. [Page 12]. Cite multiple pages as [Page 3][Page 7].
            - If the excerpts do not answer the question, reply that the document does not contain that information.
            - Be concise. Keep numbers, names, dates and table values exactly as written.
            - Answer in the same language as the question.
        """.trimIndent()
    }
}
