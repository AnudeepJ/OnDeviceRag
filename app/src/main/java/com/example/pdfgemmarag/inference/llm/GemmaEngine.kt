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
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
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
    private val closer = Executors.newSingleThreadExecutor { r -> Thread(r, "gemma-close").apply { isDaemon = true } }

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
            closer.execute {
                runCatching { conversation.cancelProcess() }.onFailure { Log.w(TAG, "cancelProcess: ${it.message}") }
                runCatching { conversation.close() }.onFailure { Log.w(TAG, "close after cancel: ${it.message}") }
            }
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
        active.getAndSet(null)?.let { old -> closer.execute { runCatching { old.close() } } }

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
        val tCreate = SystemClock.elapsedRealtime()
        val conversation = engine.createConversation(config)
        Log.i(TAG, "createConversation ${SystemClock.elapsedRealtime() - tCreate} ms")
        active.set(conversation)
        val generation = Generation(conversation)
        val finished = AtomicBoolean(false)
        Log.i(TAG, "generate start backend=$backendName promptChars=${userMessage.length} history=${history.size}")

        val watchdog = Executors.newSingleThreadScheduledExecutor { r ->
            Thread(r, "gemma-watchdog").apply { isDaemon = true }
        }
        val firstTokenMs = AtomicLong(-1)
        val lastTokenMs = AtomicLong(-1)
        val assembled = StringBuilder()

        fun complete(cancelled: Boolean) {
            if (!finished.compareAndSet(false, true)) return
            watchdog.shutdownNow()
            active.compareAndSet(conversation, null)
            sink.onDone(cancelled)
        }

        fun fail(t: Throwable) {
            if (!finished.compareAndSet(false, true)) return
            watchdog.shutdownNow()
            active.compareAndSet(conversation, null)
            sink.onError(t)
        }

        watchdog.schedule({
            if (firstTokenMs.get() < 0 && !finished.get()) {
                Log.w(TAG, "no first token after ${FIRST_TOKEN_TIMEOUT_SEC}s — unlocking UI (async cancel)")
                if (backendName == "GPU") GpuMarker.write(context, "first-token timeout on GPU")
                fail(IllegalStateException("Gemma did not start in time. The model will use CPU next. Tap Stop if needed, then ask again."))
                generation.cancel()
            }
        }, FIRST_TOKEN_TIMEOUT_SEC, TimeUnit.SECONDS)

        // LiteRT-LM often delivers the full answer via onMessage and never calls onDone.
        watchdog.scheduleAtFixedRate({
            val last = lastTokenMs.get()
            if (last < 0 || finished.get()) return@scheduleAtFixedRate
            val idle = SystemClock.elapsedRealtime() - last
            val text = synchronized(assembled) { assembled.toString() }
            val need = if (looksComplete(text)) IDLE_COMPLETE_MS else IDLE_CONTINUE_MS
            if (idle >= need) {
                Log.i(TAG, "idle ${idle}ms after last token (complete=${looksComplete(text)}) — closing turn")
                // Do not cancelProcess() here: that wedges the GPU engine so the next
                // question never emits a first token.
                complete(cancelled = false)
            }
        }, 1, 1, TimeUnit.SECONDS)

        conversation.sendMessageAsync(
            Message.user(userMessage),
            object : MessageCallback {
                override fun onMessage(message: Message) {
                    val full = message.contents.toString()
                    if (full.isEmpty() || generation.cancelled || finished.get()) return
                    val delta = synchronized(assembled) {
                        val soFar = assembled.toString()
                        val piece = when {
                            full.startsWith(soFar) -> full.substring(soFar.length)
                            soFar.endsWith(full) -> ""
                            else -> full
                        }
                        if (piece.isNotEmpty()) {
                            if (!full.startsWith(soFar) && soFar.isNotEmpty()) assembled.clear()
                            assembled.append(piece)
                        }
                        piece
                    }
                    if (delta.isEmpty()) return
                    firstTokenMs.compareAndSet(-1, SystemClock.elapsedRealtime())
                    lastTokenMs.set(SystemClock.elapsedRealtime())
                    Log.i(TAG, "token +${delta.length} total=${assembled.length}")
                    sink.onToken(delta)
                }

                override fun onDone() {
                    Log.i(TAG, "sdk onDone cancelled=${generation.cancelled}")
                    complete(generation.cancelled)
                }

                override fun onError(throwable: Throwable) {
                    Log.w(TAG, "sdk onError", throwable)
                    if (throwable is CancellationException || generation.cancelled) complete(cancelled = true)
                    else fail(throwable)
                }
            },
        )
        return generation
    }

    fun cancelActive() {
        active.getAndSet(null)?.let { conv ->
            closer.execute {
                runCatching { conv.cancelProcess() }
                runCatching { conv.close() }
            }
        }
    }

        override fun close() {
        active.getAndSet(null)?.let { conv -> closer.execute { runCatching { conv.close() } } }
        closer.execute { runCatching { if (engine.isInitialized()) engine.close() } }
        closer.shutdown()
    }

    companion object {
        private const val TAG = "GemmaEngine"
        const val MAX_HISTORY_TURNS = 4
        const val FIRST_TOKEN_TIMEOUT_SEC = 45L
        const val IDLE_COMPLETE_MS = 2500L
        const val IDLE_CONTINUE_MS = 8000L

        internal fun looksComplete(text: String): Boolean {
            val t = text.trim()
            if (t.length < 12) return false
            return t.endsWith('.') || t.endsWith('!') || t.endsWith('?') ||
                t.endsWith(']') || t.endsWith("].") || t.contains("[Page")
        }
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
