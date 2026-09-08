package com.example.pdfgemmarag.inference.llm

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.example.pdfgemmarag.core.model.GpuMarker
import com.example.pdfgemmarag.core.model.ModelPaths
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
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
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
    private val onUnrecoverableNativeTimeout: (String) -> Unit,
) : Closeable {

    val backendName: String
    private val engine: Engine
    private val active = AtomicReference<Conversation?>(null)
    private val closed = AtomicBoolean(false)
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
    inner class Generation internal constructor(
        private val cancelledFlag: AtomicBoolean,
        private val cancelAction: () -> Unit,
    ) {
        val cancelled: Boolean get() = cancelledFlag.get()

        fun cancel() {
            if (cancelledFlag.compareAndSet(false, true)) cancelAction()
        }
    }

    /**
     * Starts a streaming turn. Only one generation runs at a time; callers must cancel and await
     * the previous terminal callback before starting another.
     * The returned handle can cancel; [sink] callbacks arrive after native cancel/close, which are
     * time-boxed so a GPU-prefill hang cannot strand the next question.
     */
    fun generate(
        systemInstruction: String,
        history: List<QaPair>,
        userMessage: String,
        sink: TokenSink,
        temperature: Double = 0.0,
        maxOutputTokens: Int = DEFAULT_MAX_OUTPUT_TOKENS,
    ): Generation {
        check(engine.isInitialized()) { "engine not initialised" }
        check(active.get() == null) { "A previous generation is still finishing" }

        val initial = ArrayList<Message>()
        for (qa in history.takeLast(MAX_HISTORY_TURNS)) {
            initial += Message.user(qa.question)
            initial += Message.model(qa.answer)
        }
        val config = ConversationConfig(
            systemInstruction = Contents.of(systemInstruction),
            initialMessages = initial,
            samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = temperature, seed = 0),
            maxOutputToken = maxOutputTokens,
            thinkingConfig = ThinkingConfig(false, 0),
        )
        val tCreate = SystemClock.elapsedRealtime()
        val conversation = engine.createConversation(config)
        Log.i(TAG, "createConversation ${SystemClock.elapsedRealtime() - tCreate} ms")
        if (!active.compareAndSet(null, conversation)) {
            runCatching { conversation.close() }
            error("A previous generation is still finishing")
        }
        val finished = AtomicBoolean(false)
        val cancelledFlag = AtomicBoolean(false)
        Log.i(TAG, "generate start backend=$backendName promptChars=${userMessage.length} history=${history.size}")

        val watchdog = Executors.newSingleThreadScheduledExecutor { r ->
            Thread(r, "gemma-watchdog").apply { isDaemon = true }
        }
        val firstTokenSeen = AtomicBoolean(false)
        val idleFuture = AtomicReference<ScheduledFuture<*>?>(null)
        val assembled = StringBuilder()
        val recoveryRequested = AtomicBoolean(false)

        fun requestProcessRecovery(reason: String) {
            if (!recoveryRequested.compareAndSet(false, true)) return
            Log.e(TAG, "native runtime quarantined: $reason")
            onUnrecoverableNativeTimeout(reason)
        }

        fun releaseConversation(cancelProcess: Boolean): Boolean {
            // cancelProcess can hang for the entire GPU prefill. Skip it until a token has
            // arrived. Any later native timeout quarantines this process; no second JNI call is
            // made against a possibly still-running conversation.
            val invokeCancel = cancelProcess && firstTokenSeen.get()
            if (invokeCancel) {
                val completed = runCatching {
                    TimedNative.run(CANCEL_PROCESS_TIMEOUT_SEC, "cancelProcess") { conversation.cancelProcess() }
                }.onFailure { Log.w(TAG, "cancelProcess: ${it.message}") }.getOrDefault(true)
                if (!completed) {
                    requestProcessRecovery("cancelProcess timed out after ${CANCEL_PROCESS_TIMEOUT_SEC}s")
                    return false
                }
            }
            val closed = runCatching {
                TimedNative.run(CONVERSATION_CLOSE_TIMEOUT_SEC, "conversation-close") { conversation.close() }
            }.onFailure { Log.w(TAG, "conversation close: ${it.message}") }.getOrDefault(true)
            if (!closed) {
                requestProcessRecovery("conversation close timed out after ${CONVERSATION_CLOSE_TIMEOUT_SEC}s")
                return false
            }
            return true
        }

        fun notifySink(cancelled: Boolean, error: Throwable?) {
            active.compareAndSet(conversation, null)
            try {
                if (error != null) sink.onError(error) else sink.onDone(cancelled)
            } catch (t: Throwable) {
                Log.w(TAG, "sink failed", t)
            }
        }

        fun terminal(cancelled: Boolean, error: Throwable? = null, cancelProcess: Boolean = false) {
            if (!finished.compareAndSet(false, true)) return
            watchdog.shutdownNow()
            idleFuture.getAndSet(null)?.cancel(false)
            try {
                closer.execute {
                    if (releaseConversation(cancelProcess)) {
                        notifySink(cancelled, error)
                    }
                }
            } catch (_: RejectedExecutionException) {
                requestProcessRecovery("conversation cleanup executor was unavailable")
            }
        }

        val generation = Generation(cancelledFlag) {
            if (firstTokenSeen.get()) {
                terminal(cancelled = true, cancelProcess = true)
            } else {
                // Pixel 10 GPU: close()/cancelProcess() during prefill kills :inference.
                // Give the callback a short chance to arrive. If native prefill remains stuck, the
                // only safe cancellation boundary is recycling this isolated process.
                Log.i(TAG, "cancel during prefill; waiting briefly for a callback")
                watchdog.schedule({
                    if (!firstTokenSeen.get() && !finished.get() && cancelledFlag.get()) {
                        if (finished.compareAndSet(false, true)) {
                            idleFuture.getAndSet(null)?.cancel(false)
                            requestProcessRecovery(
                                "cancelled prefill produced no callback after ${PREFILL_CANCEL_RECOVERY_TIMEOUT_SEC}s",
                            )
                        }
                    }
                }, PREFILL_CANCEL_RECOVERY_TIMEOUT_SEC, TimeUnit.SECONDS)
            }
        }

        watchdog.schedule({
            if (!firstTokenSeen.get() && !finished.get() && !generation.cancelled &&
                finished.compareAndSet(false, true)
            ) {
                Log.w(TAG, "no first token after ${FIRST_TOKEN_TIMEOUT_SEC}s")
                idleFuture.getAndSet(null)?.cancel(false)
                requestProcessRecovery(
                    "first token timed out after ${FIRST_TOKEN_TIMEOUT_SEC}s",
                )
            }
        }, FIRST_TOKEN_TIMEOUT_SEC, TimeUnit.SECONDS)

        conversation.sendMessageAsync(
            Message.user(userMessage),
            object : MessageCallback {
                override fun onMessage(message: Message) {
                    val full = message.contents.toString()
                    if (full.isEmpty() || finished.get()) return
                    if (generation.cancelled) {
                        firstTokenSeen.set(true)
                        terminal(cancelled = true, cancelProcess = true)
                        return
                    }
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
                    firstTokenSeen.set(true)
                    idleFuture.getAndSet(
                        watchdog.schedule({
                            if (!finished.get() && !generation.cancelled) {
                                terminal(
                                    cancelled = false,
                                    error = IllegalStateException("Gemma stopped responding before completing the answer."),
                                    cancelProcess = true,
                                )
                            }
                        }, STALLED_GENERATION_TIMEOUT_SEC, TimeUnit.SECONDS),
                    )?.cancel(false)
                    Log.i(TAG, "token +${delta.length} total=${assembled.length}")
                    sink.onToken(delta)
                }

                override fun onDone() {
                    Log.i(TAG, "sdk onDone cancelled=${generation.cancelled}")
                    terminal(generation.cancelled)
                }

                override fun onError(throwable: Throwable) {
                    Log.w(TAG, "sdk onError", throwable)
                    if (throwable is CancellationException || generation.cancelled) terminal(cancelled = true)
                    else terminal(cancelled = false, error = throwable)
                }
            },
        )
        return generation
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        val task = closer.submit {
            active.get()?.let { conv ->
                val cancelled = runCatching {
                    TimedNative.run(CANCEL_PROCESS_TIMEOUT_SEC, "cancelProcess") { conv.cancelProcess() }
                }.onFailure { Log.w(TAG, "engine close cancelProcess: ${it.message}") }.getOrDefault(true)
                if (!cancelled) {
                    onUnrecoverableNativeTimeout("engine cancelProcess timed out after ${CANCEL_PROCESS_TIMEOUT_SEC}s")
                    return@submit
                }
                val conversationClosed = runCatching {
                    TimedNative.run(CONVERSATION_CLOSE_TIMEOUT_SEC, "conversation-close") { conv.close() }
                }.onFailure { Log.w(TAG, "engine close conversation: ${it.message}") }.getOrDefault(true)
                if (!conversationClosed) {
                    onUnrecoverableNativeTimeout("engine conversation close timed out after ${CONVERSATION_CLOSE_TIMEOUT_SEC}s")
                    return@submit
                }
                active.compareAndSet(conv, null)
            }
            if (engine.isInitialized()) {
                val engineClosed = runCatching {
                    TimedNative.run(ENGINE_CLOSE_TIMEOUT_SEC, "engine-close") { engine.close() }
                }.onFailure { Log.w(TAG, "engine close: ${it.message}") }.getOrDefault(true)
                if (!engineClosed) {
                    onUnrecoverableNativeTimeout("engine close timed out after ${ENGINE_CLOSE_TIMEOUT_SEC}s")
                }
            }
        }
        runCatching { task.get(60, TimeUnit.SECONDS) }.onFailure { Log.e(TAG, "engine close did not finish", it) }
        closer.shutdownNow()
    }

    companion object {
        private const val TAG = "GemmaEngine"
        const val MAX_HISTORY_TURNS = 4
        const val DEFAULT_MAX_OUTPUT_TOKENS = 512
        const val FIRST_TOKEN_TIMEOUT_SEC = 45L
        const val STALLED_GENERATION_TIMEOUT_SEC = 30L
        const val PREFILL_CANCEL_RECOVERY_TIMEOUT_SEC = 10L
        const val CANCEL_PROCESS_TIMEOUT_SEC = 3L
        const val CONVERSATION_CLOSE_TIMEOUT_SEC = 10L
        const val ENGINE_CLOSE_TIMEOUT_SEC = 15L
        fun gpuMarker(context: Context): File = GpuMarker.file(context)

        /** Warm only after a successful initialize, never merely because a partial cache exists. */
        fun hasWarmGpuCache(context: Context): Boolean {
            val model = File(context.filesDir, "models").listFiles()
                ?.firstOrNull { it.name.endsWith(ModelPaths.LLM_EXTENSION, ignoreCase = true) }
                ?: return false
            return GpuMarker.hasReadyCache(context, model)
        }

        val SYSTEM_INSTRUCTION = """
            You are a precise assistant that answers questions about a PDF document using only the excerpts supplied in each message.
            Rules:
            - Use only the excerpts. Never rely on outside knowledge.
            - Cite every fact with its page in square brackets, e.g. [Page 12]. Cite multiple pages as [Page 3][Page 7].
            - If the excerpts do not answer the question, reply that the document does not contain that information.
            - Be concise. Keep numbers, names, dates and table values exactly as written.
            - Never round, normalize, or infer a numeric value. Copy every number exactly from an excerpt.
            - Answer in the same language as the question.
        """.trimIndent()
    }
}
