package com.example.pdfgemmarag.inference.service

import android.app.ActivityManager
import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.DeadObjectException
import android.os.IBinder
import android.os.RemoteException
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.example.pdfgemmarag.R
import com.example.pdfgemmarag.RagApplication
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.DocumentInfo
import com.example.pdfgemmarag.core.model.EngineStatus
import com.example.pdfgemmarag.core.model.GenerationStats
import com.example.pdfgemmarag.core.model.GpuMarker
import com.example.pdfgemmarag.core.model.IndexingProgress
import com.example.pdfgemmarag.core.model.InferenceRecoveryMarker
import com.example.pdfgemmarag.core.model.ModelInstaller
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.core.model.QaPair
import com.example.pdfgemmarag.core.process.ProcessGuard
import com.example.pdfgemmarag.inference.chat.AnswerQuestionUseCase
import com.example.pdfgemmarag.inference.chat.PlainChatUseCase
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import com.example.pdfgemmarag.inference.index.IndexPdfUseCase
import com.example.pdfgemmarag.inference.llm.GemmaEngine
import com.example.pdfgemmarag.inference.ocr.MlKitOcr
import com.example.pdfgemmarag.inference.pdf.AprysePdfExtractor
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.RetrievalProbe
import com.example.pdfgemmarag.ui.download.ModelDownloadReceiver
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * The only component in the `:inference` process. Owns every native runtime (LiteRT-LM, LiteRT,
 * Apryse, AppSearch/Icing) and exposes them to `:ui` through AIDL. Long jobs (indexing, model
 * install) promote the service to a `dataSync` foreground service so the OS keeps it alive when the
 * activity is backgrounded.
 */
class AiInferenceService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var thermal: ThermalMonitor

    // Lazily-created runtimes (all confined to this process).
    private val runtimeLock = Mutex()
    private var store: AppSearchVectorStore? = null
    private var embedder: EmbeddingGemmaEmbedder? = null
    private var pdf: AprysePdfExtractor? = null
    private var ocr: MlKitOcr? = null

    @Volatile private var gemma: GemmaEngine? = null
    @Volatile private var currentStatus = EngineStatus(EngineStatus.State.UNLOADED)
    private val engineLock = Mutex()

    private val generationIds = AtomicLong(System.currentTimeMillis())
    /** Service-side admission gate: UI state is never the authority for native conversation safety. */
    private val generationMutex = Mutex()
    private val generations = ConcurrentHashMap<Long, GemmaEngine.Generation>()
    private val knownGenerations = ConcurrentHashMap.newKeySet<Long>()
    private val cancelledGenerations = ConcurrentHashMap.newKeySet<Long>()
    private val recoveryRequested = AtomicBoolean(false)
    @Volatile private var indexingJob: Job? = null
    private val installMutex = Mutex()
    private val pendingInstalls = AtomicInteger(0)

    override fun onCreate() {
        super.onCreate()
        ProcessGuard.requireInferenceProcess(this)
        thermal = ThermalMonitor(this)
        Log.i(TAG, "service created in ${ProcessGuard.currentProcessName(this)} (pid ${android.os.Process.myPid()})")
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_INSTALL_MODEL -> {
                val source = intent.getStringExtra(EXTRA_SOURCE) ?: return START_NOT_STICKY
                val target = intent.getStringExtra(EXTRA_TARGET_NAME) ?: return START_NOT_STICKY
                val sha = intent.getStringExtra(EXTRA_SHA256) ?: ""
                val size = intent.getLongExtra(EXTRA_SIZE, -1)
                val downloadId = intent.getLongExtra(EXTRA_DOWNLOAD_ID, -1)
                startInstall(File(source), target, sha, size, deleteSource = true, callback = null, downloadId = downloadId)
            }
            ACTION_KEEPALIVE -> Unit
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "service destroyed")
        requestCancelAllGenerations()
        scope.cancel()
        thermal.release()
        gemma?.close()
        embedder?.close()
        ocr?.close()
        store?.close()
        super.onDestroy()
    }

    // ------------------------------------------------------------------ runtimes

    private suspend fun requireStore(): AppSearchVectorStore = runtimeLock.withLock {
        store ?: AppSearchVectorStore.open(this).also { store = it }
    }

    private suspend fun requireEmbedder(): EmbeddingGemmaEmbedder = runtimeLock.withLock {
        embedder ?: run {
            val model = ModelPaths.embeddingModel(this)
            val tok = ModelPaths.embeddingTokenizer(this)
            check(model.exists() && tok.exists()) { "Embedding model not installed (${model.name}, ${tok.name})" }
            EmbeddingGemmaEmbedder(this, model, tok, preferGpu = !GemmaEngine.gpuMarker(this).exists() && !GpuMarker.embedderGpuDisabled(this)).also { embedder = it }
        }
    }

    private suspend fun requirePdf(): AprysePdfExtractor = runtimeLock.withLock {
        pdf ?: AprysePdfExtractor(this).also { pdf = it }
    }

    private suspend fun requireOcr(): MlKitOcr = runtimeLock.withLock { ocr ?: MlKitOcr().also { ocr = it } }

    private fun createGemmaEngine(modelPath: String, allowGpu: Boolean): GemmaEngine = GemmaEngine(
        context = this,
        modelPath = modelPath,
        allowGpu = allowGpu,
        onUnrecoverableNativeTimeout = ::recycleInferenceProcess,
    )

    private fun recycleInferenceProcess(reason: String) {
        if (!recoveryRequested.compareAndSet(false, true)) return
        Log.e(TAG, "recycling inference process after unrecoverable native timeout: $reason")
        InferenceRecoveryMarker.write(this, reason)
        android.os.Process.killProcess(android.os.Process.myPid())
    }

    // ------------------------------------------------------------------ binder

    private val binder = object : IAiInferenceService.Stub() {

        override fun ping(): Int = 1

        override fun loadEngine(modelPath: String, allowGpu: Boolean, callback: IEngineCallback) {
            scope.launch {
                val modelFile = File(modelPath)
                if (!modelFile.isFile || !modelFile.canRead()) {
                    safe { callback.onFailed("Model file is missing or unreadable. Import the complete .litertlm file again.") }
                    return@launch
                }
                if (!modelFile.name.endsWith(ModelPaths.LLM_EXTENSION, ignoreCase = true) ||
                    modelFile.length() < ModelPaths.MIN_LLM_BYTES
                ) {
                    safe { callback.onFailed("Model file is incomplete or is not a .litertlm model (${modelFile.length()} bytes).") }
                    return@launch
                }
                requestCancelAllGenerations()
                generationMutex.withLock {
                    engineLock.withLock {
                        gemma?.close(); gemma = null
                        currentStatus = EngineStatus(EngineStatus.State.LOADING, modelPath = modelPath, modelName = File(modelPath).name)
                        val useGpu = allowGpu && !GemmaEngine.gpuMarker(this@AiInferenceService).exists()
                        safe { callback.onProgress("Creating engine (${if (useGpu) "GPU" else "CPU"})") }
                        var candidate: GemmaEngine? = null
                        try {
                            val e = createGemmaEngine(modelPath, allowGpu).also { candidate = it }
                            safe { callback.onProgress("Initialising on ${e.backendName}. First GPU start compiles shaders and can take up to two minutes.") }
                            withContext(Dispatchers.IO) { e.initialize() }
                            if (e.backendName == "GPU") GpuMarker.markCacheReady(this@AiInferenceService, modelFile)
                            gemma = e
                            candidate = null
                            currentStatus = EngineStatus(
                                EngineStatus.State.READY, backend = e.backendName, modelPath = modelPath,
                                modelName = File(modelPath).name, embedderLoaded = embedder != null, thermalStatus = thermal.status.value,
                            )
                            safe { callback.onReady(currentStatus) }
                        } catch (t: Throwable) {
                            candidate?.close(); candidate = null
                            Log.e(TAG, "engine init failed (gpu=$useGpu)", t)
                            if (useGpu) {
                                // A clean exception (not a crash) on GPU: persist the decision and retry on CPU now.
                                GpuMarker.write(this@AiInferenceService, "init exception: ${t.message}")
                                safe { callback.onProgress("GPU initialisation failed; retrying on CPU") }
                                try {
                                    val e = createGemmaEngine(modelPath, allowGpu = false).also { candidate = it }
                                    withContext(Dispatchers.IO) { e.initialize() }
                                    gemma = e
                                    candidate = null
                                    currentStatus = EngineStatus(EngineStatus.State.READY, backend = "CPU", modelPath = modelPath, modelName = File(modelPath).name)
                                    safe { callback.onReady(currentStatus) }
                                    return@withLock
                                } catch (t2: Throwable) {
                                    candidate?.close(); candidate = null
                                    Log.e(TAG, "CPU init failed too", t2)
                                    // Both backends failing points to the model/runtime, not a proven GPU
                                    // incompatibility. Do not permanently poison future valid models.
                                    GemmaEngine.gpuMarker(this@AiInferenceService).delete()
                                    currentStatus = EngineStatus(EngineStatus.State.FAILED, message = t2.message ?: "init failed")
                                    safe { callback.onFailed(t2.message ?: t2.javaClass.simpleName) }
                                    return@withLock
                                }
                            }
                            currentStatus = EngineStatus(EngineStatus.State.FAILED, message = t.message ?: "init failed")
                            safe { callback.onFailed(t.message ?: t.javaClass.simpleName) }
                        }
                    }
                }
            }
        }

        override fun unloadEngine() {
            scope.launch {
                requestCancelAllGenerations()
                generationMutex.withLock {
                    engineLock.withLock {
                        gemma?.close(); gemma = null
                        currentStatus = EngineStatus(EngineStatus.State.UNLOADED)
                    }
                }
            }
        }

        override fun getEngineStatus(): EngineStatus = this@AiInferenceService.currentStatus.copy(
            embedderLoaded = embedder != null, thermalStatus = thermal.status.value,
        )

        override fun ask(
            docHash: String,
            activeIndexNamespace: String,
            question: String,
            history: List<QaPair>,
            callback: IStreamCallback,
        ): Long {
            val id = generationIds.incrementAndGet()
            knownGenerations.add(id)
            scope.launch {
                generationMutex.withLock {
                    if (cancelledGenerations.remove(id)) {
                        knownGenerations.remove(id)
                        safe { callback.onError(id, "Stopped") }
                        return@withLock
                    }
                    val e = gemma
                    if (e == null || !e.isInitialized) {
                        knownGenerations.remove(id)
                        safe { callback.onError(id, "Model is not loaded") }
                        return@withLock
                    }
                    val terminal = CompletableDeferred<Unit>()
                    val terminalDelivered = AtomicBoolean(false)

                    fun finish(block: () -> Unit) {
                        if (!terminalDelivered.compareAndSet(false, true)) return
                        generations.remove(id)
                        knownGenerations.remove(id)
                        cancelledGenerations.remove(id)
                        safe(block)
                        terminal.complete(Unit)
                    }

                    val listener = object : AnswerQuestionUseCase.Listener {
                        override fun onRetrieved(citations: List<Citation>) { safe { callback.onRetrieved(id, citations) } }
                        override fun onToken(text: String) {
                            try { callback.onToken(id, text) } catch (e: RemoteException) { onClientGone(id) }
                        }
                        override fun onDone(stats: GenerationStats) {
                            finish { callback.onDone(id, stats) }
                        }
                        override fun onError(message: String) {
                            finish { callback.onError(id, message) }
                        }
                    }
                    try {
                        // Empty docHash = plain chat: no embedder or AppSearch needed, so it works before any model beyond Gemma is installed.
                        val handle = if (docHash == PlainChatUseCase.PLAIN_CHAT_DOC_HASH) {
                            PlainChatUseCase(e).start(id, question, history, listener)
                        } else {
                            AnswerQuestionUseCase({ requireEmbedder() }, requireStore(), e).start(
                                id, docHash, activeIndexNamespace, question, history, listener,
                            )
                        }
                        if (handle != null && !terminal.isCompleted) {
                            generations[id] = handle
                            if (cancelledGenerations.remove(id)) handle.cancel()
                            // GemmaEngine clears its active conversation before invoking the
                            // terminal callback, so releasing this mutex makes the next turn safe.
                            val completed = withTimeoutOrNull(GENERATION_GATE_TIMEOUT_MS) { terminal.await() }
                            if (completed == null) {
                                Log.e(TAG, "generation $id missing terminal callback; draining cancel")
                                generations[id]?.cancel()
                                val drained = withTimeoutOrNull(GENERATION_CANCEL_DRAIN_MS) { terminal.await() }
                                if (drained == null) {
                                    recycleInferenceProcess("generation $id did not reach a terminal callback")
                                }
                            }
                        }
                    } catch (t: Throwable) {
                        Log.e(TAG, "ask failed", t)
                        finish { callback.onError(id, t.message ?: t.javaClass.simpleName) }
                    }
                }
            }
            return id
        }

        override fun cancelGeneration(generationId: Long) {
            Log.i(TAG, "cancel requested id=$generationId active=${generations.keys}")
            requestGenerationCancellation(generationId)
        }

        override fun indexDocument(docHash: String, pdfPath: String, displayName: String, callback: IIndexingCallback) {
            if (indexingJob?.isActive == true) { safe { callback.onFailed(docHash, "Another document is already being indexed") }; return }
            indexingJob = scope.launch {
                Log.i(TAG, "index start hash=$docHash name=$displayName path=$pdfPath")
                startForegroundCompat(buildNotification("Indexing $displayName", "Preparing", 0, 0, indeterminate = true))
                try {
                    val useCase = IndexPdfUseCase(requirePdf(), requireOcr(), requireEmbedder(), requireStore(), awaitCool = { thermal.awaitCool() })
                    var lastNotify = 0L
                    val info = useCase.run(docHash, pdfPath, displayName) { p ->
                        val paused = thermal.isThrottled
                        safe { callback.onProgress(p.copy(paused = paused)) }
                        val now = System.currentTimeMillis()
                        if (now - lastNotify > 1000) {
                            lastNotify = now
                            updateNotification(buildNotification("Indexing $displayName", "${p.stage.name.lowercase()} ${p.current}/${p.total}", p.current, p.total, false))
                        }
                    }
                    safe { callback.onCompleted(info) }
                } catch (t: kotlinx.coroutines.CancellationException) {
                    safe { callback.onCancelled(docHash) }
                } catch (t: Throwable) {
                    safe { callback.onFailed(docHash, t.message ?: t.javaClass.simpleName) }
                } finally {
                    indexingJob = null
                    stopForegroundIfIdle()
                }
            }
        }

        override fun cancelIndexing() { indexingJob?.cancel() }

        override fun isIndexing(): Boolean = indexingJob?.isActive == true

        override fun deleteDocument(docHash: String) {
            scope.launch { runCatching { requireStore().removeDocument(docHash) }.onFailure { Log.e(TAG, "delete failed", it) } }
        }

        override fun listDocuments(callback: IDocumentsCallback) {
            scope.launch {
                runCatching { requireStore().listDocuments() }
                    .onSuccess { documents -> safe { callback.onResult(documents) } }
                    .onFailure { error -> safe { callback.onError(error.message ?: error.javaClass.simpleName) } }
            }
        }

        override fun getCitation(chunkId: String, callback: ICitationCallback) {
            getCitationAsync(callback) { requireStore().getCitation(chunkId) }
        }

        override fun getCitationInNamespace(indexNamespace: String, chunkId: String, callback: ICitationCallback) {
            getCitationAsync(callback) { requireStore().getCitation(indexNamespace, chunkId) }
        }

        override fun installModel(sourcePath: String, targetFileName: String, expectedSha256: String, expectedSize: Long, deleteSource: Boolean, callback: IInstallCallback?) {
            startInstall(File(sourcePath), targetFileName, expectedSha256, expectedSize, deleteSource, callback, downloadId = -1)
        }

        override fun getDiagnostics(callback: ITextResultCallback) {
            textResultAsync(callback) { diagnostics() }
        }

        override fun runSelfTest(callback: ITextResultCallback) {
            textResultAsync(callback) {
                SelfTest(this@AiInferenceService, { requireEmbedder() }, { requireStore() }).run()
            }
        }

        override fun probeRetrieval(docHash: String, callback: ITextResultCallback) {
            textResultAsync(callback) {
                val store = requireStore()
                val embedder = if (ModelPaths.embeddingReady(this@AiInferenceService)) requireEmbedder() else null
                if (embedder == null) Log.w(TAG, "probe without embedder; keyword path only")
                RetrievalProbe.run(store, { q -> embedder?.embedQuery(q) ?: FloatArray(512) }, docHash.ifBlank { null })
                    .also { report -> runCatching { File(filesDir, "retrieval_probe.txt").writeText(report) } }
            }
        }
    }

    private fun getCitationAsync(callback: ICitationCallback, block: suspend () -> Citation?) {
        scope.launch {
            runCatching { block() }
                .onSuccess { citation ->
                    if (citation == null) safe { callback.onNotFound() }
                    else safe { callback.onResult(citation) }
                }
                .onFailure { error -> safe { callback.onError(error.message ?: error.javaClass.simpleName) } }
        }
    }

    private fun textResultAsync(callback: ITextResultCallback, block: suspend () -> String) {
        scope.launch {
            runCatching { block() }
                .onSuccess { value -> safe { callback.onResult(value) } }
                .onFailure { error -> safe { callback.onError(error.message ?: error.javaClass.simpleName) } }
        }
    }

    private fun onClientGone(generationId: Long) {
        Log.w(TAG, "client died mid-stream; cancelling generation $generationId")
        requestGenerationCancellation(generationId)
    }

    private fun requestGenerationCancellation(generationId: Long) {
        if (!knownGenerations.contains(generationId)) return
        cancelledGenerations.add(generationId)
        // Close the contains/add race with a terminal callback so a late stop cannot leave an
        // arbitrary ID in this long-lived process for the remainder of the app session.
        if (!knownGenerations.contains(generationId)) {
            cancelledGenerations.remove(generationId)
            return
        }
        generations[generationId]?.cancel()
    }

    private fun requestCancelAllGenerations() {
        for (id in knownGenerations.toList()) {
            cancelledGenerations.add(id)
            generations[id]?.cancel()
        }
    }

    private fun startInstall(source: File, targetName: String, sha: String, size: Long, deleteSource: Boolean, callback: IInstallCallback?, downloadId: Long) {
        pendingInstalls.incrementAndGet()
        scope.launch(Dispatchers.IO) {
            var installError: String? = null
            try {
                installMutex.withLock {
                    startForegroundCompat(buildNotification("Installing model", targetName, 0, 0, indeterminate = true))
                    try {
                        val target = File(ModelPaths.modelsDir(this@AiInferenceService), targetName)
                        val total = if (size > 0) size else source.length()
                        var last = 0L
                        val hash = ModelInstaller.verifyAndInstall(source, target, sha, size, deleteSource) { copied ->
                            val now = System.currentTimeMillis()
                            if (now - last > 500) {
                                last = now
                                safe { callback?.onProgress(copied, total) }
                                updateNotification(buildNotification("Installing model", "${copied * 100 / total.coerceAtLeast(1)}%", (copied shr 20).toInt(), (total shr 20).toInt(), false))
                            }
                        }
                        safe { callback?.onInstalled(target.absolutePath, hash) }
                    } catch (t: Throwable) {
                        installError = t.message ?: t.javaClass.simpleName
                        Log.e(TAG, "install failed", t)
                        safe { callback?.onFailed(requireNotNull(installError)) }
                    }
                }
            } finally {
                if (downloadId >= 0) notifyDownloadInstallResult(downloadId, installError)
                pendingInstalls.decrementAndGet()
                stopForegroundIfIdle()
            }
        }
    }

    private fun notifyDownloadInstallResult(downloadId: Long, error: String?) {
        val result = Intent(this, ModelDownloadReceiver::class.java)
            .setAction(ACTION_INSTALL_MODEL_RESULT)
            .putExtra(EXTRA_DOWNLOAD_ID, downloadId)
        if (error != null) result.putExtra(EXTRA_INSTALL_ERROR, error)
        sendBroadcast(result)
    }

    private suspend fun diagnostics(): String {
        val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val sb = StringBuilder()
        sb.appendLine("process=${ProcessGuard.currentProcessName(this)} pid=${android.os.Process.myPid()}")
        sb.appendLine("device=${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT} abi=${Build.SUPPORTED_ABIS.joinToString()}")
        sb.appendLine("totalMem=${mi.totalMem shr 20}MB availMem=${mi.availMem shr 20}MB lowMemory=${mi.lowMemory}")
        sb.appendLine("thermal=${thermal.status.value} gpuMarker=${GemmaEngine.gpuMarker(this).exists()} warmGpuCache=${GemmaEngine.hasWarmGpuCache(this)}")
        sb.appendLine("engine=${currentStatus.state} backend=${currentStatus.backend} model=${currentStatus.modelName}")
        sb.appendLine("embedder=${embedder?.let { "${it.backend} seq=${it.sequenceLength} vocab=${it.tokenizer.vocabSize} ${it.tokenizer.normalizerNote}" } ?: "not loaded"}")
        sb.appendLine("apryse=${AprysePdfExtractor.version()} licenseKey=${if (AprysePdfExtractor.licenseKey(this).isBlank()) "MISSING" else "present"}")
        sb.appendLine("models=${ModelPaths.installedLlms(this).joinToString { "${it.name} (${it.length() shr 20}MB)" }}")
        sb.appendLine("embeddingReady=${ModelPaths.embeddingReady(this)}")
        runCatching { requireStore() }.onSuccess { s ->
            sb.appendLine("appsearch features (hybridOk=${s.features.hybridOk}):")
            sb.appendLine(s.features.toString().prependIndent("  "))
            val docs = runCatching { s.listDocuments() }.getOrDefault(emptyList())
            sb.appendLine("indexed docs=${docs.size}")
            docs.forEach { d ->
                sb.appendLine("  ${d.displayName} hash=${d.docHash.take(12)} pages=${d.pageCount} chunks=${d.chunkCount}")
            }
            if (docs.isEmpty()) sb.appendLine("  INDEX EMPTY — if the UI still lists a PDF, AppSearch was wiped or never finished")
        }.onFailure { sb.appendLine("appsearch: ${it.message}") }
        return sb.toString()
    }

    // ------------------------------------------------------------------ foreground plumbing

    private fun startForegroundCompat(n: Notification) {
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0
        ServiceCompat.startForeground(this, NOTIFICATION_ID, n, type)
    }

    private fun updateNotification(n: Notification) {
        getSystemService(android.app.NotificationManager::class.java).notify(NOTIFICATION_ID, n)
    }

    private fun stopForegroundIfIdle() {
        if (indexingJob?.isActive != true && pendingInstalls.get() == 0) {
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        }
    }

    private fun buildNotification(title: String, text: String, progress: Int, max: Int, indeterminate: Boolean): Notification {
        val launch = packageManager.getLaunchIntentForPackage(packageName)
        val pi = launch?.let { PendingIntent.getActivity(this, 0, it, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT) }
        return NotificationCompat.Builder(this, RagApplication.CHANNEL_INDEXING)
            .setSmallIcon(R.drawable.ic_stat_rag)
            .setContentTitle(title)
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(if (indeterminate) 0 else max, progress, indeterminate)
            .apply { if (pi != null) setContentIntent(pi) }
            .build()
    }

    private inline fun safe(block: () -> Unit) {
        try { block() } catch (e: DeadObjectException) { Log.w(TAG, "client dead") } catch (e: RemoteException) { Log.w(TAG, "remote failure: ${e.message}") }
    }

    companion object {
        private const val TAG = "AiInferenceService"
        const val NOTIFICATION_ID = 1001
        private const val GENERATION_GATE_TIMEOUT_MS = 180_000L
        private const val GENERATION_CANCEL_DRAIN_MS =
            (GemmaEngine.CANCEL_PROCESS_TIMEOUT_SEC + GemmaEngine.CONVERSATION_CLOSE_TIMEOUT_SEC + 7L) * 1_000L
        const val ACTION_INSTALL_MODEL = "com.example.pdfgemmarag.action.INSTALL_MODEL"
        const val ACTION_INSTALL_MODEL_RESULT = "com.example.pdfgemmarag.action.INSTALL_MODEL_RESULT"
        const val ACTION_KEEPALIVE = "com.example.pdfgemmarag.action.KEEPALIVE"
        const val EXTRA_SOURCE = "source"
        const val EXTRA_TARGET_NAME = "targetName"
        const val EXTRA_SHA256 = "sha256"
        const val EXTRA_SIZE = "size"
        const val EXTRA_DOWNLOAD_ID = "download_id"
        const val EXTRA_INSTALL_ERROR = "install_error"

    }
}
