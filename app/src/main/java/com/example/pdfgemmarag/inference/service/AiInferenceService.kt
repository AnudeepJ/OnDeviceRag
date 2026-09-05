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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
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
    private val generations = ConcurrentHashMap<Long, GemmaEngine.Generation>()
    private val cancelledGenerations = ConcurrentHashMap.newKeySet<Long>()
    @Volatile private var indexingJob: Job? = null
    @Volatile private var installJob: Job? = null

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
                startInstall(File(source), target, sha, size, deleteSource = true, callback = null)
            }
            ACTION_KEEPALIVE -> Unit
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "service destroyed")
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

    // ------------------------------------------------------------------ binder

    private val binder = object : IAiInferenceService.Stub() {

        override fun ping(): Int = 1

        override fun loadEngine(modelPath: String, allowGpu: Boolean, callback: IEngineCallback) {
            scope.launch {
                engineLock.withLock {
                    gemma?.close(); gemma = null
                    currentStatus = EngineStatus(EngineStatus.State.LOADING, modelPath = modelPath, modelName = File(modelPath).name)
                    val useGpu = allowGpu && !GemmaEngine.gpuMarker(this@AiInferenceService).exists()
                    safe { callback.onProgress("Creating engine (${if (useGpu) "GPU" else "CPU"})") }
                    try {
                        val e = GemmaEngine(this@AiInferenceService, modelPath, allowGpu)
                        safe { callback.onProgress("Initialising on ${e.backendName}. First GPU start compiles shaders and can take up to two minutes.") }
                        withContext(Dispatchers.IO) { e.initialize() }
                        gemma = e
                        currentStatus = EngineStatus(
                            EngineStatus.State.READY, backend = e.backendName, modelPath = modelPath,
                            modelName = File(modelPath).name, embedderLoaded = embedder != null, thermalStatus = thermal.status.value,
                        )
                        safe { callback.onReady(currentStatus) }
                    } catch (t: Throwable) {
                        Log.e(TAG, "engine init failed (gpu=$useGpu)", t)
                        if (useGpu) {
                            // A clean exception (not a crash) on GPU: persist the decision and retry on CPU now.
                            GpuMarker.write(this@AiInferenceService, "init exception: ${t.message}")
                            safe { callback.onProgress("GPU initialisation failed; retrying on CPU") }
                            try {
                                val e = GemmaEngine(this@AiInferenceService, modelPath, allowGpu = false)
                                withContext(Dispatchers.IO) { e.initialize() }
                                gemma = e
                                currentStatus = EngineStatus(EngineStatus.State.READY, backend = "CPU", modelPath = modelPath, modelName = File(modelPath).name)
                                safe { callback.onReady(currentStatus) }
                                return@withLock
                            } catch (t2: Throwable) {
                                Log.e(TAG, "CPU init failed too", t2)
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

        override fun unloadEngine() {
            scope.launch {
                engineLock.withLock {
                    gemma?.close(); gemma = null
                    currentStatus = EngineStatus(EngineStatus.State.UNLOADED)
                }
            }
        }

        override fun getEngineStatus(): EngineStatus = this@AiInferenceService.currentStatus.copy(
            embedderLoaded = embedder != null, thermalStatus = thermal.status.value,
        )

        override fun ask(docHash: String, question: String, history: List<QaPair>, callback: IStreamCallback): Long {
            val id = generationIds.incrementAndGet()
            scope.launch {
                val e = gemma
                if (e == null || !e.isInitialized) { safe { callback.onError(id, "Model is not loaded") }; return@launch }
                try {
                    val terminal = AtomicBoolean(false)
                    val listener = object : AnswerQuestionUseCase.Listener {
                        override fun onRetrieved(citations: List<Citation>) { safe { callback.onRetrieved(id, citations) } }
                        override fun onToken(text: String) {
                            try { callback.onToken(id, text) } catch (e: RemoteException) { onClientGone(id) }
                        }
                        override fun onDone(stats: GenerationStats) {
                            terminal.set(true); cancelledGenerations.remove(id); generations.remove(id)
                            safe { callback.onDone(id, stats) }
                        }
                        override fun onError(message: String) {
                            terminal.set(true); cancelledGenerations.remove(id); generations.remove(id)
                            safe { callback.onError(id, message) }
                        }
                    }
                    // Empty docHash = plain chat: no embedder or AppSearch needed, so it works before any model beyond Gemma is installed.
                    if (cancelledGenerations.remove(id)) {
                        safe { callback.onError(id, "Stopped") }
                        return@launch
                    }
                    val handle = if (docHash == PlainChatUseCase.PLAIN_CHAT_DOC_HASH) {
                        PlainChatUseCase(e).start(id, question, history, listener)
                    } else {
                        AnswerQuestionUseCase(requireEmbedder(), requireStore(), e).start(id, docHash, question, history, listener)
                    }
                    if (cancelledGenerations.remove(id)) {
                        handle?.cancel()
                        safe { callback.onError(id, "Stopped") }
                        return@launch
                    }
                    if (handle != null && !terminal.get()) generations[id] = handle
                } catch (t: Throwable) {
                    Log.e(TAG, "ask failed", t)
                    safe { callback.onError(id, t.message ?: t.javaClass.simpleName) }
                }
            }
            return id
        }

        override fun cancelGeneration(generationId: Long) {
            Log.i(TAG, "cancel requested id=$generationId active=${generations.keys}")
            cancelledGenerations.add(generationId)
            val handle = generations.remove(generationId)
            if (handle != null) {
                handle.cancel()
            }
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

        override fun listDocuments(): List<DocumentInfo> = runBlocking { runCatching { requireStore().listDocuments() }.getOrDefault(emptyList()) }

        override fun getCitation(chunkId: String): Citation? = runBlocking { runCatching { requireStore().getCitation(chunkId) }.getOrNull() }

        override fun installModel(sourcePath: String, targetFileName: String, expectedSha256: String, expectedSize: Long, deleteSource: Boolean, callback: IInstallCallback?) {
            startInstall(File(sourcePath), targetFileName, expectedSha256, expectedSize, deleteSource, callback)
        }

        override fun getDiagnostics(): String = runBlocking { diagnostics() }

        override fun runSelfTest(): String = runBlocking {
            SelfTest(this@AiInferenceService, { requireEmbedder() }, { requireStore() }).run()
        }

        override fun probeRetrieval(docHash: String): String = runBlocking {
            val store = requireStore()
            val embedder = if (ModelPaths.embeddingReady(this@AiInferenceService)) requireEmbedder() else null
            if (embedder == null) Log.w(TAG, "probe without embedder; keyword path only")
            val report = RetrievalProbe.run(store, { q -> embedder?.embedQuery(q) ?: FloatArray(512) }, docHash.ifBlank { null })
            runCatching { File(filesDir, "retrieval_probe.txt").writeText(report) }
            report
        }
    }

    private fun onClientGone(generationId: Long) {
        Log.w(TAG, "client died mid-stream; cancelling generation $generationId")
        generations.remove(generationId)?.cancel()
    }

    private fun startInstall(source: File, targetName: String, sha: String, size: Long, deleteSource: Boolean, callback: IInstallCallback?) {
        if (installJob?.isActive == true) { safe { callback?.onFailed("Another install is running") }; return }
        installJob = scope.launch(Dispatchers.IO) {
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
                Log.e(TAG, "install failed", t)
                safe { callback?.onFailed(t.message ?: t.javaClass.simpleName) }
            } finally {
                installJob = null
                stopForegroundIfIdle()
            }
        }
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
        if (indexingJob?.isActive != true && installJob?.isActive != true) {
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
        const val ACTION_INSTALL_MODEL = "com.example.pdfgemmarag.action.INSTALL_MODEL"
        const val ACTION_KEEPALIVE = "com.example.pdfgemmarag.action.KEEPALIVE"
        const val EXTRA_SOURCE = "source"
        const val EXTRA_TARGET_NAME = "targetName"
        const val EXTRA_SHA256 = "sha256"
        const val EXTRA_SIZE = "size"

    }
}
