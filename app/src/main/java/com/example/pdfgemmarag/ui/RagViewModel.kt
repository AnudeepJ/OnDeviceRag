package com.example.pdfgemmarag.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.pdfgemmarag.RagApplication
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.DocumentInfo
import com.example.pdfgemmarag.core.model.EngineStatus
import com.example.pdfgemmarag.core.model.GenerationStats
import com.example.pdfgemmarag.core.model.GpuMarker
import com.example.pdfgemmarag.core.model.IndexingProgress
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.core.model.QaPair
import com.example.pdfgemmarag.inference.service.IEngineCallback
import com.example.pdfgemmarag.inference.service.IIndexingCallback
import com.example.pdfgemmarag.inference.service.IInstallCallback
import com.example.pdfgemmarag.inference.service.IStreamCallback
import com.example.pdfgemmarag.inference.service.diagnosticsAsync
import com.example.pdfgemmarag.inference.service.getCitationAsync
import com.example.pdfgemmarag.inference.service.getCitationInNamespaceAsync
import com.example.pdfgemmarag.inference.service.probeRetrievalAsync
import com.example.pdfgemmarag.inference.service.runSelfTestAsync
import com.example.pdfgemmarag.ui.data.DocumentEntity
import com.example.pdfgemmarag.ui.data.MessageEntity
import com.example.pdfgemmarag.ui.download.CatalogEntry
import com.example.pdfgemmarag.ui.download.ModelDownloadManager
import com.example.pdfgemmarag.ui.download.ModelCatalog
import com.example.pdfgemmarag.ui.service.EngineWatchdog
import com.example.pdfgemmarag.ui.service.ServiceConnectionManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Single view model for the :ui process; all heavy work is proxied to the :inference service. */
class RagViewModel(app: Application) : AndroidViewModel(app) {

    private val ragApp = app as RagApplication
    private val connection: ServiceConnectionManager = ragApp.serviceConnection
    private val db = ragApp.database
    val downloads: ModelDownloadManager = ragApp.modelDownloads
    val gate = DeviceGate(app)
    private val prefs = app.getSharedPreferences("rag_ui", Context.MODE_PRIVATE)

    data class UiState(
        val serviceAlive: Boolean = false,
        val engine: EngineStatus = EngineStatus(EngineStatus.State.UNLOADED),
        val engineMessage: String = "",
        val installedModels: List<File> = emptyList(),
        val selectedModelPath: String? = null,
        val embeddingReady: Boolean = false,
        val gpuDisabled: Boolean = false,
        val gpuDisabledReason: String? = null,
        val indexing: IndexingProgress? = null,
        val indexingDocName: String? = null,
        val install: Triple<String, Long, Long>? = null, // fileName, copied, total
        val notice: String? = null,
        val diagnostics: String? = null,
        val selfTest: String? = null,
        val probe: String? = null,
        val busy: Boolean = false,
    )

    data class ChatState(
        val docHash: String? = null,
        val generationId: Long = -1,
        val generating: Boolean = false,
        val streamingText: String = "",
        val streamingCitations: List<Citation> = emptyList(),
        val lastStats: GenerationStats? = null,
        val error: String? = null,
    )

    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui
    private val _chat = MutableStateFlow(ChatState())
    val chat: StateFlow<ChatState> = _chat
    private data class ActiveTurn(val clientToken: Long, val docHash: String, @Volatile var generationId: Long = -1)
    private val turnIds = AtomicLong()
    private val turnLock = Any()
    @Volatile private var activeTurn: ActiveTurn? = null
    private val reindexPreviousStatus = HashMap<String, String>()
    private val loadingModel = AtomicReference<String?>(null)

    val documents: StateFlow<List<DocumentEntity>> =
        db.documents().observeAll().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    fun messages(docHash: String): Flow<List<MessageEntity>> = db.messages().observe(docHash)

    private val watchdog: EngineWatchdog = EngineWatchdog(app, connection, viewModelScope) { reason ->
        _ui.update { it.copy(gpuDisabled = true, gpuDisabledReason = reason, notice = "GPU disabled ($reason). Reloading on CPU.") }
        viewModelScope.launch { delay(1500); _ui.value.selectedModelPath?.let { loadEngine(it) } }
    }

    init {
        viewModelScope.launch {
            runCatching { db.documents().markLegacyIndexes(CURRENT_INDEX_VERSION) }
                .onFailure { Log.e(TAG, "Could not mark legacy indexes", it) }
        }
        connection.bind()
        refreshLocalState()
        viewModelScope.launch {
            connection.events.collect { ev ->
                when (ev) {
                    ServiceConnectionManager.Event.Connected -> {
                        _ui.update { it.copy(serviceAlive = true) }
                        syncEngineStatus()
                    }
                    ServiceConnectionManager.Event.Died -> onInferenceDied()
                }
            }
        }
        // Download progress polling while any download is active.
        viewModelScope.launch {
            while (isActive) {
                downloads.refresh()
                refreshLocalState()
                delay(if (downloads.states.value.values.any { it.running }) 1000 else 5000)
            }
        }
    }

    // ------------------------------------------------------------------ models / engine

    fun refreshLocalState() {
        val app = getApplication<Application>()
        _ui.update {
            val installed = ModelPaths.installedLlms(app)
            // A single installed LLM is the natural default.  Remember it so entering chat does
            // not require the user to visit Models and press a second, unrelated "Load" button.
            val remembered = prefs.getString(KEY_MODEL, null)?.takeIf { p -> File(p).exists() }
            it.copy(
                installedModels = installed,
                embeddingReady = ModelPaths.embeddingReady(app),
                gpuDisabled = GpuMarker.exists(app),
                gpuDisabledReason = GpuMarker.reason(app),
                selectedModelPath = it.selectedModelPath ?: remembered ?: installed.singleOrNull()?.absolutePath,
            )
        }
    }

    /** Starts the remembered model on demand.  Loading a 2+ GB model at app launch wastes RAM. */
    private fun loadPreferredModelIfNeeded() {
        if (_ui.value.engine.state == EngineStatus.State.READY || _ui.value.engine.state == EngineStatus.State.LOADING) return
        val path = _ui.value.selectedModelPath
            ?.takeIf { File(it).exists() }
            ?: _ui.value.installedModels.singleOrNull()?.absolutePath
            ?: return
        loadEngine(path)
    }

    private fun syncEngineStatus() = viewModelScope.launch {
        runCatching { connection.await().engineStatus }.onSuccess { s -> _ui.update { it.copy(engine = s) } }
    }

    fun loadEngine(modelPath: String) {
        val app = getApplication<Application>()
        val gpuFallbackRequired = _ui.value.engine.backend == "GPU" && GpuMarker.exists(app)
        if (_ui.value.engine.state == EngineStatus.State.READY &&
            _ui.value.engine.modelPath == modelPath &&
            !gpuFallbackRequired
        ) return
        if (!loadingModel.compareAndSet(null, modelPath)) return
        viewModelScope.launch {
            prefs.edit { putString(KEY_MODEL, modelPath) }
            val gpuAttempt = !GpuMarker.exists(app)
            _ui.update { it.copy(selectedModelPath = modelPath, engine = EngineStatus(EngineStatus.State.LOADING, modelName = File(modelPath).name), engineMessage = "Starting…") }
            watchdog.onLoadStarted(gpuAttempt, File(modelPath))
            try {
                connection.startService()
                connection.await().loadEngine(modelPath, true, object : IEngineCallback.Stub() {
                    override fun onProgress(stage: String) { _ui.update { it.copy(engineMessage = stage) } }
                    override fun onReady(status: EngineStatus) {
                        loadingModel.compareAndSet(modelPath, null)
                        watchdog.onLoadFinished()
                        _ui.update { it.copy(engine = status, engineMessage = "Ready on ${status.backend}", gpuDisabled = GpuMarker.exists(app)) }
                    }
                    override fun onFailed(message: String) {
                        loadingModel.compareAndSet(modelPath, null)
                        watchdog.onLoadFinished()
                        _ui.update { it.copy(engine = EngineStatus(EngineStatus.State.FAILED, message = message), engineMessage = message, gpuDisabled = GpuMarker.exists(app)) }
                    }
                })
            } catch (t: Throwable) {
                loadingModel.compareAndSet(modelPath, null)
                watchdog.onLoadFinished()
                _ui.update { it.copy(engine = EngineStatus(EngineStatus.State.FAILED, message = t.message ?: ""), engineMessage = t.message ?: "load failed") }
            }
        }
    }

    fun unloadEngine() = viewModelScope.launch {
        loadingModel.set(null)
        runCatching { connection.await().unloadEngine() }
        _ui.update { it.copy(engine = EngineStatus(EngineStatus.State.UNLOADED), engineMessage = "") }
    }

    fun retryGpu() {
        GpuMarker.clear(getApplication())
        refreshLocalState()
        _ui.value.selectedModelPath?.let { loadEngine(it) }
    }

    fun deleteModel(file: File) = viewModelScope.launch {
        if (_ui.value.engine.modelPath == file.absolutePath) unloadEngine()
        file.delete(); refreshLocalState()
    }

    fun download(entry: CatalogEntry) {
        if (entry.kind == CatalogEntry.Kind.LLM && !gate.hasSpaceForModel(entry.sizeBytes)) {
            notice("Not enough free storage: need ${(entry.sizeBytes * 2) shr 20} MB for download + install."); return
        }
        if (!entry.downloadable) {
            notice("Download is not configured for ${entry.fileName}. Set MODEL_CDN_BASE_URL and its SHA-256, or import a complete local file.")
            return
        }
        runCatching { downloads.enqueue(entry) }.onFailure { notice(it.message ?: "Could not start download") }
    }

    fun cancelDownload(entry: CatalogEntry) = downloads.cancel(entry)

    fun downloadRequiredModels() {
        val missing = ModelCatalog.requiredEntries.filter {
            !downloads.installed(it) && downloads.states.value[it.id]?.running != true
        }
        if (missing.isEmpty()) { notice("All required models are already installed or downloading."); return }
        val unavailable = missing.firstOrNull { !it.downloadable }
        if (unavailable != null) {
            notice("Model download is not configured. Set MODEL_CDN_BASE_URL, rebuild, or add local model files.")
            return
        }
        // All downloads may coexist, while the largest one is copied into private storage.
        val requiredFree = missing.sumOf { it.sizeBytes } + missing.maxOf { it.sizeBytes } + DeviceGate.SAFETY_MARGIN
        if (gate.freeBytes() < requiredFree) {
            notice("Not enough free storage: need about ${requiredFree shr 20} MB for the required downloads and installation.")
            return
        }
        val failures = missing.mapNotNull { entry ->
            runCatching { downloads.enqueue(entry) }.exceptionOrNull()?.message
        }
        notice(if (failures.isEmpty()) "Downloading ${missing.size} required model files over Wi-Fi." else "Some downloads could not start: ${failures.first()}")
    }

    /** Local `.litertlm`/`.tflite`/`.model` picker: copy through staging, then verify+install in :inference. */
    fun importLocalModel(uri: Uri, displayName: String) = viewModelScope.launch {
        val app = getApplication<Application>()
        val safeDisplayName = File(displayName).name
        val targetName = when {
            safeDisplayName.endsWith(".litertlm", true) -> safeDisplayName
            safeDisplayName.endsWith(".tflite", true) -> ModelPaths.EMBEDDING_MODEL_FILE
            safeDisplayName.endsWith(".model", true) -> ModelPaths.EMBEDDING_TOKENIZER_FILE
            else -> { notice("Unsupported file: $displayName"); return@launch }
        }
        _ui.update { it.copy(busy = true, notice = "Copying $displayName…") }
        var staged: File? = null
        try {
            staged = withContext(Dispatchers.IO) {
                val directory = ModelPaths.modelStagingDir(app)
                directory.listFiles { file -> file.name.endsWith(".incoming") }?.forEach(File::delete)
                val f = File(directory, "${UUID.randomUUID()}-$targetName.incoming")
                val input = app.contentResolver.openInputStream(uri)
                    ?: throw IllegalArgumentException("The selected file could not be opened")
                input.use { source -> f.outputStream().use { source.copyTo(it, 1 shl 20) } }
                val minimumSize = when {
                    targetName.endsWith(".litertlm", true) -> ModelPaths.MIN_LLM_BYTES
                    targetName.endsWith(".tflite", true) -> 1024L * 1024
                    else -> 100L * 1024
                }
                if (f.length() < minimumSize) {
                    val actual = f.length()
                    f.delete()
                    throw IllegalArgumentException("$displayName is incomplete ($actual bytes); expected at least $minimumSize bytes")
                }
                f
            }
            connection.startService()
            val stagedFile = requireNotNull(staged)
            connection.await().installModel(stagedFile.absolutePath, targetName, "", stagedFile.length(), true, object : IInstallCallback.Stub() {
                override fun onProgress(bytesCopied: Long, totalBytes: Long) { _ui.update { it.copy(install = Triple(targetName, bytesCopied, totalBytes)) } }
                    override fun onInstalled(targetPath: String, sha256: String) {
                        _ui.update { it.copy(install = null, busy = false, notice = "Installed $targetName (sha256 ${sha256.take(12)}…)") }
                        refreshLocalState()
                        // The selected Gemma file is ready to use as soon as copying completes.
                        // Embedding assets are deliberately not loaded here; they are lazy-loaded
                        // only when the user indexes a document.
                        if (gate.canLoadLlm && targetName.endsWith(ModelPaths.LLM_EXTENSION, ignoreCase = true)) loadEngine(targetPath)
                    }
                override fun onFailed(message: String) {
                    stagedFile.delete()
                    _ui.update { it.copy(install = null, busy = false, notice = "Install failed: $message. Select the complete .litertlm file and retry.") }
                }
            })
        } catch (t: Throwable) {
            withContext(Dispatchers.IO) { staged?.delete() }
            _ui.update { it.copy(busy = false, notice = "Import failed: ${t.message}") }
        }
    }

    // ------------------------------------------------------------------ documents

    fun importPdf(uri: Uri, displayName: String) = viewModelScope.launch {
        val app = getApplication<Application>()
        if (!_ui.value.embeddingReady) { notice("Install the EmbeddingGemma model and tokenizer first (Models screen)."); return@launch }
        if (!gate.hasSpaceForIndexing()) { notice("Less than 1 GB free; free up storage before indexing."); return@launch }
        if (_ui.value.indexing != null) { notice("Another document is being indexed."); return@launch }
        _ui.update { it.copy(busy = true) }
        try {
            val (hash, file) = withContext(Dispatchers.IO) {
                val staging = File(ModelPaths.stagingDir(app), "import-${System.currentTimeMillis()}.pdf")
                val digest = MessageDigest.getInstance("SHA-256")
                app.contentResolver.openInputStream(uri)!!.use { input ->
                    staging.outputStream().use { out ->
                        val buf = ByteArray(1 shl 20)
                        while (true) { val n = input.read(buf); if (n < 0) break; digest.update(buf, 0, n); out.write(buf, 0, n) }
                    }
                }
                val hash = digest.digest().joinToString("") { "%02x".format(it) }.take(32)
                val target = ModelPaths.pdfFile(app, hash)
                if (target.exists()) staging.delete() else staging.renameTo(target)
                hash to target
            }
            val name = displayName.ifBlank { "document.pdf" }
            db.documents().upsert(DocumentEntity(hash, name, 0, 0, "", System.currentTimeMillis(), STATUS_INDEXING, null, file.absolutePath))
            _ui.update { it.copy(busy = false, indexing = IndexingProgress(hash, IndexingProgress.Stage.PREPARING, 0, 1), indexingDocName = name) }
            // Chat and indexing must not both hold big models on 6-8 GB devices.
            if (gate.chatBlockedWhileIndexing && _ui.value.engine.state == EngineStatus.State.READY) {
                notice("Unloading the chat model while indexing (low-RAM device)."); unloadEngine()
            }
            connection.startService()
            connection.await().indexDocument(hash, file.absolutePath, name, indexingCallback)
        } catch (t: Throwable) {
            Log.e(TAG, "import failed", t)
            _ui.update { it.copy(busy = false, indexing = null, notice = "Import failed: ${t.message}") }
        }
    }

    private val indexingCallback = object : IIndexingCallback.Stub() {
        override fun onProgress(progress: IndexingProgress) { _ui.update { it.copy(indexing = progress) } }
        override fun onCompleted(document: DocumentInfo) {
            viewModelScope.launch {
                reindexPreviousStatus.remove(document.docHash)
                db.documents().get(document.docHash)?.let {
                    db.documents().update(
                        it.copy(
                            pageCount = document.pageCount,
                            chunkCount = document.chunkCount,
                            script = document.script,
                            status = STATUS_READY,
                            indexedAt = System.currentTimeMillis(),
                            indexVersion = document.indexVersion,
                            activeIndexNamespace = document.activeIndexNamespace,
                        ),
                    )
                }
                _ui.update { it.copy(indexing = null, indexingDocName = null, notice = "Indexed ${document.displayName}: ${document.pageCount} pages, ${document.chunkCount} chunks") }
            }
        }
        override fun onCancelled(docHash: String) {
            viewModelScope.launch {
                val previous = reindexPreviousStatus.remove(docHash)
                db.documents().get(docHash)?.let { current ->
                    if (previous != null) db.documents().update(current.copy(status = previous, error = null))
                    else {
                        db.documents().delete(current)
                        ModelPaths.pdfFile(getApplication(), docHash).delete()
                    }
                }
                _ui.update { it.copy(indexing = null, indexingDocName = null, notice = "Indexing cancelled") }
            }
        }
        override fun onFailed(docHash: String, message: String) {
            viewModelScope.launch {
                val previous = reindexPreviousStatus.remove(docHash)
                db.documents().get(docHash)?.let {
                    db.documents().update(it.copy(status = previous ?: STATUS_FAILED, error = if (previous == null) message else null))
                }
                _ui.update { it.copy(indexing = null, indexingDocName = null, notice = "Indexing failed: $message") }
            }
        }
    }

    fun cancelIndexing() = viewModelScope.launch { runCatching { connection.await().cancelIndexing() } }

    fun reindex(doc: DocumentEntity) = viewModelScope.launch {
        if (_ui.value.indexing != null) { notice("Another document is being indexed."); return@launch }
        reindexPreviousStatus[doc.hash] = doc.status
        db.documents().update(doc.copy(status = STATUS_INDEXING, error = null))
        _ui.update { it.copy(indexing = IndexingProgress(doc.hash, IndexingProgress.Stage.PREPARING, 0, 1), indexingDocName = doc.name) }
        connection.startService()
        connection.await().indexDocument(doc.hash, doc.pdfPath, doc.name, indexingCallback)
    }

    fun deleteDocument(doc: DocumentEntity) = viewModelScope.launch {
        runCatching { connection.await().deleteDocument(doc.hash) }
        db.messages().deleteForDocument(doc.hash)
        db.documents().delete(doc)
        File(doc.pdfPath).delete()
        if (_chat.value.docHash == doc.hash) _chat.value = ChatState()
    }

    // ------------------------------------------------------------------ chat

    fun openChat(docHash: String) {
        if (_chat.value.docHash == docHash) return
        if (_chat.value.generating) stopGeneration()
        _chat.value = ChatState(docHash = docHash)
        loadPreferredModelIfNeeded()
    }

    fun ask(question: String) = viewModelScope.launch {
        val docHash = _chat.value.docHash ?: return@launch
        val q = question.trim(); if (q.isEmpty() || _chat.value.generating) return@launch
        if (_ui.value.engine.state != EngineStatus.State.READY) { notice("Load a Gemma model first (Models screen)."); return@launch }
        if (gate.chatBlockedWhileIndexing && _ui.value.indexing != null) { notice("Chat is paused while indexing on this device."); return@launch }

        val history = db.messages().latest(docHash, 8).reversed().let { toQaPairs(it) }
        db.messages().insert(MessageEntity(docHash = docHash, role = "user", text = q))
        val turn = ActiveTurn(turnIds.incrementAndGet(), docHash)
        synchronized(turnLock) { activeTurn = turn }
        _chat.update { it.copy(generating = true, streamingText = "", streamingCitations = emptyList(), error = null, lastStats = null) }
        try {
            val service = connection.await()
            val id = service.ask(docHash, q, history, callbackFor(turn))
            turn.generationId = id
            if (activeTurn === turn) {
                _chat.update { it.copy(generationId = id) }
            } else {
                // Stop may have been tapped before the synchronous Binder call returned.
                runCatching { service.cancelGeneration(id) }
            }
        } catch (t: Throwable) {
            completeTurn(turn, error = "Inference service unavailable: ${t.message}")
        }
    }

    private fun completeTurn(
        turn: ActiveTurn,
        stats: GenerationStats? = null,
        cancelled: Boolean = false,
        error: String? = null,
    ) {
        val snapshot = synchronized(turnLock) {
            if (activeTurn !== turn) return
            activeTurn = null
            val state = _chat.value
            _chat.value = state.copy(
                generating = false,
                streamingText = "",
                streamingCitations = emptyList(),
                lastStats = stats,
                generationId = -1,
                error = error,
            )
            state
        }
        viewModelScope.launch {
            if (snapshot.streamingText.isNotBlank() || cancelled) {
                db.messages().insert(
                    MessageEntity(
                        docHash = turn.docHash, role = "model",
                        text = snapshot.streamingText.ifBlank { "(stopped)" },
                        citationIds = snapshot.streamingCitations.joinToString(",") { it.chunkId },
                        citationNamespaces = snapshot.streamingCitations.joinToString(",") { it.indexNamespace },
                        sourceSectionId = snapshot.streamingCitations.map { it.sectionId }.distinct().singleOrNull().orEmpty(),
                        backend = stats?.backend ?: "",
                        tokensPerSecond = stats?.approxTokensPerSecond ?: 0.0,
                        cancelled = cancelled || error != null || (stats?.cancelled == true),
                    ),
                )
            }
        }
    }

    private fun callbackFor(turn: ActiveTurn) = object : IStreamCallback.Stub() {
        override fun onRetrieved(generationId: Long, citations: List<Citation>) {
            turn.generationId = generationId
            _chat.update { if (activeTurn === turn) it.copy(generationId = generationId, streamingCitations = citations) else it }
        }
        override fun onToken(generationId: Long, token: String) {
            turn.generationId = generationId
            _chat.update { if (activeTurn === turn) it.copy(generationId = generationId, streamingText = it.streamingText + token) else it }
        }
        override fun onDone(generationId: Long, stats: GenerationStats) {
            completeTurn(turn, stats, stats.cancelled)
        }
        override fun onError(generationId: Long, message: String) {
            completeTurn(turn, cancelled = _chat.value.streamingText.isNotBlank(), error = message)
        }
    }

    fun stopGeneration() {
        val turn = synchronized(turnLock) { activeTurn } ?: return
        // Unlock the composer immediately. LiteRT-LM cancelProcess() often does not return
        // during GPU prefill, so waiting for onDone left the UI stuck.
        completeTurn(turn, cancelled = true, error = "Stopped.")
        viewModelScope.launch {
            val id = turn.generationId
            if (id >= 0) runCatching { connection.await().cancelGeneration(id) }
                .onFailure { Log.w(TAG, "cancelGeneration failed", it) }
        }
    }

    suspend fun citation(indexNamespace: String, chunkId: String): Citation? = withContext(Dispatchers.IO) {
        runCatching {
            if (indexNamespace.isBlank()) connection.await().getCitationAsync(chunkId)
            else connection.await().getCitationInNamespaceAsync(indexNamespace, chunkId)
        }.getOrNull()
    }

    private fun toQaPairs(messages: List<MessageEntity>): List<QaPair> {
        val pairs = ArrayList<QaPair>()
        var pendingQ: String? = null
        for (m in messages) {
            if (m.role == "user") pendingQ = m.text
            else if (pendingQ != null && !m.cancelled) {
                pairs += QaPair(pendingQ, m.text.take(1200), m.sourceSectionId)
                pendingQ = null
            }
        }
        return pairs.takeLast(4)
    }

    // ------------------------------------------------------------------ recovery / diagnostics

    private fun onInferenceDied() {
        Log.w(TAG, "inference process died")
        // A Binder death cannot deliver the load callback that normally clears this guard. Without
        // resetting it, the recovery load below is silently rejected forever.
        loadingModel.set(null)
        val wasGenerating = _chat.value.generating
        val wasIndexing = _ui.value.indexing != null
        synchronized(turnLock) { activeTurn = null }
        if (wasGenerating && _ui.value.engine.backend == "GPU") {
            GpuMarker.write(getApplication(), "inference process died during GPU generation")
        }
        _chat.update { if (wasGenerating) it.copy(generating = false, error = "The model process was restarted by the system.", generationId = -1) else it }
        _ui.update {
            it.copy(
                serviceAlive = false,
                engine = EngineStatus(EngineStatus.State.UNLOADED),
                indexing = null, indexingDocName = null,
                notice = when {
                    wasIndexing -> "Indexing interrupted (process killed). Re-index the document."
                    it.engine.state == EngineStatus.State.READY -> "Reloading model…"
                    else -> it.notice
                },
            )
        }
        if (wasIndexing) viewModelScope.launch {
            documents.value.filter { it.status == STATUS_INDEXING }.forEach { db.documents().update(it.copy(status = STATUS_FAILED, error = "interrupted")) }
        }
        // LMK recovery: transparently reload the last model once the service is back.
        val model = _ui.value.selectedModelPath
        if (model != null && !wasIndexing) viewModelScope.launch { delay(1000); loadEngine(model) }
    }

    fun loadDiagnostics() = viewModelScope.launch {
        val local = "ui: ${gate.describe()}\n"
        val remote = runCatching { connection.await().diagnosticsAsync() }.getOrElse { "service unavailable: ${it.message}" }
        _ui.update { it.copy(diagnostics = local + remote) }
    }

    fun runSelfTest() = viewModelScope.launch {
        _ui.update { it.copy(selfTest = "Running…") }
        val result = runCatching { connection.await().runSelfTestAsync() }.getOrElse { "failed: ${it.message}" }
        _ui.update { it.copy(selfTest = result) }
    }

    fun probeRetrieval() = viewModelScope.launch {
        _ui.update { it.copy(probe = "Probing live index…") }
        val result = runCatching { connection.await().probeRetrievalAsync("") }
            .getOrElse { "failed: ${it.message}" }
        _ui.update { it.copy(probe = result) }
    }

    fun notice(text: String) = _ui.update { it.copy(notice = text) }
    fun clearNotice() = _ui.update { it.copy(notice = null) }

    override fun onCleared() {
        super.onCleared()
        // Keep the binding: the service should outlive configuration changes; the OS reclaims it with the app.
    }

    companion object {
        private const val TAG = "RagViewModel"
        private const val KEY_MODEL = "selected_model"
        const val STATUS_INDEXING = "INDEXING"
        const val STATUS_READY = "READY"
        const val STATUS_FAILED = "FAILED"
        const val STATUS_REINDEX_REQUIRED = "REINDEX_REQUIRED"
        const val CURRENT_INDEX_VERSION = 21
    }
}
