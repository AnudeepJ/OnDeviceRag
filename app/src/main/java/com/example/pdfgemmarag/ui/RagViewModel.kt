package com.example.pdfgemmarag.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.RemoteException
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
import com.example.pdfgemmarag.ui.data.DocumentEntity
import com.example.pdfgemmarag.ui.data.MessageEntity
import com.example.pdfgemmarag.ui.download.CatalogEntry
import com.example.pdfgemmarag.ui.download.ModelDownloadManager
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
import java.security.MessageDigest

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

    val documents: StateFlow<List<DocumentEntity>> =
        db.documents().observeAll().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    fun messages(docHash: String): Flow<List<MessageEntity>> = db.messages().observe(docHash)

    private val watchdog: EngineWatchdog = EngineWatchdog(app, connection, viewModelScope) { reason ->
        _ui.update { it.copy(gpuDisabled = true, gpuDisabledReason = reason, notice = "GPU disabled ($reason). Reloading on CPU.") }
        viewModelScope.launch { delay(1500); _ui.value.selectedModelPath?.let { loadEngine(it) } }
    }

    init {
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
            it.copy(
                installedModels = ModelPaths.installedLlms(app),
                embeddingReady = ModelPaths.embeddingReady(app),
                gpuDisabled = GpuMarker.exists(app),
                gpuDisabledReason = GpuMarker.reason(app),
                selectedModelPath = it.selectedModelPath ?: prefs.getString(KEY_MODEL, null)?.takeIf { p -> File(p).exists() },
            )
        }
    }

    private fun syncEngineStatus() = viewModelScope.launch {
        runCatching { connection.await().engineStatus }.onSuccess { s -> _ui.update { it.copy(engine = s) } }
    }

    fun loadEngine(modelPath: String) = viewModelScope.launch {
        val app = getApplication<Application>()
        prefs.edit { putString(KEY_MODEL, modelPath) }
        val gpuAttempt = !GpuMarker.exists(app)
        _ui.update { it.copy(selectedModelPath = modelPath, engine = EngineStatus(EngineStatus.State.LOADING, modelName = File(modelPath).name), engineMessage = "Starting…") }
        watchdog.onLoadStarted(gpuAttempt)
        try {
            connection.startService()
            connection.await().loadEngine(modelPath, true, object : IEngineCallback.Stub() {
                override fun onProgress(stage: String) { _ui.update { it.copy(engineMessage = stage) } }
                override fun onReady(status: EngineStatus) {
                    watchdog.onLoadFinished()
                    _ui.update { it.copy(engine = status, engineMessage = "Ready on ${status.backend}", gpuDisabled = GpuMarker.exists(app)) }
                }
                override fun onFailed(message: String) {
                    watchdog.onLoadFinished()
                    _ui.update { it.copy(engine = EngineStatus(EngineStatus.State.FAILED, message = message), engineMessage = message, gpuDisabled = GpuMarker.exists(app)) }
                }
            })
        } catch (t: Throwable) {
            watchdog.onLoadFinished()
            _ui.update { it.copy(engine = EngineStatus(EngineStatus.State.FAILED, message = t.message ?: ""), engineMessage = t.message ?: "load failed") }
        }
    }

    fun unloadEngine() = viewModelScope.launch {
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
        if (entry.sha256.isBlank()) notice("No SHA-256 configured for ${entry.fileName}; it will install unverified.")
        downloads.enqueue(entry)
    }

    fun cancelDownload(entry: CatalogEntry) = downloads.cancel(entry)

    /** Local `.litertlm`/`.tflite`/`.model` picker: copy through staging, then verify+install in :inference. */
    fun importLocalModel(uri: Uri, displayName: String) = viewModelScope.launch {
        val app = getApplication<Application>()
        val targetName = when {
            displayName.endsWith(".litertlm") -> displayName
            displayName.endsWith(".tflite") -> ModelPaths.EMBEDDING_MODEL_FILE
            displayName.endsWith(".model") -> ModelPaths.EMBEDDING_TOKENIZER_FILE
            else -> { notice("Unsupported file: $displayName"); return@launch }
        }
        _ui.update { it.copy(busy = true, notice = "Copying $displayName…") }
        try {
            val staged = withContext(Dispatchers.IO) {
                val f = File(ModelPaths.stagingDir(app), targetName)
                app.contentResolver.openInputStream(uri)!!.use { input -> f.outputStream().use { input.copyTo(it, 1 shl 20) } }
                f
            }
            connection.startService()
            connection.await().installModel(staged.absolutePath, targetName, "", -1, true, object : IInstallCallback.Stub() {
                override fun onProgress(bytesCopied: Long, totalBytes: Long) { _ui.update { it.copy(install = Triple(targetName, bytesCopied, totalBytes)) } }
                override fun onInstalled(targetPath: String, sha256: String) {
                    _ui.update { it.copy(install = null, busy = false, notice = "Installed $targetName (sha256 ${sha256.take(12)}…)") }
                    refreshLocalState()
                }
                override fun onFailed(message: String) { _ui.update { it.copy(install = null, busy = false, notice = "Install failed: $message") } }
            })
        } catch (t: Throwable) {
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
                db.documents().get(document.docHash)?.let {
                    db.documents().update(it.copy(pageCount = document.pageCount, chunkCount = document.chunkCount, script = document.script, status = STATUS_READY, indexedAt = System.currentTimeMillis()))
                }
                _ui.update { it.copy(indexing = null, indexingDocName = null, notice = "Indexed ${document.displayName}: ${document.pageCount} pages, ${document.chunkCount} chunks") }
            }
        }
        override fun onCancelled(docHash: String) {
            viewModelScope.launch {
                db.documents().get(docHash)?.let { db.documents().delete(it) }
                ModelPaths.pdfFile(getApplication(), docHash).delete()
                _ui.update { it.copy(indexing = null, indexingDocName = null, notice = "Indexing cancelled") }
            }
        }
        override fun onFailed(docHash: String, message: String) {
            viewModelScope.launch {
                db.documents().get(docHash)?.let { db.documents().update(it.copy(status = STATUS_FAILED, error = message)) }
                _ui.update { it.copy(indexing = null, indexingDocName = null, notice = "Indexing failed: $message") }
            }
        }
    }

    fun cancelIndexing() = viewModelScope.launch { runCatching { connection.await().cancelIndexing() } }

    fun reindex(doc: DocumentEntity) = viewModelScope.launch {
        if (_ui.value.indexing != null) { notice("Another document is being indexed."); return@launch }
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

    fun openChat(docHash: String) { if (_chat.value.docHash != docHash) _chat.value = ChatState(docHash = docHash) }

    fun ask(question: String) = viewModelScope.launch {
        val docHash = _chat.value.docHash ?: return@launch
        val q = question.trim(); if (q.isEmpty() || _chat.value.generating) return@launch
        if (_ui.value.engine.state != EngineStatus.State.READY) { notice("Load a Gemma model first (Models screen)."); return@launch }
        if (gate.chatBlockedWhileIndexing && _ui.value.indexing != null) { notice("Chat is paused while indexing on this device."); return@launch }

        val history = db.messages().latest(docHash, 8).reversed().let { toQaPairs(it) }
        db.messages().insert(MessageEntity(docHash = docHash, role = "user", text = q))
        _chat.update { it.copy(generating = true, streamingText = "", streamingCitations = emptyList(), error = null, lastStats = null) }
        try {
            val id = connection.await().ask(docHash, q, history, streamCallback)
            _chat.update { it.copy(generationId = id) }
            watchIdleComplete(id)
        } catch (t: RemoteException) {
            _chat.update { it.copy(generating = false, error = "Inference service unavailable: ${t.message}") }
        }
    }

    /** LiteRT-LM sometimes never delivers onDone after the last token. Unlock the composer. */
    private fun watchIdleComplete(generationId: Long) = viewModelScope.launch {
        var last = ""
        var quiet = 0
        while (_chat.value.generating && _chat.value.generationId == generationId) {
            delay(1000)
            val text = _chat.value.streamingText
            if (text.isNotBlank() && text == last) {
                quiet++
                val need = if (text.trim().endsWith('.') || text.contains("[Page")) 4 else 10
                if (quiet >= need) {
                    Log.i(TAG, "UI idle-complete after ${quiet}s")
                    finishTurn(generationId, cancelled = false)
                    return@launch
                }
            } else {
                quiet = 0
                last = text
            }
        }
    }

    private fun finishTurn(generationId: Long, stats: GenerationStats? = null, cancelled: Boolean = false) {
        val state = _chat.value
        if (!state.generating) return
        if (state.generationId != generationId && state.generationId != -1L) return
        viewModelScope.launch {
            val latest = _chat.value
            if (!latest.generating) return@launch
            latest.docHash?.let { doc ->
                db.messages().insert(
                    MessageEntity(
                        docHash = doc, role = "model",
                        text = latest.streamingText.ifBlank { if (cancelled) "(stopped)" else "(no answer)" },
                        citationIds = latest.streamingCitations.joinToString(",") { it.chunkId },
                        backend = stats?.backend ?: "",
                        tokensPerSecond = stats?.approxTokensPerSecond ?: 0.0,
                        cancelled = cancelled || (stats?.cancelled == true),
                    ),
                )
            }
            _chat.update { it.copy(generating = false, streamingText = "", streamingCitations = emptyList(), lastStats = stats, generationId = -1) }
        }
    }

    private val streamCallback = object : IStreamCallback.Stub() {
        override fun onRetrieved(generationId: Long, citations: List<Citation>) {
            _chat.update { if (it.generationId in listOf(-1L, generationId)) it.copy(streamingCitations = citations) else it }
        }
        override fun onToken(generationId: Long, token: String) {
            _chat.update { if (it.generationId == generationId || it.generationId == -1L) it.copy(streamingText = it.streamingText + token) else it }
        }
        override fun onDone(generationId: Long, stats: GenerationStats) {
            finishTurn(generationId, stats, stats.cancelled)
        }
        override fun onError(generationId: Long, message: String) {
            val state = _chat.value
            if (state.generating && state.streamingText.isNotBlank()) {
                finishTurn(generationId, cancelled = false)
            } else {
                _chat.update { it.copy(generating = false, error = message, generationId = -1) }
            }
        }
    }

    fun stopGeneration() {
        val state = _chat.value
        if (!state.generating) return
        // Unlock the composer immediately. LiteRT-LM cancelProcess() often does not return
        // during GPU prefill, so waiting for onDone left the UI stuck.
        _chat.update { it.copy(generating = false, error = "Stopped.", generationId = -1) }
        viewModelScope.launch {
            val partial = state.streamingText
            state.docHash?.let { doc ->
                db.messages().insert(
                    MessageEntity(
                        docHash = doc, role = "model",
                        text = partial.ifBlank { "(stopped)" },
                        citationIds = state.streamingCitations.joinToString(",") { it.chunkId },
                        cancelled = true,
                    ),
                )
            }
            runCatching { connection.await().cancelGeneration(state.generationId) }
                .onFailure { Log.w(TAG, "cancelGeneration failed", it) }
        }
    }

    suspend fun citation(chunkId: String): Citation? = withContext(Dispatchers.IO) {
        runCatching { connection.await().getCitation(chunkId) }.getOrNull()
    }

    private fun toQaPairs(messages: List<MessageEntity>): List<QaPair> {
        val pairs = ArrayList<QaPair>()
        var pendingQ: String? = null
        for (m in messages) {
            if (m.role == "user") pendingQ = m.text
            else if (pendingQ != null && !m.cancelled) { pairs += QaPair(pendingQ, m.text.take(1200)); pendingQ = null }
        }
        return pairs.takeLast(4)
    }

    // ------------------------------------------------------------------ recovery / diagnostics

    private fun onInferenceDied() {
        Log.w(TAG, "inference process died")
        val wasGenerating = _chat.value.generating
        val wasIndexing = _ui.value.indexing != null
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
        val remote = runCatching { withContext(Dispatchers.IO) { connection.await().diagnostics } }.getOrElse { "service unavailable: ${it.message}" }
        _ui.update { it.copy(diagnostics = local + remote) }
    }

    fun runSelfTest() = viewModelScope.launch {
        _ui.update { it.copy(selfTest = "Running…") }
        val result = runCatching { withContext(Dispatchers.IO) { connection.await().runSelfTest() } }.getOrElse { "failed: ${it.message}" }
        _ui.update { it.copy(selfTest = result) }
    }

    fun probeRetrieval() = viewModelScope.launch {
        _ui.update { it.copy(probe = "Probing live index…") }
        val result = runCatching { withContext(Dispatchers.IO) { connection.await().probeRetrieval("") } }
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
    }
}
