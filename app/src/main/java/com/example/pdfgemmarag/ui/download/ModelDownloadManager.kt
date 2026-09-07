package com.example.pdfgemmarag.ui.download

import android.app.DownloadManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Environment
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.inference.service.AiInferenceService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File

/**
 * Model downloads through the system [DownloadManager] (resumable, survives process death, shows
 * its own notification). On completion [ModelDownloadReceiver] hands the file to the :inference
 * service, which verifies the SHA-256 while copying into `filesDir/models/` and deletes the download.
 */
class ModelDownloadManager(private val context: Context) {

    data class DownloadState(
        val entryId: String,
        val downloadId: Long,
        val status: Int,
        val bytesSoFar: Long,
        val totalBytes: Long,
        val reason: Int = 0,
        val message: String? = null,
    ) {
        val fraction: Float get() = if (totalBytes <= 0) 0f else (bytesSoFar.toFloat() / totalBytes).coerceIn(0f, 1f)
        val running: Boolean get() = status == DownloadManager.STATUS_RUNNING || status == DownloadManager.STATUS_PENDING || status == DownloadManager.STATUS_PAUSED
    }

    private val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private val _states = MutableStateFlow<Map<String, DownloadState>>(emptyMap())
    val states: StateFlow<Map<String, DownloadState>> = _states

    /** Where DownloadManager writes: app-specific external dir, no storage permission needed. */
    private fun downloadDir(): File {
        val root = checkNotNull(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)) {
            "App download storage is unavailable"
        }
        return File(root, "models").apply { check(exists() || mkdirs()) { "Could not create model download directory" } }
    }

    fun installed(entry: CatalogEntry): Boolean {
        val f = File(ModelPaths.modelsDir(context), entry.fileName)
        if (!f.isFile) return false
        return entry.expectedSizeBytes?.let { f.length() == it } ?: (f.length() > entry.sizeBytes / 2)
    }

    fun enqueue(entry: CatalogEntry): Long {
        require(entry.downloadable) {
            "Model download is not configured. Set an HTTPS MODEL_CDN_BASE_URL and the 64-character SHA-256 for ${entry.fileName}."
        }
        removeOwnedDownloads(entry.id)
        prefs.edit { remove(errorKey(entry.id)) }
        val target = File(downloadDir(), entry.fileName)
        check(!target.exists() || target.delete()) { "Could not clear the previous download for ${entry.fileName}" }
        val request = DownloadManager.Request(Uri.parse(entry.url))
            .setTitle(entry.displayName)
            .setDescription("Downloading model for on-device RAG")
            // Let DownloadManager resolve the app-specific external path. Passing a raw file://
            // destination is handled inconsistently by some scoped-storage OEM builds.
            .setDestinationInExternalFilesDir(context, Environment.DIRECTORY_DOWNLOADS, "models/${entry.fileName}")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setAllowedOverMetered(false)
            .setAllowedOverRoaming(false)
        val id = dm.enqueue(request)
        prefs.edit { putString(keyFor(id), entry.id) }
        Log.i(TAG, "enqueued ${entry.fileName} as download $id")
        refresh()
        return id
    }

    fun cancel(entry: CatalogEntry) {
        removeOwnedDownloads(entry.id)
        refresh()
    }

    /** Polls DownloadManager for every download we own. Cheap; the UI calls it once a second while visible. */
    fun refresh() {
        val ids = prefs.all.keys.filter { it.startsWith(KEY_PREFIX) }.mapNotNull { it.removePrefix(KEY_PREFIX).toLongOrNull() }
        val map = HashMap<String, DownloadState>()
        prefs.all.forEach { (key, value) ->
            if (key.startsWith(ERROR_PREFIX) && value is String) {
                val entryId = key.removePrefix(ERROR_PREFIX)
                map[entryId] = DownloadState(entryId, -1, DownloadManager.STATUS_FAILED, 0, 0, message = value)
            }
        }
        if (ids.isEmpty()) { _states.value = map; return }
        val completed = ArrayList<Long>()
        dm.query(DownloadManager.Query().setFilterById(*ids.toLongArray()))?.use { c ->
            val idIdx = c.getColumnIndex(DownloadManager.COLUMN_ID)
            val statusIdx = c.getColumnIndex(DownloadManager.COLUMN_STATUS)
            val soFarIdx = c.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
            val totalIdx = c.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
            val reasonIdx = c.getColumnIndex(DownloadManager.COLUMN_REASON)
            while (c.moveToNext()) {
                val id = c.getLong(idIdx)
                val entryId = prefs.getString(keyFor(id), null) ?: continue
                val status = c.getInt(statusIdx)
                map[entryId] = DownloadState(entryId, id, status, c.getLong(soFarIdx), c.getLong(totalIdx), c.getInt(reasonIdx))
                if (status == DownloadManager.STATUS_SUCCESSFUL) completed += id
            }
        }
        _states.value = map
        completed.forEach(::onDownloadComplete)
    }

    /** Called by the receiver when DownloadManager reports completion for one of our ids. */
    fun onDownloadComplete(downloadId: Long) {
        val entryId = prefs.getString(keyFor(downloadId), null) ?: return
        val entry = ModelCatalog.entries.firstOrNull { it.id == entryId } ?: return
        if (installed(entry)) {
            clearDownload(downloadId, entryId)
            return
        }
        if (!entry.downloadable) {
            Log.e(TAG, "refusing unverified catalog install for ${entry.fileName}")
            prefs.edit { remove(keyFor(downloadId)) }
            refresh()
            return
        }
        var success = false
        dm.query(DownloadManager.Query().setFilterById(downloadId))?.use { c ->
            if (c.moveToFirst()) {
                success = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)) == DownloadManager.STATUS_SUCCESSFUL
            }
        }
        if (!success) { Log.w(TAG, "download $downloadId for ${entry.fileName} did not succeed"); refresh(); return }
        // We supplied this exact app-specific destination. Do not trust COLUMN_LOCAL_URI to be a
        // file URI: some OEM DownloadManager implementations return a content:// URI instead.
        val path = File(downloadDir(), entry.fileName).absolutePath
        val now = System.currentTimeMillis()
        val installStarted = prefs.getLong(installKey(downloadId), 0L)
        if (now - installStarted < INSTALL_RETRY_AFTER_MS) return
        Log.i(TAG, "download complete: $path -> verify+install in :inference")
        val intent = Intent(context, AiInferenceService::class.java)
            .setAction(AiInferenceService.ACTION_INSTALL_MODEL)
            .putExtra(AiInferenceService.EXTRA_SOURCE, path)
            .putExtra(AiInferenceService.EXTRA_TARGET_NAME, entry.fileName)
            .putExtra(AiInferenceService.EXTRA_SHA256, entry.sha256)
            // Reject truncated responses and tiny HTML/error payloads before spending another full
            // streaming pass hashing and copying the artifact. The SHA remains authoritative.
            .putExtra(AiInferenceService.EXTRA_SIZE, entry.expectedSizeBytes ?: -1L)
            .putExtra(AiInferenceService.EXTRA_DOWNLOAD_ID, downloadId)
        prefs.edit { putLong(installKey(downloadId), now) }
        runCatching { ContextCompat.startForegroundService(context, intent) }
            .onFailure {
                prefs.edit { remove(installKey(downloadId)) }
                Log.e(TAG, "could not start verified install for ${entry.fileName}", it)
            }
    }

    /** Called by the inference service after the queued verify+install operation reaches terminal state. */
    fun onInstallResult(downloadId: Long, error: String?) {
        val entryId = prefs.getString(keyFor(downloadId), null) ?: return
        clearDownload(downloadId, entryId)
        if (error != null) prefs.edit { putString(errorKey(entryId), error) }
        refresh()
    }

    private fun clearDownload(downloadId: Long, entryId: String) {
        dm.remove(downloadId)
        prefs.edit {
            remove(keyFor(downloadId))
            remove(installKey(downloadId))
            val entry = ModelCatalog.entries.firstOrNull { it.id == entryId }
            if (entry != null && installed(entry)) remove(errorKey(entryId))
        }
    }

    private fun removeOwnedDownloads(entryId: String) {
        val ids = prefs.all.filterValues { it == entryId }.keys
            .filter { it.startsWith(KEY_PREFIX) }
            .mapNotNull { it.removePrefix(KEY_PREFIX).toLongOrNull() }
        ids.forEach(dm::remove)
        prefs.edit { ids.forEach { remove(keyFor(it)); remove(installKey(it)) } }
    }

    private fun keyFor(id: Long) = KEY_PREFIX + id
    private fun installKey(id: Long) = INSTALL_PREFIX + id
    private fun errorKey(entryId: String) = ERROR_PREFIX + entryId

    companion object {
        private const val TAG = "ModelDownloadManager"
        private const val PREFS = "model_downloads"
        private const val KEY_PREFIX = "dl_"
        private const val INSTALL_PREFIX = "install_"
        private const val ERROR_PREFIX = "error_"
        private const val INSTALL_RETRY_AFTER_MS = 15L * 60 * 1000
    }
}
