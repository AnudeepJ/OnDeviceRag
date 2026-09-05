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
    ) {
        val fraction: Float get() = if (totalBytes <= 0) 0f else (bytesSoFar.toFloat() / totalBytes).coerceIn(0f, 1f)
        val running: Boolean get() = status == DownloadManager.STATUS_RUNNING || status == DownloadManager.STATUS_PENDING || status == DownloadManager.STATUS_PAUSED
    }

    private val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private val _states = MutableStateFlow<Map<String, DownloadState>>(emptyMap())
    val states: StateFlow<Map<String, DownloadState>> = _states

    /** Where DownloadManager writes: app-specific external dir, no storage permission needed. */
    private fun downloadDir(): File = File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "models").apply { mkdirs() }

    fun installed(entry: CatalogEntry): Boolean {
        val f = File(ModelPaths.modelsDir(context), entry.fileName)
        return f.exists() && f.length() > entry.sizeBytes / 2
    }

    fun enqueue(entry: CatalogEntry): Long {
        val target = File(downloadDir(), entry.fileName)
        if (target.exists()) target.delete()
        val request = DownloadManager.Request(Uri.parse(entry.url))
            .setTitle(entry.displayName)
            .setDescription("Downloading model for on-device RAG")
            .setDestinationUri(Uri.fromFile(target))
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setAllowedOverMetered(false)
            .setAllowedOverRoaming(false)
        val id = dm.enqueue(request)
        prefs.edit { putString(keyFor(id), entry.id) }
        Log.i(TAG, "enqueued ${entry.fileName} as download $id")
        refresh()
        return id
    }

    fun cancel(entry: CatalogEntry) {
        _states.value[entry.id]?.let { dm.remove(it.downloadId); prefs.edit { remove(keyFor(it.downloadId)) } }
        refresh()
    }

    /** Polls DownloadManager for every download we own. Cheap; the UI calls it once a second while visible. */
    fun refresh() {
        val ids = prefs.all.keys.filter { it.startsWith(KEY_PREFIX) }.mapNotNull { it.removePrefix(KEY_PREFIX).toLongOrNull() }
        if (ids.isEmpty()) { _states.value = emptyMap(); return }
        val map = HashMap<String, DownloadState>()
        dm.query(DownloadManager.Query().setFilterById(*ids.toLongArray()))?.use { c ->
            val idIdx = c.getColumnIndex(DownloadManager.COLUMN_ID)
            val statusIdx = c.getColumnIndex(DownloadManager.COLUMN_STATUS)
            val soFarIdx = c.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
            val totalIdx = c.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
            val reasonIdx = c.getColumnIndex(DownloadManager.COLUMN_REASON)
            while (c.moveToNext()) {
                val id = c.getLong(idIdx)
                val entryId = prefs.getString(keyFor(id), null) ?: continue
                map[entryId] = DownloadState(entryId, id, c.getInt(statusIdx), c.getLong(soFarIdx), c.getLong(totalIdx), c.getInt(reasonIdx))
            }
        }
        _states.value = map
    }

    /** Called by the receiver when DownloadManager reports completion for one of our ids. */
    fun onDownloadComplete(downloadId: Long) {
        val entryId = prefs.getString(keyFor(downloadId), null) ?: return
        val entry = ModelCatalog.entries.firstOrNull { it.id == entryId } ?: return
        var success = false
        var localUri: String? = null
        dm.query(DownloadManager.Query().setFilterById(downloadId))?.use { c ->
            if (c.moveToFirst()) {
                success = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)) == DownloadManager.STATUS_SUCCESSFUL
                localUri = c.getString(c.getColumnIndexOrThrow(DownloadManager.COLUMN_LOCAL_URI))
            }
        }
        if (!success) { Log.w(TAG, "download $downloadId for ${entry.fileName} did not succeed"); refresh(); return }
        val path = localUri?.let { Uri.parse(it).path } ?: File(downloadDir(), entry.fileName).absolutePath
        Log.i(TAG, "download complete: $path -> verify+install in :inference")
        val intent = Intent(context, AiInferenceService::class.java)
            .setAction(AiInferenceService.ACTION_INSTALL_MODEL)
            .putExtra(AiInferenceService.EXTRA_SOURCE, path)
            .putExtra(AiInferenceService.EXTRA_TARGET_NAME, entry.fileName)
            .putExtra(AiInferenceService.EXTRA_SHA256, entry.sha256)
            .putExtra(AiInferenceService.EXTRA_SIZE, -1L) // CDN size may differ from the catalog estimate; hash is authoritative
        ContextCompat.startForegroundService(context, intent)
        prefs.edit { remove(keyFor(downloadId)) }
        refresh()
    }

    private fun keyFor(id: Long) = KEY_PREFIX + id

    companion object {
        private const val TAG = "ModelDownloadManager"
        private const val PREFS = "model_downloads"
        private const val KEY_PREFIX = "dl_"
    }
}
