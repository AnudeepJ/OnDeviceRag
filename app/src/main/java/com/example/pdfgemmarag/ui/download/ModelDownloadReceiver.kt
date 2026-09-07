package com.example.pdfgemmarag.ui.download

import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.example.pdfgemmarag.RagApplication
import com.example.pdfgemmarag.core.process.ProcessGuard
import com.example.pdfgemmarag.inference.service.AiInferenceService

class ModelDownloadReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == AiInferenceService.ACTION_INSTALL_MODEL_RESULT) {
            val id = intent.getLongExtra(AiInferenceService.EXTRA_DOWNLOAD_ID, -1)
            if (id >= 0 && ProcessGuard.isUiProcess(context)) {
                val error = intent.getStringExtra(AiInferenceService.EXTRA_INSTALL_ERROR)
                (context.applicationContext as RagApplication).modelDownloads.onInstallResult(id, error)
            }
            return
        }
        if (intent.action != DownloadManager.ACTION_DOWNLOAD_COMPLETE) return
        val id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1)
        if (id < 0) return
        // Receivers without android:process run in the default (:ui) process, where the manager lives.
        if (!ProcessGuard.isUiProcess(context)) return
        val app = context.applicationContext as RagApplication
        app.modelDownloads.onDownloadComplete(id)
    }
}
