package com.example.pdfgemmarag.ui.download

import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.example.pdfgemmarag.RagApplication
import com.example.pdfgemmarag.core.process.ProcessGuard

class ModelDownloadReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != DownloadManager.ACTION_DOWNLOAD_COMPLETE) return
        val id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1)
        if (id < 0) return
        // Receivers without android:process run in the default (:ui) process, where the manager lives.
        if (!ProcessGuard.isUiProcess(context)) return
        val app = context.applicationContext as RagApplication
        app.modelDownloads.onDownloadComplete(id)
    }
}
