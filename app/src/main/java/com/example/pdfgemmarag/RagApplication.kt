package com.example.pdfgemmarag

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import com.example.pdfgemmarag.core.process.ProcessGuard
import com.example.pdfgemmarag.ui.data.ChatDatabase
import com.example.pdfgemmarag.ui.download.ModelDownloadManager
import com.example.pdfgemmarag.ui.service.ServiceConnectionManager

class RagApplication : Application() {

    // UI-process singletons. Never created in :inference.
    lateinit var serviceConnection: ServiceConnectionManager
        private set
    lateinit var database: ChatDatabase
        private set
    lateinit var modelDownloads: ModelDownloadManager
        private set

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
        if (ProcessGuard.isInferenceProcess(this)) {
            Log.i(TAG, "Application created in :inference process")
            // Nothing else: AiInferenceService owns all native runtimes and creates them lazily.
            return
        }
        Log.i(TAG, "Application created in :ui process")
        database = ChatDatabase.create(this)
        serviceConnection = ServiceConnectionManager(this)
        modelDownloads = ModelDownloadManager(this)
    }

    private fun createNotificationChannels() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(
                CHANNEL_INDEXING,
                "Document indexing",
                NotificationManager.IMPORTANCE_LOW,
            ).apply { description = "Progress while a PDF is being indexed on-device" },
        )
        nm.createNotificationChannel(
            NotificationChannel(
                CHANNEL_MODELS,
                "Model downloads",
                NotificationManager.IMPORTANCE_LOW,
            ).apply { description = "Model download, verification and installation" },
        )
    }

    companion object {
        private const val TAG = "RagApplication"
        const val CHANNEL_INDEXING = "indexing"
        const val CHANNEL_MODELS = "models"
    }
}
