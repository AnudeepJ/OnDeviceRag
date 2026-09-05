package com.example.pdfgemmarag.core.model

import android.content.Context
import android.util.Log
import java.io.File

/**
 * `filesDir/gpu_disabled.marker`: presence means "initialise Gemma on CPU". Written by the :ui
 * watchdog when :inference dies or stalls during a GPU attempt, or by the service on a clean GPU
 * exception. Written atomically (temp + rename) so a partially written file is never observed.
 */
object GpuMarker {
    private const val TAG = "GpuMarker"
    const val FILE_NAME = "gpu_disabled.marker"

    fun file(context: Context): File = File(context.filesDir, FILE_NAME)

    fun exists(context: Context): Boolean = file(context).exists()

    fun write(context: Context, reason: String) {
        val marker = file(context)
        val tmp = File(marker.parentFile, marker.name + ".tmp")
        tmp.writeText("${System.currentTimeMillis()}\n$reason\n")
        if (!tmp.renameTo(marker)) marker.writeText("${System.currentTimeMillis()}\n$reason\n")
        Log.w(TAG, "GPU disabled: $reason")
    }

    fun clear(context: Context) {
        if (file(context).delete()) Log.i(TAG, "GPU marker cleared; next load retries GPU")
        if (embedderFile(context).delete()) Log.i(TAG, "embedder GPU marker cleared")
    }

    /**
     * Separate marker for the EmbeddingGemma `CompiledModel`: a failed GPU compile is a clean exception
     * but costs ~25 s per process start (measured on the emulator), so the decision is persisted too.
     */
    const val EMBEDDER_FILE_NAME = "embedder_gpu_disabled.marker"
    fun embedderFile(context: Context): File = File(context.filesDir, EMBEDDER_FILE_NAME)
    fun embedderGpuDisabled(context: Context): Boolean = embedderFile(context).exists()
    fun writeEmbedder(context: Context, reason: String) {
        runCatching { embedderFile(context).writeText("${System.currentTimeMillis()}\n$reason\n") }
        Log.w(TAG, "embedder GPU disabled: $reason")
    }

    fun reason(context: Context): String? = runCatching { file(context).readText().lines().getOrNull(1) }.getOrNull()
}
