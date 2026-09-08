package com.example.pdfgemmarag.core.model

import android.content.Context
import android.util.Log
import java.io.File

/**
 * One-shot hand-off from `:inference` to `:ui` when the service deliberately recycles itself.
 *
 * A timed-out JNI call cannot be made safe by interrupting its Java worker. The inference process
 * is therefore the recovery boundary. This short-lived marker lets the UI distinguish that
 * controlled recycle from a GPU driver/process crash, so one slow turn does not permanently force
 * subsequent sessions onto CPU.
 */
object InferenceRecoveryMarker {
    private const val TAG = "InferenceRecovery"
    private const val FILE_NAME = "inference_recovery.marker"
    private const val MAX_AGE_MS = 2 * 60 * 1_000L

    fun write(context: Context, reason: String) {
        val marker = file(context)
        val tmp = File(marker.parentFile, "$FILE_NAME.tmp")
        val value = "${System.currentTimeMillis()}\n$reason\n"
        runCatching {
            tmp.writeText(value)
            if (!tmp.renameTo(marker)) marker.writeText(value)
        }.onFailure { Log.e(TAG, "could not record controlled inference recovery", it) }
    }

    /** Returns a fresh recovery reason once; stale markers are deleted and ignored. */
    fun consume(context: Context): String? {
        val marker = file(context)
        val lines = runCatching { marker.readLines() }.getOrNull()
        marker.delete()
        val timestamp = lines?.firstOrNull()?.toLongOrNull() ?: return null
        if (System.currentTimeMillis() - timestamp !in 0..MAX_AGE_MS) return null
        return lines.getOrNull(1)?.takeIf(String::isNotBlank) ?: "native operation timed out"
    }

    private fun file(context: Context): File = File(context.filesDir, FILE_NAME)
}
