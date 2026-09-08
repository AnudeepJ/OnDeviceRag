package com.example.pdfgemmarag.ui.service

import android.content.Context
import android.util.Log
import com.example.pdfgemmarag.core.model.GpuMarker
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

/**
 * Lives in :ui and watches a GPU load attempt from the outside. A GPU driver crash takes the whole
 * :inference process with it, so the only reliable observer is another process:
 *
 * - if the service dies while a GPU attempt is in flight, or
 * - if no `onReady` arrives within 120 s (cold shader cache) / 30 s (warm cache),
 *
 * the watchdog writes `gpu_disabled.marker` and restarts the service. The next load reads the
 * marker and goes straight to CPU. "Retry GPU" in settings deletes the marker.
 */
class EngineWatchdog(
    private val context: Context,
    private val connection: ServiceConnectionManager,
    private val scope: CoroutineScope,
    private val onFallback: (reason: String) -> Unit,
) {
    private var timer: Job? = null
    @Volatile private var gpuAttemptInFlight = false

    init {
        scope.launch {
            connection.events.collect { ev ->
                if (ev is ServiceConnectionManager.Event.Died && gpuAttemptInFlight) {
                    if (ev.controlledRecoveryReason != null) {
                        Log.w(TAG, "GPU load interrupted by controlled process recycle")
                        onLoadFinished()
                    } else {
                        trip("inference process died during GPU initialisation")
                    }
                }
            }
        }
    }

    fun onLoadStarted(gpuAttempt: Boolean, model: File) {
        timer?.cancel()
        gpuAttemptInFlight = gpuAttempt
        if (!gpuAttempt) return
        val timeoutMs = if (GpuMarker.hasReadyCache(context, model)) WARM_TIMEOUT_MS else COLD_TIMEOUT_MS
        Log.i(TAG, "watching GPU init, timeout ${timeoutMs / 1000}s")
        timer = scope.launch {
            delay(timeoutMs)
            if (gpuAttemptInFlight) {
                trip("no engine ready signal after ${timeoutMs / 1000}s")
                connection.restart()
            }
        }
    }

    fun onLoadFinished() {
        timer?.cancel(); timer = null
        gpuAttemptInFlight = false
    }

    private fun trip(reason: String) {
        gpuAttemptInFlight = false
        timer?.cancel(); timer = null
        GpuMarker.write(context, reason)
        onFallback(reason)
    }

    companion object {
        private const val TAG = "EngineWatchdog"
        const val COLD_TIMEOUT_MS = 120_000L
        const val WARM_TIMEOUT_MS = 30_000L
    }
}
