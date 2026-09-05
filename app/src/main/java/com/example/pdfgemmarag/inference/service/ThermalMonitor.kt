package com.example.pdfgemmarag.inference.service

import android.content.Context
import android.os.PowerManager
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first

/**
 * Tracks [PowerManager] thermal status. Indexing pauses at SEVERE and above; generation is never
 * throttled because the user is waiting on it and a single answer is short.
 */
class ThermalMonitor(context: Context) {
    private val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val _status = MutableStateFlow(pm.currentThermalStatus)
    val status: StateFlow<Int> = _status

    private val listener = PowerManager.OnThermalStatusChangedListener { s ->
        Log.i(TAG, "thermal status -> $s")
        _status.value = s
    }

    init {
        pm.addThermalStatusListener(listener)
    }

    val isThrottled: Boolean get() = _status.value >= PowerManager.THERMAL_STATUS_SEVERE

    /** Suspends until the device cools below SEVERE. */
    suspend fun awaitCool() {
        if (!isThrottled) return
        Log.w(TAG, "pausing background work: thermal status ${_status.value}")
        _status.first { it < PowerManager.THERMAL_STATUS_SEVERE }
        Log.i(TAG, "resuming background work")
    }

    fun release() = pm.removeThermalStatusListener(listener)

    companion object { private const val TAG = "ThermalMonitor" }
}
