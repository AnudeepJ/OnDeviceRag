package com.example.pdfgemmarag.ui.service

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import com.example.pdfgemmarag.inference.service.AiInferenceService
import com.example.pdfgemmarag.inference.service.IAiInferenceService
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withTimeout

/**
 * Keeps a binding to [AiInferenceService] in the :inference process alive for the UI process and
 * surfaces death events so the watchdog and view model can react (marker file, "Reloading model").
 */
class ServiceConnectionManager(private val context: Context) {

    sealed class Event {
        data object Connected : Event()
        /** The :inference process died (LMK, native crash). Binder will reconnect automatically. */
        data object Died : Event()
    }

    private val _service = MutableStateFlow<IAiInferenceService?>(null)
    val service: StateFlow<IAiInferenceService?> = _service
    private val _events = MutableSharedFlow<Event>(extraBufferCapacity = 8)
    val events: SharedFlow<Event> = _events
    @Volatile var bound = false
        private set

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            Log.i(TAG, "connected to :inference")
            _service.value = IAiInferenceService.Stub.asInterface(binder)
            _events.tryEmit(Event.Connected)
        }

        override fun onServiceDisconnected(name: ComponentName) {
            Log.w(TAG, ":inference disconnected (process died)")
            _service.value = null
            _events.tryEmit(Event.Died)
        }

        override fun onBindingDied(name: ComponentName) {
            Log.w(TAG, "binding died; rebinding")
            _service.value = null
            _events.tryEmit(Event.Died)
            unbind(); bind()
        }
    }

    fun bind() {
        if (bound) return
        val intent = Intent(context, AiInferenceService::class.java)
        bound = context.bindService(intent, connection, Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT)
        Log.i(TAG, "bindService -> $bound")
    }

    fun unbind() {
        if (!bound) return
        runCatching { context.unbindService(connection) }
        bound = false
        _service.value = null
    }

    /** Hard restart: used by the watchdog when the GPU init stalls. */
    fun restart() {
        Log.w(TAG, "restarting :inference service")
        unbind()
        context.stopService(Intent(context, AiInferenceService::class.java))
        bind()
    }

    /** Ensures the service is started (not just bound) so `startForeground` is legal later. */
    fun startService() {
        context.startService(Intent(context, AiInferenceService::class.java).setAction(AiInferenceService.ACTION_KEEPALIVE))
    }

    suspend fun await(timeoutMs: Long = 15_000): IAiInferenceService {
        bind()
        return withTimeout(timeoutMs) { _service.filterNotNull().first() }
    }

    companion object { private const val TAG = "ServiceConnection" }
}
