package com.example.pdfgemmarag.spikes

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.core.model.EngineStatus
import com.example.pdfgemmarag.core.model.GpuMarker
import com.example.pdfgemmarag.inference.service.AiInferenceService
import com.example.pdfgemmarag.inference.service.IAiInferenceService
import com.example.pdfgemmarag.inference.service.IEngineCallback
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Spike 4, the part that runs without a GPU: the GPU -> CPU fallback contract across the process
 * boundary. Binds the real `:inference` service, asks it to load a deliberately broken model on the
 * GPU and checks that (a) the request is reported through the AIDL callback rather than hanging,
 * (b) a GPU failure persists `gpu_disabled.marker` in the shared filesDir, and (c) the next load
 * request reads the marker and goes straight to CPU. If LiteRT-LM crashes the process instead of
 * throwing, the binder death is observed and reported: that is the path the UI watchdog covers.
 *
 * Real-device GPU initialisation timing (Gemma 4 E2B, cold and warm shader cache) still has to be
 * measured on the two weakest target phones; see docs/SPIKES.md.
 */
@RunWith(AndroidJUnit4::class)
class Spike4GpuFallbackTest {

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private var service: IAiInferenceService? = null
    private var died = CountDownLatch(1)
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            service = IAiInferenceService.Stub.asInterface(binder)
            binder.linkToDeath({ died.countDown() }, 0)
            connected.countDown()
        }
        override fun onServiceDisconnected(name: ComponentName) { service = null; died.countDown() }
    }
    private var connected = CountDownLatch(1)

    @Before
    fun bind() {
        GpuMarker.clear(ctx)
        val intent = Intent(ctx, AiInferenceService::class.java)
        assertTrue(ctx.bindService(intent, connection, Context.BIND_AUTO_CREATE))
        assertTrue("service did not bind", connected.await(20, TimeUnit.SECONDS))
        assertEquals(1, service!!.ping())
    }

    @After
    fun unbind() {
        runCatching { ctx.unbindService(connection) }
        GpuMarker.clear(ctx)
    }

    private class Outcome(val progress: List<String>, val ready: EngineStatus?, val failed: String?, val died: Boolean)

    private fun load(path: String, allowGpu: Boolean, timeoutSec: Long = 90): Outcome {
        val done = CountDownLatch(1)
        val progress = ArrayList<String>()
        val ready = AtomicReference<EngineStatus?>()
        val failed = AtomicReference<String?>()
        service!!.loadEngine(path, allowGpu, object : IEngineCallback.Stub() {
            override fun onProgress(stage: String) { synchronized(progress) { progress += stage }; Log.i("SPIKE4", "progress: $stage") }
            override fun onReady(status: EngineStatus) { ready.set(status); done.countDown() }
            override fun onFailed(message: String) { failed.set(message); done.countDown() }
        })
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeoutSec)
        while (System.nanoTime() < deadline && done.count > 0 && died.count > 0) done.await(200, TimeUnit.MILLISECONDS)
        return Outcome(synchronized(progress) { progress.toList() }, ready.get(), failed.get(), died.count == 0L)
    }

    @Test
    fun brokenModelOnGpuFallsBackToCpuAndPersistsMarker() {
        val bogus = File(ctx.cacheDir, "not-a-model.litertlm").apply { writeBytes(ByteArray(4096) { 0x42 }) }

        val first = load(bogus.absolutePath, allowGpu = true)
        Log.i("SPIKE4", "first load: progress=${first.progress} failed=${first.failed} ready=${first.ready} died=${first.died}")
        if (first.died) {
            // Native crash instead of an exception: the :ui EngineWatchdog is the component that writes
            // the marker in that case (it observes binder death); the service itself cannot.
            Log.w("SPIKE4", "inference process died on broken model; watchdog path required")
            assertTrue("marker must not be written by a dead process", !GpuMarker.exists(ctx))
            return
        }
        assertTrue("expected onFailed for a broken model, got ready=${first.ready}", first.failed != null)
        assertTrue("first stage must attempt GPU: ${first.progress}", first.progress.firstOrNull()?.contains("GPU") == true)
        assertTrue("service must report the CPU retry: ${first.progress}", first.progress.any { it.contains("retrying on CPU") })
        assertTrue("GPU failure must persist gpu_disabled.marker", GpuMarker.exists(ctx))
        Log.i("SPIKE4", "marker reason: ${GpuMarker.reason(ctx)}")

        // Second attempt with allowGpu=true must honour the marker and not touch the GPU at all.
        val second = load(bogus.absolutePath, allowGpu = true)
        Log.i("SPIKE4", "second load: progress=${second.progress} failed=${second.failed}")
        assertTrue(second.failed != null && !second.died)
        assertTrue("marker must force CPU: ${second.progress}", second.progress.firstOrNull()?.contains("CPU") == true)
        assertTrue("no GPU retry chatter expected: ${second.progress}", second.progress.none { it.contains("retrying on CPU") })

        // "Retry GPU" clears the marker; the next load tries the GPU again.
        GpuMarker.clear(ctx)
        val third = load(bogus.absolutePath, allowGpu = true)
        assertTrue("after clearing the marker the GPU is attempted again: ${third.progress}", third.progress.firstOrNull()?.contains("GPU") == true)
        assertEquals(EngineStatus.State.FAILED, service!!.engineStatus.state)
    }
}
