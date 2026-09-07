package com.example.pdfgemmarag.spikes

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.EngineStatus
import com.example.pdfgemmarag.core.model.GenerationStats
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.core.model.QaPair
import com.example.pdfgemmarag.inference.service.AiInferenceService
import com.example.pdfgemmarag.inference.service.IAiInferenceService
import com.example.pdfgemmarag.inference.service.IEngineCallback
import com.example.pdfgemmarag.inference.service.IInstallCallback
import com.example.pdfgemmarag.inference.service.IStreamCallback
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Phase 2 acceptance on a device/emulator: engine load through AIDL, streaming tokens tagged with
 * the generationId, clean completion stats, and cancel mid-generation. Uses whichever `.litertlm`
 * is already installed, or stages one from `/data/local/tmp/ondevice-rag/` when necessary (any
 * LiteRT-LM model; Qwen3-0.6B on the emulator, Gemma 4 E2B on a phone).
 */
@RunWith(AndroidJUnit4::class)
class Phase2GenerationDeviceTest {

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val staging = File("/data/local/tmp/ondevice-rag")
    private var service: IAiInferenceService? = null
    private val connected = CountDownLatch(1)
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) { service = IAiInferenceService.Stub.asInterface(binder); connected.countDown() }
        override fun onServiceDisconnected(name: ComponentName) { service = null }
    }
    private lateinit var modelFile: File

    @Before
    fun bind() {
        val installed = ModelPaths.installedLlms(ctx).firstOrNull()
        val staged = staging.listFiles { f -> f.name.endsWith(ModelPaths.LLM_EXTENSION) }?.firstOrNull()
        assumeTrue("no installed or staged .litertlm model", installed != null || staged != null)
        ctx.bindService(Intent(ctx, AiInferenceService::class.java), connection, Context.BIND_AUTO_CREATE)
        assertTrue(connected.await(20, TimeUnit.SECONDS))
        modelFile = installed ?: File(ModelPaths.modelsDir(ctx), requireNotNull(staged).name)
        if (installed == null) {
            val source = requireNotNull(staged)
            val done = CountDownLatch(1)
            val error = AtomicReference<String?>()
            service!!.installModel(source.absolutePath, source.name, "", source.length(), true, object : IInstallCallback.Stub() {
                override fun onProgress(bytesCopied: Long, totalBytes: Long) {}
                override fun onInstalled(targetPath: String, sha256: String) { Log.i("PHASE2", "installed $targetPath sha256=$sha256"); done.countDown() }
                override fun onFailed(message: String) { error.set(message); done.countDown() }
            })
            assertTrue(done.await(300, TimeUnit.SECONDS)); assertTrue("install failed: ${error.get()}", error.get() == null)
        }
        loadEngine()
    }

    private fun loadEngine() {
        if (service!!.engineStatus.let { it.state == EngineStatus.State.READY && it.modelPath == modelFile.absolutePath }) return
        val done = CountDownLatch(1)
        val failure = AtomicReference<String?>()
        val t0 = SystemClock.elapsedRealtime()
        // The emulator's SwiftShader "GPU" initialises but generates nothing usable; force CPU there.
        val allowGpu = !isEmulator()
        service!!.loadEngine(modelFile.absolutePath, allowGpu, object : IEngineCallback.Stub() {
            override fun onProgress(stage: String) { Log.i("PHASE2", "load: $stage") }
            override fun onReady(status: EngineStatus) { Log.i("PHASE2", "ready on ${status.backend} in ${SystemClock.elapsedRealtime() - t0}ms"); done.countDown() }
            override fun onFailed(message: String) { failure.set(message); done.countDown() }
        })
        assertTrue("engine load timed out", done.await(600, TimeUnit.SECONDS))
        assertTrue("engine load failed: ${failure.get()}", failure.get() == null)
    }

    private fun isEmulator(): Boolean =
        android.os.Build.HARDWARE.contains("ranchu") || android.os.Build.HARDWARE.contains("goldfish") || android.os.Build.FINGERPRINT.contains("generic")

    private class Stream {
        val tokens = StringBuilder()
        val ids = HashSet<Long>()
        val done = CountDownLatch(1)
        var stats: GenerationStats? = null
        var error: String? = null
        var firstTokenAt = -1L
        val callback = object : IStreamCallback.Stub() {
            override fun onRetrieved(generationId: Long, citations: List<Citation>) { ids += generationId }
            override fun onToken(generationId: Long, token: String) {
                ids += generationId
                if (firstTokenAt < 0) firstTokenAt = SystemClock.elapsedRealtime()
                synchronized(tokens) { tokens.append(token) }
            }
            override fun onDone(generationId: Long, s: GenerationStats) { ids += generationId; stats = s; done.countDown() }
            override fun onError(generationId: Long, message: String) { ids += generationId; error = message; done.countDown() }
        }
    }

    @Test
    fun plainChatStreamsTokensTaggedWithGenerationId() {
        val stream = Stream()
        val t0 = SystemClock.elapsedRealtime()
        val id = service!!.ask("", "Reply with exactly the words: hello from the phone", emptyList<QaPair>(), stream.callback)
        assertTrue("generation did not finish", stream.done.await(600, TimeUnit.SECONDS))
        val text = synchronized(stream.tokens) { stream.tokens.toString() }
        Log.i("PHASE2", "gen $id: ttft=${stream.firstTokenAt - t0}ms total=${SystemClock.elapsedRealtime() - t0}ms stats=${stream.stats}\n$text")
        assertTrue("error: ${stream.error}", stream.error == null)
        assertEquals(setOf(id), stream.ids)
        assertTrue("no tokens streamed", text.isNotBlank())
        val stats = stream.stats!!
        assertEquals(id, stats.generationId)
        assertTrue(!stats.cancelled)
        assertTrue(stats.retrievedChunks == 0)
        assertTrue("backend=${stats.backend}", stats.backend == "CPU" || stats.backend == "GPU")
    }

    @Test
    fun historyIsCarriedIntoTheNextTurn() {
        val stream = Stream()
        val history = listOf(QaPair("My name is Anudeep and my favourite colour is teal.", "Nice to meet you, Anudeep."))
        service!!.ask("", "What is my favourite colour? Answer with one word.", history, stream.callback)
        assertTrue(stream.done.await(600, TimeUnit.SECONDS))
        val text = synchronized(stream.tokens) { stream.tokens.toString() }
        Log.i("PHASE2", "history turn: $text")
        assertTrue(text, text.contains("teal", ignoreCase = true))
    }

    @Test
    fun cancelThenImmediateReaskWaitsForNativeClose() {
        val stream = Stream()
        val id = service!!.ask("", "Write a very long story about a dragon, at least 800 words.", emptyList<QaPair>(), stream.callback)
        // Wait for the first token, then cancel.
        val deadline = SystemClock.elapsedRealtime() + 300_000
        while (stream.firstTokenAt < 0 && SystemClock.elapsedRealtime() < deadline && stream.done.count > 0) Thread.sleep(50)
        assertTrue("never produced a token", stream.firstTokenAt > 0)
        val lengthAtCancel = synchronized(stream.tokens) { stream.tokens.length }
        service!!.cancelGeneration(id)

        // Match the UI race precisely: it unlocks the composer as soon as Stop is tapped, before
        // LiteRT-LM's asynchronous cancelProcess()/close has delivered the first terminal callback.
        val next = Stream()
        val id2 = service!!.ask("", "Say OK.", emptyList<QaPair>(), next.callback)

        assertTrue("onDone/onError not delivered after cancel", stream.done.await(60, TimeUnit.SECONDS))
        Thread.sleep(1500) // stragglers must be suppressed
        val finalLength = synchronized(stream.tokens) { stream.tokens.length }
        Log.i("PHASE2", "cancel: tokens at cancel=$lengthAtCancel final=$finalLength stats=${stream.stats} error=${stream.error}")
        assertTrue("expected cancelled stats, got error=${stream.error}", stream.stats?.cancelled == true)
        assertTrue("stream kept growing long after cancel", finalLength - lengthAtCancel < 400)
        assertEquals(EngineStatus.State.READY, service!!.engineStatus.state)
        assertTrue(next.done.await(600, TimeUnit.SECONDS))
        assertTrue("immediate re-ask failed: ${next.error}", next.error == null && id2 != id)
        assertTrue("race escaped service gate", next.error?.contains("still finishing", ignoreCase = true) != true)
    }
}
