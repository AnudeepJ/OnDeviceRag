package com.example.pdfgemmarag.spikes

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.inference.service.AiInferenceService
import com.example.pdfgemmarag.inference.service.IAiInferenceService
import com.example.pdfgemmarag.inference.service.IInstallCallback
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
 * Spike 2b on a device/emulator with model files staged under `/data/local/tmp/ondevice-rag/`:
 *
 * ```
 * adb push tokenizer.model embeddinggemma-300M_seq512_mixed-precision.tflite /data/local/tmp/ondevice-rag/
 * ```
 *
 * Installs them through the production `installModel` path (SHA-256 + atomic copy into
 * `filesDir/models`), then runs the service's `runSelfTest()` (tokenizer parity, embedding latency,
 * cosine sanity, AppSearch hybrid) and asserts on the report. Skipped when the files are absent.
 */
@RunWith(AndroidJUnit4::class)
class Spike2DeviceEmbedderTest {

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val staging = File("/data/local/tmp/ondevice-rag")
    private var service: IAiInferenceService? = null
    private val connected = CountDownLatch(1)
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) { service = IAiInferenceService.Stub.asInterface(binder); connected.countDown() }
        override fun onServiceDisconnected(name: ComponentName) { service = null }
    }

    @Before
    fun bind() {
        assumeTrue("no staged models under $staging", File(staging, "tokenizer.model").isFile && File(staging, "embeddinggemma-300M_seq512_mixed-precision.tflite").isFile)
        ctx.bindService(Intent(ctx, AiInferenceService::class.java), connection, Context.BIND_AUTO_CREATE)
        assertTrue(connected.await(20, TimeUnit.SECONDS))
    }

    private fun install(source: File, target: String) {
        if (File(ModelPaths.modelsDir(ctx), target).let { it.isFile && it.length() == source.length() }) return
        val done = CountDownLatch(1)
        val error = AtomicReference<String?>()
        // Empty SHA: the staged copies are dev files; the installer still hashes and reports the digest.
        service!!.installModel(source.absolutePath, target, "", source.length(), true, object : IInstallCallback.Stub() {
            override fun onProgress(bytesCopied: Long, totalBytes: Long) {}
            override fun onInstalled(targetPath: String, sha256: String) { Log.i("SPIKE2", "installed $targetPath sha256=$sha256"); done.countDown() }
            override fun onFailed(message: String) { error.set(message); done.countDown() }
        })
        assertTrue("install of $target timed out", done.await(180, TimeUnit.SECONDS))
        assertTrue("install failed: ${error.get()}", error.get() == null)
    }

    @Test
    fun installEmbeddingModelsAndRunSelfTest() {
        install(File(staging, "tokenizer.model"), ModelPaths.EMBEDDING_TOKENIZER_FILE)
        install(File(staging, "embeddinggemma-300M_seq512_mixed-precision.tflite"), ModelPaths.EMBEDDING_MODEL_FILE)
        assertTrue(ModelPaths.embeddingReady(ctx))

        val report = service!!.runSelfTest()
        Log.i("SPIKE2", "\n$report")
        assertTrue(report, report.contains("round-trip 7/7"))
        assertTrue(report, report.contains("parity 50/50"))
        assertTrue(report, report.contains("sanity ok"))
        assertTrue(report, report.contains("hybridOk=true"))
        assertTrue(report, !report.contains("FAILED") && !report.contains("UNEXPECTED"))
    }
}
