package com.example.pdfgemmarag.eval

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.inference.service.AiInferenceService
import com.example.pdfgemmarag.inference.service.IAiInferenceService
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeFalse
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Binds the real `:inference` service and probes whatever PDF is already in AppSearch.
 * Does not open LocalStorage from this process (that would corrupt the index).
 */
@RunWith(AndroidJUnit4::class)
class LiveRetrievalProbeTest {

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private var service: IAiInferenceService? = null
    private val connected = CountDownLatch(1)
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            service = IAiInferenceService.Stub.asInterface(binder)
            connected.countDown()
        }
        override fun onServiceDisconnected(name: ComponentName) { service = null }
    }

    @Before
    fun bind() {
        val intent = Intent(ctx, AiInferenceService::class.java)
        assertTrue(ctx.bindService(intent, connection, Context.BIND_AUTO_CREATE))
        assertTrue("service did not bind", connected.await(20, TimeUnit.SECONDS))
        assertTrue(service!!.ping() == 1)
    }

    @After
    fun unbind() {
        runCatching { ctx.unbindService(connection) }
    }

    @Test
    fun probeIndexedBaytownPdf() {
        val diag = service!!.diagnostics
        Log.i(TAG, "diagnostics:\n$diag")
        val report = service!!.probeRetrieval("")
        Log.i(TAG, "probe:\n$report")
        File(ctx.getExternalFilesDir(null), "live_probe.txt").writeText(report)
        assumeFalse("no document in AppSearch — re-index the PDF after install", report.contains("INDEX EMPTY"))
        assertTrue("Q1 should hit Baytown/date, got:\n$report", report.contains("Q1 city+date: HIT") || report.contains("baytown", ignoreCase = true))
        assertTrue("formwork keyword should hit, got:\n$report", report.contains("KW formwork: HIT") || report.contains("Q2 formwork payment: HIT"))
        assertTrue("plywood should hit, got:\n$report", report.contains("KW plywood: HIT") || report.contains("Q3 plywood standard: HIT"))
    }

    companion object { private const val TAG = "LIVE_PROBE" }
}
