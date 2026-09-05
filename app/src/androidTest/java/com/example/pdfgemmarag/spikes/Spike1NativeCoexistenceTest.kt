package com.example.pdfgemmarag.spikes

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Spike 1: `litertlm-android` and `litert` both ship `libLiteRt.so`; the APK keeps one copy
 * (`pickFirsts`) and both JNI shims must resolve against it inside one process without a linker error.
 */
@RunWith(AndroidJUnit4::class)
class Spike1NativeCoexistenceTest {

    @Test
    fun bothJniLibrariesLoadIntoOneProcess() {
        val loaded = ArrayList<String>()
        val errors = ArrayList<String>()
        for (lib in listOf("LiteRt", "litertlm_jni", "litert_jni")) {
            try {
                System.loadLibrary(lib)
                loaded += lib
            } catch (t: Throwable) {
                errors += "$lib: ${t.javaClass.simpleName} ${t.message}"
            }
        }
        // Touch the Kotlin entry points too so their static initialisers run.
        val litertlmClass = runCatching { Class.forName("com.google.ai.edge.litertlm.Engine") }
        val litertClass = runCatching { Class.forName("com.google.ai.edge.litert.CompiledModel") }
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val env = runCatching { com.google.ai.edge.litert.Environment.create(ctx) }
        env.getOrNull()?.close()

        // With useLegacyPackaging=false the .so files are mmapped straight out of base.apk, so
        // /proc/self/maps shows the APK path rather than individual library names.
        val maps = File("/proc/self/maps").readLines()
        val mappedSo = maps.mapNotNull { Regex("(/[^ ]+\\.so)$").find(it)?.groupValues?.get(1) }
            .filter { it.contains("litert", ignoreCase = true) }.toSortedSet()
        val apkMaps = maps.count { it.contains("base.apk") && it.contains("r-xp") }
        val report = buildString {
            appendLine("loaded=$loaded errors=$errors")
            appendLine("Engine class=${litertlmClass.isSuccess} CompiledModel class=${litertClass.isSuccess} Environment=${env.isSuccess} ${env.exceptionOrNull()?.message ?: ""}")
            appendLine("executable APK maps=$apkMaps; .so paths=$mappedSo")
        }
        Log.i("SPIKE1", report)
        assertTrue(report, errors.isEmpty())
        assertTrue(report, litertlmClass.isSuccess && litertClass.isSuccess && env.isSuccess)
        assertTrue(report, apkMaps > 0 || mappedSo.any { it.endsWith("/libLiteRt.so") })
        // A second libLiteRt.so extracted to disk would indicate pickFirsts did not take effect.
        assertTrue(report, mappedSo.count { it.endsWith("/libLiteRt.so") } <= 1)
    }
}
