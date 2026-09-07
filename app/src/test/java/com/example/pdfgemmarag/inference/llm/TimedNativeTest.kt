package com.example.pdfgemmarag.inference.llm

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TimedNativeTest {
    @Test
    fun `returns true when the call finishes inside the timeout`() {
        assertTrue(TimedNative.run(2, "quick") { })
    }

    @Test
    fun `returns false when the call exceeds the timeout`() {
        val started = System.nanoTime()
        val completed = TimedNative.run(1, "sleep") { Thread.sleep(10_000) }
        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        assertFalse(completed)
        assertTrue("elapsedMs=$elapsedMs", elapsedMs in 700..5_000)
    }
}
