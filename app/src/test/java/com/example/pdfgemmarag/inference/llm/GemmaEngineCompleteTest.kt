package com.example.pdfgemmarag.inference.llm

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GemmaEngineCompleteTest {
    @Test
    fun `citation ending looks complete`() {
        assertTrue(GemmaEngine.looksComplete("Plywood for formwork must conform to PS 1, Class 1. [Page 17]."))
        assertTrue(GemmaEngine.looksComplete("No separate payment will be made."))
        assertFalse(GemmaEngine.looksComplete("Plywood"))
        assertFalse(GemmaEngine.looksComplete(""))
    }
}
