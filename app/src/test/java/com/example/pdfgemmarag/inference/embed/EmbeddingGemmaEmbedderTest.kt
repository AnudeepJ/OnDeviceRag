package com.example.pdfgemmarag.inference.embed

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EmbeddingGemmaEmbedderTest {

    @Test
    fun `usable signal rejects zero and non-finite embeddings`() {
        assertFalse(EmbeddingGemmaEmbedder.hasUsableSignal(FloatArray(512), 512))
        assertFalse(EmbeddingGemmaEmbedder.hasUsableSignal(floatArrayOf(Float.NaN, 1f), 2))
        assertFalse(EmbeddingGemmaEmbedder.hasUsableSignal(floatArrayOf(Float.POSITIVE_INFINITY), 1))
    }

    @Test
    fun `usable signal accepts finite non-zero embedding within requested dimensions`() {
        val values = FloatArray(768)
        values[511] = 0.25f
        assertTrue(EmbeddingGemmaEmbedder.hasUsableSignal(values, 512))
    }

    @Test
    fun `usable signal ignores values outside selected Matryoshka dimensions`() {
        val values = FloatArray(768)
        values[700] = 1f
        assertFalse(EmbeddingGemmaEmbedder.hasUsableSignal(values, 512))
    }
}
