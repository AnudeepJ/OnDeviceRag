package com.example.pdfgemmarag.eval

import android.util.Log
import androidx.appsearch.app.EmbeddingVector
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.PdfChunkDocument
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.random.Random

/**
 * Indexes the eval corpus with synthetic embeddings (keyword + ranking path) and reports page-hit@5.
 * When EmbeddingGemma is installed, [com.example.pdfgemmarag.inference.service.SelfTest] runs the
 * same dataset with real query vectors.
 */
@RunWith(AndroidJUnit4::class)
class RetrievalEvalDeviceTest {

    private lateinit var store: AppSearchVectorStore
    private val ns = RetrievalEvalDataset.DOC_HASH + "-${System.currentTimeMillis()}"

    @Before
    fun open(): Unit = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        store = AppSearchVectorStore.open(ctx, "eval_harness_db", enforceProcess = false)
        val now = System.currentTimeMillis()
        val docs = RetrievalEvalDataset.chunks().map { c ->
            PdfChunkDocument().apply {
                namespace = ns
                id = "$ns:${c.chunkIndex}"
                creationTimestampMillis = now
                text = c.text
                pageNumber = c.page
                chunkIndex = c.chunkIndex
                docName = RetrievalEvalDataset.DOC_NAME
                script = c.lang
                pageCount = 250
                embedding = EmbeddingVector(rand(c.chunkIndex), EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
            }
        }
        store.putChunks(docs)
    }

    @After
    fun close(): Unit = runBlocking {
        runCatching { store.removeDocument(ns) }
        store.close()
    }

    @Test
    fun pageHitAt5OverFullDataset(): Unit = runBlocking {
        val report = RetrievalEval.run(
            store = store,
            docHash = ns,
            embed = { rand(it.hashCode()) },
            topK = 5,
            similarityFloor = 0.0,
            keywordWeight = 0.2,
        )
        Log.i("EVAL", report.toString())
        assertEquals(210, report.results.size)
        RetrievalEvalDataset.LANGS.forEach { lang ->
            val row = report.byLanguage.first { it.lang == lang }
            assertEquals(30, row.n)
            assertTrue("$lang hit@5 too low: $row", row.hitAt5 >= 0.7)
        }
        assertTrue("overall hit@5 too low: $report", report.overallHitAt5 >= 0.75)
    }

    private fun rand(seed: Int): FloatArray {
        val r = Random(seed)
        val v = FloatArray(512) { r.nextFloat() - 0.5f }
        EmbeddingGemmaEmbedder.l2Normalize(v)
        return v
    }
}
