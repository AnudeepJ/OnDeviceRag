package com.example.pdfgemmarag.eval

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RetrievalEvalTest {

    @Test
    fun `dataset has 30 questions per language`() {
        val qs = RetrievalEvalDataset.questions()
        assertEquals(210, qs.size)
        RetrievalEvalDataset.LANGS.forEach { lang ->
            assertEquals("30 questions for $lang", 30, qs.count { it.lang == lang })
        }
        assertEquals(15, RetrievalEvalDataset.facts.size)
        assertEquals(105, RetrievalEvalDataset.chunks().size)
    }

    @Test
    fun `page-hit at K and MRR`() {
        val expected = setOf(42, 43)
        assertTrue(RetrievalEval.pageHitAtK(listOf(42, 1, 2, 3, 4), expected, 1))
        assertTrue(RetrievalEval.pageHitAtK(listOf(9, 42, 1, 2, 3), expected, 5))
        assertFalse(RetrievalEval.pageHitAtK(listOf(9, 8, 7, 6, 5), expected, 5))
        assertFalse(RetrievalEval.pageHitAtK(listOf(9, 42), expected, 1))
        assertEquals(0.5, RetrievalEval.reciprocalRank(listOf(9, 42), expected), 1e-9)
        assertEquals(0.0, RetrievalEval.reciprocalRank(listOf(1, 2, 3), expected), 1e-9)
    }

    @Test
    fun `summarise reports per-language page-hit at 5`() {
        val q = RetrievalEvalDataset.questions().first { it.lang == "en" && it.factId == "revenue" }
        val hit = RetrievalEval.score(q, listOf(cit(5), cit(99)))
        val miss = RetrievalEval.score(q, listOf(cit(99), cit(100)))
        val report = RetrievalEval.summarise(listOf(hit, miss), 12)
        assertEquals(0.5, report.overallHitAt1, 1e-9)
        assertEquals(0.5, report.overallHitAt5, 1e-9)
        assertEquals(1, report.byLanguage.size)
        assertTrue(report.toString().contains("hit@5"))
    }

    private fun cit(page: Int) = Citation("c$page", "doc", page, page, 1.0, "text $page")
}
