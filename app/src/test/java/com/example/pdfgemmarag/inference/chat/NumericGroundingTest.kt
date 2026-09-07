package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.QaPair
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NumericGroundingTest {
    @Test
    fun `independent short question is not rewritten as follow up`() {
        val question = "Summarize all concrete curing requirements"
        val history = listOf(QaPair("What are the water cement ratios?", "0.45, 0.40, 0.55"))
        assertEquals(question, AnswerQuestionUseCase.rewriteForRetrieval(question, history))
    }

    @Test
    fun `pronoun follow up includes previous question`() {
        val history = listOf(QaPair("What is the curing period?", "Seven days"))
        val rewritten = AnswerQuestionUseCase.rewriteForRetrieval("When does it start?", history)
        assertEquals(
            "Previous question: What is the curing period?\nCurrent question: When does it start?",
            rewritten,
        )
    }

    @Test
    fun `repairs one unambiguous truncated source decimal`() {
        val source = "Ratios are 0.45, 0.40, and 0.55."
        val answer = "They are 0.45, 0.40, and 0.5."
        assertEquals(
            "They are 0.45, 0.40, and 0.55.",
            AnswerQuestionUseCase.correctTruncatedDecimals(answer, source),
        )
    }

    @Test
    fun `does not guess when multiple source decimals share prefix`() {
        val source = "Options are 2.50 and 2.55."
        val answer = "The value is 2.5."
        assertEquals(answer, AnswerQuestionUseCase.correctTruncatedDecimals(answer, source))
    }

    @Test
    fun `leaves exact source decimals unchanged`() {
        val source = "Maximum ratio is 0.55."
        val answer = "Use 0.55 [Page 62]."
        assertEquals(answer, AnswerQuestionUseCase.correctTruncatedDecimals(answer, source))
    }

    @Test
    fun `table lead prefers rows matching query numbers and includes neighbour`() {
        fun citation(id: String, index: Int, text: String, kind: String = "TABLE") = Citation(
            id, "doc", 63, index, 0.0, text,
            indexNamespace = "doc:v21:x", sectionId = "classes", contentKind = kind,
        )
        val candidates = listOf(
            citation("header", 10, "Class | Strength | Cement", "PARAGRAPH"),
            citation("other", 11, "Class A | 4000 | 564"),
            citation("target", 12, "Class D | 5,000 | 658", "PARAGRAPH"),
            citation("following", 13, "Class E | 3000 | 470"),
        )

        val lead = AnswerQuestionUseCase.buildTableLead(
            "Which class has 5000 psi and 658 pounds of cement?",
            candidates,
        )

        assertTrue(lead.text.contains("Class D"))
        assertTrue(lead.text.contains("5,000"))
        assertTrue(lead.text.contains("658"))
        assertTrue(lead.citations.any { it.chunkId == "target" })
        assertTrue(lead.decisive)
    }

    @Test
    fun `table header and paired limits request structural neighbours even when labeled paragraph`() {
        fun citation(text: String) = Citation(
            "c", "doc", 62, 1, 0.0, text,
            indexNamespace = "doc:v21:x", contentKind = "PARAGRAPH",
        )

        assertTrue(AnswerQuestionUseCase.looksLikeStructuredFragment(citation("Concrete Type Portland Cement Concrete")))
        assertTrue(AnswerQuestionUseCase.looksLikeStructuredFragment(citation("Minimum Slump 1 inch Maximum Slump 3 inches")))
        assertTrue(AnswerQuestionUseCase.looksLikeStructuredFragment(citation("Variation from plumb or drawing dimensions")))
        assertTrue(AnswerQuestionUseCase.looksLikeStructuredFragment(citation("Cross section of columns, walls, and beams")))
        assertTrue(AnswerQuestionUseCase.looksLikeStructuredFragment(citation("Tolerance for formed surfaces")))
    }

    @Test
    fun `ordinary ratio fact does not emit an unrelated available table`() {
        val fact = Citation("fact", "doc", 62, 1, 5.0, "Maximum water-cement ratios", contentKind = "LIST")
        val table = Citation("table", "doc", 63, 2, 0.0, "| D | 5,000 | 658 |", contentKind = "TABLE")

        val lead = AnswerQuestionUseCase.buildTableLead(
            "List the maximum water cement ratios in specification 03310",
            listOf(fact, table),
        )

        assertTrue(lead.text.isEmpty())
    }
}
