package com.example.pdfgemmarag.inference.chat

import org.junit.Assert.assertEquals
import org.junit.Test
import com.example.pdfgemmarag.core.model.QaPair

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
}
