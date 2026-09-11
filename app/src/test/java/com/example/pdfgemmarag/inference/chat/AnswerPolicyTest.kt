package com.example.pdfgemmarag.inference.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AnswerPolicyTest {
    @Test
    fun `recognizes procedural wording without document-specific terms`() {
        listOf(
            "List the required actions",
            "What is the priority order of the controls?",
            "How should this be performed?",
            "What must be done before starting?",
            "Describe the procedure",
        ).forEach { question -> assertTrue(question, AnswerPolicy.isProcedural(question)) }
        assertTrue(AnswerPolicy.isProcedural("What emergencies must a construction-site emergency plan consider?"))
    }

    @Test
    fun `does not classify explanatory fact merely by substring`() {
        assertFalse(AnswerPolicy.isProcedural("What explains the reported value?"))
        assertFalse(AnswerPolicy.isProcedural("What is the concrete strength?"))
    }

    @Test
    fun `prompt and generation can share the same procedural decision`() {
        val question = "List all five steps"

        assertTrue(AnswerPolicy.isProcedural(question))
        assertEquals(288, AnswerPolicy.maxOutputTokens(QuestionIntent.FACT, question))
        assertEquals(160, AnswerPolicy.maxOutputTokens(QuestionIntent.FACT, "What is the value?"))
    }
}
