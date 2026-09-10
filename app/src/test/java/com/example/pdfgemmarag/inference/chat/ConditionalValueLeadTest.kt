package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConditionalValueLeadTest {

    private fun citation(text: String, page: Int = 105, index: Int = 1) = Citation(
        "c$index", "doc", page, index, 1.0, text, indexNamespace = "x", contentKind = "PARAGRAPH",
    )

    @Test
    fun `unique typed class sentence keeps the value and its qualifier`() {
        val evidence = citation(
            "Maximum allowable slope for excavations less than 20 feet (6Mt) deep in Type B soil is 1:1. " +
                "Type C soil uses a different ratio.",
        )
        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What is the maximum allowable slope for excavations in Type B soil?",
            listOf(evidence),
        )
        assertTrue(lead.decisive)
        assertTrue(lead.text.contains("1:1"))
        assertTrue(lead.text.contains("20 feet"))
        assertTrue(lead.text.contains("6Mt"))
        assertFalse(lead.text.contains("Type C"))
    }

    @Test
    fun `follow up naming another class copies that sentence only`() {
        val evidence = citation(
            "Excavations less than 20 feet deep in Type B soil have a maximum allowable slope of 1:1. " +
                "Excavations less than 20 feet deep in Type C soil have a maximum allowable slope of 1½:1.",
        )
        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What about Type C soil?",
            listOf(evidence),
        )
        assertTrue(lead.decisive)
        assertTrue(lead.text.contains("1½:1") || lead.text.contains("1 1/2:1"))
        assertFalse(lead.text.contains("1:1."))
    }

    @Test
    fun `typed heading binds only its adjacent extracted value chunk`() {
        val typeA = citation("Excavations made in Type A soil", index = 719)
        val typeAValues = citation("All simple slope excavations 20 feet or less have a maximum allowable slope of 3/4:1.", index = 720)
        val typeB = citation("Excavations Made in Type B (Clay and cohesive) Soil", index = 721)
        val typeBValues = citation("All simple slope excavations 20 feet (6Mt) or less in depth will have a maximum allowable slope of 1:1.", index = 722)
        val typeC = citation("Excavations Made in Type C soil", index = 723)
        val typeCValues = citation("All simple slope excavations 20 feet or less have a maximum allowable slope of 1 1/2:1.", index = 724)

        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What is the maximum allowable slope for excavations in Type B soil?",
            listOf(typeCValues, typeA, typeBValues, typeC, typeAValues, typeB),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text, lead.text.contains("Type B"))
        assertTrue(lead.text, lead.text.contains("1:1"))
        assertTrue(lead.text, lead.text.contains("6Mt"))
        assertFalse(lead.text, lead.text.contains("3/4:1"))
        assertFalse(lead.text, lead.text.contains("1 1/2:1"))
        assertEquals(listOf(721, 722), lead.citations.map { it.chunkIndex })
    }

    @Test
    fun `typed follow up binds the adjacent value chunk`() {
        val heading = citation("Excavations Made in Type C (Sandy soil) Soil", index = 723)
        val values = citation("All simple slope excavations 20 feet or less in depth will have a maximum allowable slope of 1 1/2:1.", index = 724)

        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What about Type C soil?",
            listOf(values, heading),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text, lead.text.contains("1 1/2:1"))
        assertEquals(listOf(723, 724), lead.citations.map { it.chunkIndex })
    }

    @Test
    fun `two matching sentences stay unresolved`() {
        val evidence = citation(
            "Type B material less than 20 feet deep may be sloped at 1:1. " +
                "Type B material less than 10 feet deep may be sloped at 1:1.",
        )
        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What is the slope for Type B?",
            listOf(evidence),
        )
        assertFalse(lead.decisive)
    }

    @Test
    fun `a value without a condition is not copied`() {
        val evidence = citation("Type B soil has a slope of 1:1.")
        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What is the slope for Type B?",
            listOf(evidence),
        )
        assertFalse(lead.decisive)
    }

    @Test
    fun `questions without a typed class never trigger`() {
        val evidence = citation("Maximum allowable slope for excavations less than 20 feet deep is 1:1.")
        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What is the maximum allowable slope?",
            listOf(evidence),
        )
        assertFalse(lead.decisive)
        assertEquals("", lead.text)
    }
}
