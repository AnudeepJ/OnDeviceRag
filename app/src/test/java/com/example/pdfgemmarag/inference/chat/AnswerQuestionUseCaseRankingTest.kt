package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AnswerQuestionUseCaseRankingTest {
    @Test
    fun `manifest scoped fallback ranks the requested facts first`() {
        val unrelated = citation(1, "Admixtures shall comply with the referenced standard")
        val target = citation(2, "Maximum allowable water cement ratios are 0.45, 0.40, and 0.55")

        val ranked = AnswerQuestionUseCase.rankDirectScope(
            "List the maximum water cement ratios in specification 03310",
            listOf(unrelated, target),
        )

        assertEquals(2, ranked.first().chunkIndex)
        assertEquals(true, ranked.first().score > ranked.last().score)
    }

    @Test
    fun `section lookup resolves a uniquely matching printed cross reference`() {
        val source = citation(4, "References: 1. Section 01440 - Inspection Services 2. Section 01450 – Testing Laboratory Services")

        val answer = AnswerQuestionUseCase.buildSectionPointerAnswer(
            "What section covers the testing laboratory services?",
            listOf(source),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("Section 01450"))
        assertTrue(answer.text.contains("Testing Laboratory Services"))
    }

    @Test
    fun `list question resolves adjacent enumerated value block without generation`() {
        val heading = citation(10, "1. Maximum allowable ratios shall be as follows:")
        val values = citation(11, "a. Liquid-containing: 0.45. b. Brackish water: 0.40. c. Other: 0.55. 2. Continue procedure.")

        val answer = AnswerQuestionUseCase.buildEnumeratedValueAnswer(
            "List the maximum allowable ratios",
            listOf(heading, values),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("0.45"))
        assertTrue(answer.text.contains("0.40"))
        assertTrue(answer.text.contains("0.55"))
        assertEquals(false, answer.text.contains("Continue procedure"))
    }

    @Test
    fun `summary preserves a unique adjacent multi-value block`() {
        val heading = citation(10, "E. Maximum allowable ratios shall be as follows:")
        val values = citation(11, "a. First: 0.45. b. Second: 0.40. c. Other: 0.55. 2. Continue procedure.")

        val lead = AnswerQuestionUseCase.buildEnumeratedSummaryLead(listOf(heading, values))

        assertEquals(false, lead.decisive)
        assertTrue(lead.text.contains("0.45"))
        assertTrue(lead.text.contains("0.40"))
        assertTrue(lead.text.contains("0.55"))
        assertEquals(false, lead.text.contains("Continue procedure"))
    }

    @Test
    fun `enumerated ratios do not hijack a nearby slump question`() {
        val heading = citation(10, "E. Water-Cement Ratios: Maximum allowable water-cement ratios shall be as follows:")
        val values = citation(11, "a. Liquid-containing: 0.45. b. Brackish water: 0.40. c. Other: 0.55.")

        val answer = AnswerQuestionUseCase.buildEnumeratedValueAnswer(
            "What are the minimum and maximum slump values for Portland cement concrete and concrete dosed with superplasticizer",
            listOf(heading, values),
        )

        assertEquals(false, answer.decisive)
        assertTrue(answer.text.isEmpty())
    }

    @Test
    fun `tolerance question returns one uniquely matching table row`() {
        val table = citation(
            358,
            "| Grade | Top surfaces of curbs and railings | 3/16” in 10’ | | --- | --- | --- | " +
                "| Drawing Dimensions | Cross section of columns, caps, walls, beams, and similar members | ±1/2”, -1/4” | " +
                "|  | Thickness of deck slabs | ±1/4”, -1/8” |",
        ).copy(contentKind = "TABLE")

        val answer = AnswerQuestionUseCase.buildTableLead(
            "Got it. What about the cross-sectional dimensions? How much tolerance do we have on the column thickness?",
            listOf(
                table,
                table.copy(
                    chunkId = "duplicate",
                    pageNumber = 12,
                    score = 0.9,
                    text = table.text.replace("±", "+").replace("”", "\""),
                ),
            ),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("Cross section of columns"))
        assertTrue(answer.text.contains("+1/2\""))
        assertTrue(answer.text.contains("-1/4\""))
        assertEquals(false, answer.text.contains("deck slabs"))
    }

    @Test
    fun `variation question returns exact value from flattened schedule paragraph`() {
        val paragraph = citation(
            357,
            "Variation In Maximum From Plumb of Specified Surfaces of columns, piers and walls 1/2\" in 10’ Batter Level or Top surfaces of slabs",
        )

        val answer = AnswerQuestionUseCase.buildTableLead(
            "What is the maximum allowable variation from plumb for the new piers?",
            listOf(paragraph),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("1/2\" in 10’"))
        assertEquals(false, answer.text.contains("Top surfaces of slabs"))
    }

    @Test
    fun `curb tolerance is not hijacked by plumb paragraph through function words`() {
        val paragraph = citation(
            357,
            "Variation In Maximum From Plumb of Specified Surfaces of columns, piers and walls 1/2\" in 10’",
        )
        val table = citation(
            358,
            "| Grade | Top surfaces of curbs and railings | 3/16” in 10’ |",
        ).copy(contentKind = "TABLE")

        val answer = AnswerQuestionUseCase.buildTableLead(
            "Okay, and what is the tolerance for the level of the top surfaces of the curbs?",
            listOf(paragraph, table),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("3/16” in 10’"))
        assertEquals(false, answer.text.contains("From Plumb"))
    }

    private fun citation(index: Int, text: String) = Citation(
        chunkId = "c$index",
        docHash = "doc",
        pageNumber = 62,
        chunkIndex = index,
        score = 0.0,
        text = text,
        sectionPath = "SECTION 03310 pencils",
    )
}
