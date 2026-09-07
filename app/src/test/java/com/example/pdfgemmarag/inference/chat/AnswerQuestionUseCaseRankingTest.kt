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
            "What tolerance applies to the cross-sectional dimensions of the column?",
            listOf(table),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("Cross section of columns"))
        assertTrue(answer.text.contains("±1/2”"))
        assertTrue(answer.text.contains("-1/4”"))
        assertEquals(false, answer.text.contains("deck slabs"))
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
