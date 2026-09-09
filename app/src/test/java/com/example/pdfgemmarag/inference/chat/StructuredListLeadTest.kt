package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Complete-list answers across label styles, chunk boundaries and pages (roadmap M3). */
class StructuredListLeadTest {

    private fun chunk(
        index: Int, text: String, page: Int = 25, kind: String = "PARAGRAPH",
        continuesTo: Int = -1, continuesFrom: Int = -1,
    ) = Citation(
        "c$index", "doc", page, index, 1.0, text, indexNamespace = "doc:v22:x", sectionId = "s",
        contentKind = kind, continuesToChunkIndex = continuesTo, continuesFromChunkIndex = continuesFrom,
    )

    @Test
    fun `roman numbered list with a plain colon introduction returns every requested item`() {
        val intro = chunk(10, "There are some basic steps about how the risk assessment should be undertaken like:")
        val list = chunk(
            11,
            "I. Initiating the review\nII. Identify the hazard\nIII. Identify all parties affected by the hazard and determine how they can be affected\nIV. Evaluate or assess the risk\nV. Monitor & Review",
            kind = "LIST",
        )
        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "What are the five basic steps of the risk assessment?",
            listOf(list, intro),
        )
        assertTrue(lead.decisive)
        assertEquals(5, lead.text.lines().size)
        assertTrue(lead.text.contains("V. Monitor & Review [Page 25]"))
    }

    @Test
    fun `list continuing onto the next page is followed through the continuation edge`() {
        val intro = chunk(10, "Before starting the work the following checks are required:")
        val first = chunk(11, "\u2022 Locate all buried services\n\u2022 Identify overhead power lines", kind = "LIST", continuesTo = 12)
        val second = chunk(12, "\u2022 Test for hazardous gas before entering\n\u2022 Remove water from the excavation", page = 26, kind = "LIST", continuesFrom = 11)
        val unrelated = chunk(13, "\u2022 Unrelated later list item", page = 27, kind = "LIST")

        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "Which checks are required before starting the work?",
            listOf(intro, first, second, unrelated),
        )
        assertTrue(lead.decisive)
        assertEquals(4, lead.text.lines().size)
        assertTrue(lead.text.contains("[Page 26]"))
        assertFalse(lead.text.contains("Unrelated"))
    }

    @Test
    fun `fewer items than the requested count is not a decisive answer`() {
        val intro = chunk(10, "The basic steps are:")
        val list = chunk(11, "1. Only one step", kind = "LIST")
        val lead = AnswerQuestionUseCase.buildNumberedListLead("What are the five basic steps?", listOf(intro, list))
        assertFalse(lead.decisive)
    }

    @Test
    fun `a table reference is not mistaken for a requested item count`() {
        val intro = chunk(10, "The following classes are listed:")
        val list = chunk(11, "A. First class\nB. Second class\nC. Third class", kind = "LIST")
        val lead = AnswerQuestionUseCase.buildNumberedListLead("Which classes are listed in Table 1.1?", listOf(intro, list))
        assertTrue(lead.decisive)
        assertEquals(3, lead.text.lines().size)
    }
}
