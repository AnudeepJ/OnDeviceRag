package com.example.pdfgemmarag.inference.llm

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.chat.QuestionIntent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextSelectorTest {
    @Test
    fun `uses stable excerpt ids and exact boundary dedupe`() {
        val first = citation("c1", 17, "Requirement begins and continues with the same exact boundary text")
        val second = citation("c2", 18, "the same exact boundary text and finishes on the next page")
        val selected = ContextSelector().select("What is required?", listOf(first, second), QuestionIntent.FACT)
        assertEquals(listOf("E1", "E2"), selected.excerpts.map { it.excerptId })
        assertTrue(selected.prompt.contains("[E1]"))
        assertTrue(selected.prompt.contains("Page: 17"))
        assertTrue(selected.prompt.contains("Page: 18"))
        assertEquals(1, Regex("the same exact boundary text").findAll(selected.prompt).count())
    }

    @Test
    fun `summary keeps anchors then prioritizes high information requirements`() {
        val input = listOf(
            citation("p1-a", 1, "one a"),
            citation("p1-b", 1, "one b"),
            citation("p1-c", 1, "one c"),
            citation("p2-a", 2, "Maximum water cement ratio 0.45"),
            citation("p2-b", 2, "general procedure"),
            citation("p3-a", 3, "Air content 4 to 6 percent"),
        )

        val ordered = ContextSelector().coverageOrder(input)

        assertEquals(listOf("p1-a", "p3-a", "p2-a", "p1-b", "p1-c", "p2-b"), ordered.map { it.chunkId })
    }

    @Test
    fun `summary prioritizes a continued multi-decimal requirement block`() {
        val heading = citation("ratio-heading", 1, "Maximum allowable ratios shall be as follows")
        val values = citation("ratio-values", 2, "a. First condition: 0.45. b. Second condition: 0.40. c. Other: 0.55.")
        val prose = citation("prose", 2, "General procedural requirement with one reference 318")

        assertTrue(ContextSelector().summaryPriority(values) > ContextSelector().summaryPriority(heading))
        assertTrue(ContextSelector().summaryPriority(values) > ContextSelector().summaryPriority(prose))
    }

    @Test
    fun `summary keeps opening anchors then presents high information nearest the question`() {
        val heading = citation("heading", 1, "2.02 CONCRETE MIX").copy(contentKind = "HEADING")
        val opening = citation("opening", 1, "General objective and testing responsibility")
        val generic = citation("generic", 2, "General mixing procedure")
        val ratio = citation("ratio", 2, "Maximum water cement ratio is 0.45")

        val selected = ContextSelector().select(
            "Summarize the mix",
            listOf(heading, opening, generic, ratio),
            QuestionIntent.SECTION_SUMMARY,
        )

        assertEquals(false, selected.excerpts.any { it.chunkId == "heading" })
        assertEquals("opening", selected.excerpts.first().chunkId)
        assertTrue(selected.excerpts.indexOfFirst { it.chunkId == "ratio" } > selected.excerpts.indexOfFirst { it.chunkId == "generic" })
    }

    @Test
    fun `fact puts the excerpt with literal question coverage nearest the question`() {
        val generic = citation("generic", 45, "Curing must continue for two weeks")
        val target = citation("target", 45, "When ambient temperature falls below 32 degrees F protect the concrete")

        val ordered = ContextSelector().factPresentationOrder(
            "What about when ambient temperature is below freezing?",
            listOf(target, generic),
        )

        assertEquals("target", ordered.last().chunkId)
    }

    private fun citation(id: String, page: Int, text: String) = Citation(
        id, "doc", page, page, 1.0, text,
        indexNamespace = "doc:v21:x", sectionId = "s1", sectionPath = "Section A",
    )
}
