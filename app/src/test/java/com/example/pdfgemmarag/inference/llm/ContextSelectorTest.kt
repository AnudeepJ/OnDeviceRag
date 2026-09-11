package com.example.pdfgemmarag.inference.llm

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.chat.QuestionIntent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    @Test
    fun `document overview preserves structural breadth instead of fact ranking`() {
        val candidates = (1..6).map { index ->
            citation("topic-$index", index, "Top-level topic $index and its principal scope")
                .copy(contentKind = if (index % 2 == 1) "HEADING" else "PARAGRAPH", sectionId = "s$index")
        }

        val selected = ContextSelector(overviewBudget = 2_200).select(
            "Summarize this document",
            candidates,
            QuestionIntent.DOCUMENT_OVERVIEW,
        )

        assertEquals(candidates.map { it.chunkId }, selected.excerpts.map { it.chunkId })
        assertTrue(selected.prompt.contains("document overview"))
        assertTrue(selected.prompt.contains("key points, not a complete summary"))
    }

    @Test
    fun `list question gets complete procedural prompt instead of short fact limit`() {
        val selected = ContextSelector().select(
            "List all five required steps",
            listOf(citation("steps", 25, "I. Start II. Inspect III. Assess IV. Control V. Review")),
            QuestionIntent.FACT,
        )

        assertTrue(selected.prompt.contains("Preserve their source order"))
        assertTrue(selected.prompt.contains("do not silently omit an item"))
        assertFalse(selected.prompt.contains("at most 3 short sentences"))
    }

    @Test
    fun `proximity scored ordinary neighbor survives relative score filtering`() {
        val primary = citation("primary", 45, "Employees receive safety and health training").copy(score = 1.0)
        val neighbor = citation("neighbor", 45, "The recommended duration is not less than 48 hours")
            .copy(score = 0.85)

        val selected = ContextSelector().select(
            "What is the recommended training duration?",
            listOf(primary, neighbor),
            QuestionIntent.FACT,
        )

        assertEquals(setOf("primary", "neighbor"), selected.excerpts.map { it.chunkId }.toSet())
    }

    @Test
    fun `expanded neighbours cannot consume the primary retrieval quota`() {
        val candidates = buildList {
            repeat(4) { rank ->
                add(citation("primary-$rank", 160 + rank, "Primary evidence $rank for protective equipment policy"))
                add(
                    citation("neighbor-$rank-a", 160 + rank, "Adjacent detail A for result $rank")
                        .copy(retrievalProvenance = "ADJACENT", sourceChunkId = "primary-$rank", score = 0.85),
                )
                add(
                    citation("neighbor-$rank-b", 160 + rank, "Adjacent detail B for result $rank")
                        .copy(retrievalProvenance = "ADJACENT", sourceChunkId = "primary-$rank", score = 0.85),
                )
            }
        }

        val selected = ContextSelector(maxFactPrimary = 4, maxFactExcerpts = 8).select(
            "When is protective equipment required?",
            candidates,
            QuestionIntent.FACT,
        )

        assertTrue(selected.excerpts.map { it.chunkId }.containsAll((0..3).map { "primary-$it" }))
    }

    @Test
    fun `list evidence is retained for an implicit enumeration question`() {
        val candidates = buildList {
            repeat(4) { rank ->
                add(citation("primary-$rank", 190 + rank, "Emergency plan discussion $rank"))
                repeat(3) { neighbor ->
                    add(citation("neighbor-$rank-$neighbor", 190 + rank, "Adjacent detail $neighbor")
                        .copy(retrievalProvenance = "ADJACENT", sourceChunkId = "primary-$rank", score = 0.85))
                }
            }
            add(citation("emergency-items", 196, "A. Poisoning B. Fire C. Chemical spill D. Collapse E. Flood")
                .copy(contentKind = "LIST", retrievalProvenance = "ADJACENT", sourceChunkId = "primary-2", score = 0.8))
        }

        val selected = ContextSelector(maxFactPrimary = 4, maxFactExcerpts = 8).select(
            "What emergencies must a construction-site emergency plan consider?",
            candidates,
            QuestionIntent.FACT,
        )

        assertTrue(selected.excerpts.any { it.chunkId == "emergency-items" })
    }

    @Test
    fun `exact duration evidence outranks higher scored generic training prose`() {
        val generic = citation("generic", 44, "Safety and health training protects construction workers")
            .copy(score = 2.0)
        val exact = citation("exact", 45, "The training duration shall preferably be not less 48 hours")
            .copy(score = 0.8)

        val selected = ContextSelector(maxFactPrimary = 1, maxFactExcerpts = 1).select(
            "What is the recommended minimum duration for construction workers' safety and health training?",
            listOf(generic, exact),
            QuestionIntent.FACT,
        )

        assertEquals(listOf("exact"), selected.excerpts.map { it.chunkId })
    }

    @Test
    fun `constrained min max table question excludes a weaker sibling section`() {
        val correct = citation(
            "correct",
            62,
            "Concrete Type Minimum Slump Maximum Slump Portland Cement Concrete 2 inches 4 inches " +
                "Concrete dosed with superplasticizer 1 inch 3 inches",
        ).copy(sectionId = "mix-03300")
        val correctNeighbor = citation("correct-table-tail", 62, "Additional slump rows")
            .copy(sectionId = "mix-03300", retrievalProvenance = "STRUCTURAL", sourceChunkId = "correct")
        val distractor = citation(
            "distractor",
            17,
            "Minimum cement concrete criteria. Slump 3 to 4 inches. Maximum water cement ratio 0.50.",
        ).copy(sectionId = "mix-other")

        val selected = ContextSelector().select(
            "What are the minimum and maximum slump values for Portland cement concrete and concrete dosed with superplasticizer?",
            listOf(correct, correctNeighbor, distractor),
            QuestionIntent.FACT,
        )

        assertTrue(selected.excerpts.any { it.chunkId == "correct" })
        assertFalse(selected.excerpts.any { it.chunkId == "distractor" })
    }

    @Test
    fun `mixed numeric and color question stays in the strongest section`() {
        val target = citation(
            "curing-material",
            24,
            "Membrane-forming curing compound minimum solids content 30 percent; curing compound shall be white-pigmented.",
        ).copy(sectionId = "curing-materials")
        val distractor = citation(
            "curing-procedure",
            44,
            "After 2 days remove forms and apply curing compound to the concrete finish.",
        ).copy(sectionId = "curing-procedure")

        val selected = ContextSelector().select(
            "What minimum solids content and color requirement apply to membrane forming curing compound?",
            listOf(target, distractor),
            QuestionIntent.FACT,
        )

        assertEquals(listOf("curing-material"), selected.excerpts.map { it.chunkId })
    }

    private fun citation(id: String, page: Int, text: String) = Citation(
        id, "doc", page, page, 1.0, text,
        indexNamespace = "doc:v21:x", sectionId = "s1", sectionPath = "Section A",
    )
}
