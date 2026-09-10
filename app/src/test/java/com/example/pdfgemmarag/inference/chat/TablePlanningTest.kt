package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.inference.chunk.ScriptAwareChunker
import com.example.pdfgemmarag.inference.pdf.PageContent
import com.example.pdfgemmarag.inference.pdf.Segment
import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.SectionRecord
import com.example.pdfgemmarag.inference.store.TableRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Roadmap M4: table identity in the manifest and planner, never confused with section ids. */
class TablePlanningTest {

    private val likelihood = TableRecord(
        tableId = "t-like",
        tableNumber = "5.2",
        caption = "Table 5.2 Likelihood of occurrence",
        startPage = 28,
        endPage = 28,
        orderedRowChunkIds = listOf("c0000001"),
        aliases = listOf("likelihood", "occurrence", "5.2"),
        columnHeaders = listOf("Likelihood", "Rating"),
    )
    private val risk = TableRecord(
        tableId = "t-risk",
        tableNumber = "5.1",
        caption = "Table 5.1 Relative risk and control level",
        startPage = 28,
        endPage = 29,
        orderedRowChunkIds = listOf("c0000002"),
        aliases = listOf("relative", "risk", "control", "level", "5.1"),
        columnHeaders = listOf("Risk", "Description", "Control"),
    )
    private val section = SectionRecord(
        "s1", "", "1.1", "SCOPE", "SCOPE", 1, 2, listOf("c0000000"), 40,
        level = 1, kind = "CLAUSE", printedNumber = "1.1",
    )
    private val manifest = DocumentStructureManifest(
        "doc", "doc:v23:test", DocumentStructureManifest.INDEX_VERSION, "sig",
        listOf(section),
        listOf(likelihood, risk),
    )

    @Test
    fun `explicit table number resolves without treating it as a section`() {
        val plan = QueryPlanner().plan("In Table 5.1, what control level applies to 13-20?", manifest)
        assertEquals(QuestionIntent.FACT, plan.intent)
        assertEquals("t-risk", plan.resolvedTableId)
        assertNull(plan.explicitSectionNumber)
        assertNull(plan.resolvedSectionId)
        assertTrue("5.1" in plan.subjectText)
    }

    @Test
    fun `table 1-1 is not section 1-1`() {
        val plan = QueryPlanner().plan("What value is listed in Table 1.1?", manifest)
        assertNull(plan.explicitSectionNumber)
        assertNull(plan.resolvedSectionId)
        assertTrue("1.1" in plan.subjectText)
    }

    @Test
    fun `titled table request resolves against the caption`() {
        val plan = QueryPlanner().plan("In the likelihood table, what rating is given to a Possible occurrence?", manifest)
        assertEquals("t-like", plan.resolvedTableId)
        assertEquals(AnswerShape.TABLE, plan.shape)
    }

    @Test
    fun `relative risk title prefers the risk grid over likelihood`() {
        val plan = QueryPlanner().plan("In the relative risk table, what control level applies to 13-20?", manifest)
        assertEquals("t-risk", plan.resolvedTableId)
    }

    @Test
    fun `chunker writes table identity that the manifest keeps`() {
        val pages = listOf(
            PageContent(
                28,
                listOf(
                    Segment.Heading(28, "5.2", "RISK", 1, kind = "CLAUSE", printedNumber = "5.2"),
                    Segment.Table(
                        28,
                        "| Risk | Control |",
                        listOf("| 21-25 | Very high |"),
                        caption = "Table 5.1 Relative risk and control level",
                        tableNumber = "5.1",
                    ),
                ),
            ),
        )
        val chunks = ScriptAwareChunker().chunk("doc-hash", pages)
        val table = chunks.single { it.isTable }
        assertEquals("5.1", table.tableNumber)
        assertEquals("Table 5.1 Relative risk and control level", table.tableCaption)
        assertTrue(table.tableId.isNotBlank())
        assertTrue(table.retrievalText.contains("Risk: 21-25"))
        val built = DocumentStructureManifest.fromChunks(
            "doc-hash", "doc:v23:x", "sig", chunks, tokenCount = { it.length / 4 },
        )
        assertEquals(1, built.tables.size)
        assertEquals("5.1", built.tables.single().tableNumber)
        assertEquals(table.tableId, built.tables.single().tableId)
        assertEquals(listOf(DocumentStructureManifest.chunkId(table.chunkIndex)), built.tables.single().orderedRowChunkIds)
        assertTrue(built.tables.single().aliases.contains("relative"))
    }
}
