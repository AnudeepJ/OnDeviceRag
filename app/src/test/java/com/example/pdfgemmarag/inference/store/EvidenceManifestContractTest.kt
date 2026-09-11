package com.example.pdfgemmarag.inference.store

import com.example.pdfgemmarag.inference.chat.AnswerQuestionUseCase
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.chunk.ScriptAwareChunker
import com.example.pdfgemmarag.inference.pdf.PageContent
import com.example.pdfgemmarag.inference.pdf.Segment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EvidenceManifestContractTest {
    @Test
    fun `explicit parents support specification section then part outlines`() {
        val chunks = ScriptAwareChunker(minKeepChars = 1).chunk("spec", listOf(PageContent(1, listOf(
            Segment.Heading(1, "03 30 00", "Cast-in-Place Concrete", 1, kind = Segment.Heading.KIND_SECTION),
            Segment.Heading(1, "1", "General", 2, kind = Segment.Heading.KIND_PART),
        ))))

        val section = chunks.first { it.sectionTitle == "Cast-in-Place Concrete" }
        val part = chunks.first { it.sectionTitle == "General" }
        assertEquals(section.sectionId, part.parentSectionId)
    }

    @Test
    fun `explicit parents support book part then chapter outlines`() {
        val chunks = ScriptAwareChunker(minKeepChars = 1).chunk("book", listOf(PageContent(1, listOf(
            Segment.Heading(1, "I", "Foundations", 2, kind = Segment.Heading.KIND_PART),
            Segment.Heading(1, "1", "Principles", 1, kind = Segment.Heading.KIND_CHAPTER),
            Segment.Heading(1, "2", "Practice", 1, kind = Segment.Heading.KIND_CHAPTER),
            Segment.Heading(1, "II", "Applications", 2, kind = Segment.Heading.KIND_PART),
        ))))

        val foundations = chunks.first { it.sectionTitle == "Foundations" }
        val principles = chunks.first { it.sectionTitle == "Principles" }
        val practice = chunks.first { it.sectionTitle == "Practice" }
        val applications = chunks.first { it.sectionTitle == "Applications" }
        assertEquals(foundations.sectionId, principles.parentSectionId)
        assertEquals(foundations.sectionId, practice.parentSectionId)
        assertEquals("", applications.parentSectionId)
    }

    @Test
    fun `v25 persists hierarchy lists cells and every source page`() {
        val pages = listOf(
            PageContent(1, listOf(
                Segment.Heading(1, "13", "Excavation", 1, kind = Segment.Heading.KIND_CHAPTER),
                Segment.Heading(1, "13.1", "Preparation", 2),
                Segment.ListBlock(1, listOf(
                    Segment.ListItem("1.", "Inspect the site"),
                    Segment.ListItem("2.", "Provide access"),
                )),
                Segment.Table(1, "| Type | Slope |", listOf("| B | 1:1 |"), "Table 1 Slopes", "1"),
            )),
            PageContent(2, emptyList()),
            PageContent(3, listOf(Segment.Paragraph(3, "Closing requirement with enough text to remain indexed in the final page record."))),
        )
        val chunks = ScriptAwareChunker(minKeepChars = 1).chunk("doc", pages)
        val manifest = DocumentStructureManifest.fromChunks(
            "doc", "doc:v24:test", "sig", chunks, tokenCount = String::length, pageCount = 3,
        )

        val chapter = manifest.sections.first { it.title == "Excavation" }
        val child = manifest.sections.first { it.title == "Preparation" }
        assertEquals(chapter.sectionId, child.parentSectionId)
        assertEquals(listOf(chapter.sectionId, child.sectionId), manifest.descendantsOf(chapter).map { it.sectionId })
        assertEquals(3, manifest.pageCount)
        assertEquals(listOf(1, 2, 3), manifest.pages.map { it.pageNumber })
        assertTrue(manifest.pages[1].orderedChunkIds.isEmpty())

        val list = manifest.lists.single()
        assertTrue(list.complete)
        assertEquals(2, list.itemCount)
        assertEquals(child.sectionId, list.sectionId)

        val table = manifest.tables.single()
        val slope = table.cells.single { it.columnHeaderPath == listOf("Slope") }
        assertEquals("B", slope.rowHeader)
        assertEquals("1:1", slope.text)
        assertEquals(1, slope.pageNumber)
        assertTrue(slope.chunkId in table.orderedRowChunkIds)
    }

    @Test
    fun `bounded subtree keeps both ends instead of truncating the tail`() {
        val chunks = (0 until 100).map { index -> DocumentStructureManifest.chunkId(index) }
        val root = SectionRecord("root", "", "1", "Root", "Root", 1, 100, chunks, 100)
        val manifest = DocumentStructureManifest(
            "doc", "doc:v24:test", DocumentStructureManifest.INDEX_VERSION, "sig", listOf(root),
            pageCount = 100,
        )

        val sampled = manifest.boundedSubtreeChunkIds(root, 8)

        assertEquals(8, sampled.size)
        assertEquals(chunks.first(), sampled.first())
        assertEquals(chunks.last(), sampled.last())
        assertFalse(sampled == chunks.take(8))
    }

    @Test
    fun `degraded overview samples physical pages`() {
        val section = SectionRecord("root", "", "", "", "", 1, 8, (0..7).map(DocumentStructureManifest::chunkId), 8)
        val pages = (1..8).map { page -> PageRecord(page, listOf(DocumentStructureManifest.chunkId(page - 1))) }
        val manifest = DocumentStructureManifest(
            "doc", "doc:v24:test", DocumentStructureManifest.INDEX_VERSION, "sig", listOf(section),
            pageCount = 8, pages = pages,
        )

        val ids = AnswerQuestionUseCase.pageStratifiedChunkIds(manifest, 4)

        assertEquals(4, ids.size)
        assertEquals(DocumentStructureManifest.chunkId(0), ids.first())
        assertEquals(DocumentStructureManifest.chunkId(7), ids.last())
    }

    @Test
    fun `canonical cells retain identical source rows`() {
        val pages = listOf(PageContent(1, listOf(
            Segment.Table(
                1,
                "| Item | Value |",
                listOf("| Repeated | 7 |", "| Repeated | 7 |"),
                "Table 2 Repeated readings",
                "2",
            ),
        )))
        val chunks = ScriptAwareChunker(minKeepChars = 1).chunk("doc", pages)
        val table = DocumentStructureManifest.fromChunks(
            "doc", "doc:v24:test", "sig", chunks, String::length, pageCount = 1,
        ).tables.single()

        assertEquals(2, table.cells.count { it.columnHeaderPath == listOf("Value") && it.text == "7" })
        assertEquals(listOf(0, 1), table.cells.filter { it.columnIndex == 1 }.map { it.rowIndex })
    }

    @Test
    fun `explicit parents isolate repeated root paths`() {
        val first = SectionRecord("first", "", "1", "General", "1 General", 1, 1, listOf("c0000000"), 1)
        val second = first.copy(sectionId = "second", startPage = 2, endPage = 2, orderedChunkIds = listOf("c0000001"))
        val child = SectionRecord(
            "child", "", "1.1", "Limits", "1 General > 1.1 Limits", 2, 2,
            listOf("c0000002"), 1, parentSectionId = "second",
        )
        val manifest = DocumentStructureManifest(
            "doc", "doc:v25:test", DocumentStructureManifest.INDEX_VERSION, "sig",
            listOf(first, second, child), pageCount = 2,
        )

        assertEquals(listOf("first"), manifest.descendantsOf(first).map { it.sectionId })
        assertEquals(listOf("second", "child"), manifest.descendantsOf(second).map { it.sectionId })
    }

    @Test
    fun `canonical cells survive table token splitting`() {
        val chunker = ScriptAwareChunker(minKeepChars = 1)
        val chunks = chunker.chunk("doc", listOf(PageContent(1, listOf(
            Segment.Table(
                1, "| Item | Description | Limit |",
                listOf("| A | ${"long description ".repeat(40)} | 99 |"), "Table 1", "1",
            ),
        ))))
        val fitted = chunker.reindex(chunks.flatMap { chunker.fitToTokenWindow(it, 180, String::length) })
        val table = DocumentStructureManifest.fromChunks(
            "doc", "doc:v25:test", "sig", fitted, String::length, pageCount = 1,
        ).tables.single()

        assertTrue(table.cells.any {
            it.rowHeader == "A" && it.columnHeaderPath == listOf("Limit") && it.text == "99"
        })
        assertEquals(fitted.map { DocumentStructureManifest.chunkId(it.chunkIndex) }, table.cells.first().sourceChunkIds)
    }

    @Test
    fun `canonical cells answer a matrix lookup after table splitting`() {
        val chunker = ScriptAwareChunker(minKeepChars = 1)
        val chunks = chunker.chunk("doc", listOf(PageContent(1, listOf(
            Segment.Table(
                1, "| Likelihood | Moderate | Catastrophic |",
                listOf("| Almost Certain | ${"supporting detail ".repeat(40)}9 | 25 |"),
                "Table 1.1 Risk Level Assessment", "1.1",
            ),
        ))))
        val fitted = chunker.reindex(chunks.flatMap { chunker.fitToTokenWindow(it, 180, String::length) })
        val manifest = DocumentStructureManifest.fromChunks(
            "doc", "doc:v25:test", "sig", fitted, String::length, pageCount = 1,
        )
        val table = manifest.tables.single()
        val source = fitted.map { chunk ->
            Citation(
                DocumentStructureManifest.chunkId(chunk.chunkIndex), "doc", 1, chunk.chunkIndex,
                1.0, chunk.text, tableId = table.tableId, tableNumber = table.tableNumber,
                tableCaption = table.caption, contentKind = "TABLE",
            )
        }
        val canonical = AnswerQuestionUseCase.canonicalTableEvidence(manifest, table.tableId, source)

        assertEquals(canonical.size, canonical.map { it.chunkId }.distinct().size)
        assertTrue(canonical.any { it.retrievalProvenance == "CANONICAL_CELL" && it.sourceChunkId.isNotBlank() })

        val lead = AnswerQuestionUseCase.buildTableLead(
            "In Table 1.1, what risk score results from Almost Certain likelihood and Catastrophic consequence?",
            canonical, table.tableId,
        )

        assertTrue(lead.decisive)
        assertTrue(lead.text.contains("25"))
    }
}
