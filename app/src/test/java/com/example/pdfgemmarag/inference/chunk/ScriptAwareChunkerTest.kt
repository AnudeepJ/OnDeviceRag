package com.example.pdfgemmarag.inference.chunk

import com.example.pdfgemmarag.inference.ocr.Script
import com.example.pdfgemmarag.inference.pdf.PageContent
import com.example.pdfgemmarag.inference.pdf.Segment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ScriptAwareChunkerTest {
    @Test
    fun `oversized table is split under actual token window and repeats header`() {
        val chunker = ScriptAwareChunker()
        val chunk = Chunk(
            chunkIndex = 0,
            pageNumber = 7,
            text = "Column A | Column B\n| --- | --- |\n" + (1..30).joinToString("\n") { "row $it | " + "value ".repeat(12) },
            isTable = true,
            script = Script.LATIN,
        )
        val pieces = chunker.fitToTokenWindow(chunk, maxTokens = 80) { text -> text.length / 4 + 2 }
        assertTrue(pieces.size > 1)
        assertTrue(pieces.all { it.text.startsWith("Column A | Column B") })
        assertTrue(pieces.all { it.text.length / 4 + 2 <= 80 })
        assertTrue(pieces.all { it.pageNumber == 7 })
    }

    @Test
    fun `token splitting preserves list item offsets and only final completion`() {
        val chunker = ScriptAwareChunker()
        val chunk = Chunk(
            chunkIndex = 0,
            pageNumber = 4,
            text = (1..8).joinToString("\n") { "$it. " + "requirement ".repeat(8) },
            isTable = false,
            script = Script.LATIN,
            contentKind = "LIST",
            listId = "list-1",
            listItemStart = 10,
            listItemCount = 8,
            listComplete = true,
        )

        val pieces = chunker.fitToTokenWindow(chunk, maxTokens = 40) { it.length / 4 + 2 }

        assertTrue(pieces.size > 1)
        assertEquals((10 until 18).toList(), pieces.flatMap { piece ->
            (piece.listItemStart until piece.listItemStart + piece.listItemCount).toList()
        })
        assertTrue(pieces.dropLast(1).none { it.listComplete })
        assertTrue(pieces.last().listComplete)
    }

    @Test
    fun `adjacent page list blocks share logical identity and global item offsets`() {
        val chunks = ScriptAwareChunker(latinTarget = 80, minKeepChars = 1).chunk("doc", listOf(
            PageContent(1, listOf(Segment.ListBlock(1, listOf(
                Segment.ListItem("1.", "First check"), Segment.ListItem("2.", "Second check"),
            )))),
            PageContent(2, listOf(Segment.ListBlock(2, listOf(
                Segment.ListItem("3.", "Third check"), Segment.ListItem("4.", "Fourth check"),
            )))),
        ))

        assertEquals(1, chunks.map { it.listId }.distinct().size)
        assertEquals(listOf(0, 2), chunks.map { it.listItemStart })
        assertFalse(chunks.first().listComplete)
        assertTrue(chunks.last().listComplete)
    }

    @Test
    fun `inline labelled items contribute their true logical count`() {
        val chunks = ScriptAwareChunker(minKeepChars = 1).chunk("doc", listOf(PageContent(1, listOf(
            Segment.ListBlock(1, listOf(Segment.ListItem(
                "A.", "Fire B. Explosion C. Chemical spill D. Electrocution E. Collapse F. Flood",
            ))),
        ))))

        assertEquals(6, chunks.single().listItemCount)
        assertTrue(chunks.single().listComplete)
    }

    @Test
    fun `page continuation identity survives later same-page token fragments`() {
        val chunks = ScriptAwareChunker().reindex(listOf(
            Chunk(0, 1, "1. First", false, Script.LATIN, contentKind = "LIST", listId = "page-1", listItemCount = 1),
            Chunk(1, 2, "2. Second", false, Script.LATIN, contentKind = "LIST", listId = "page-2", listItemCount = 1),
            Chunk(2, 2, "3. Third", false, Script.LATIN, contentKind = "LIST", listId = "page-2", listItemCount = 1),
        ))

        assertEquals(listOf("page-1"), chunks.map { it.listId }.distinct())
        assertEquals(listOf(0, 1, 2), chunks.map { it.listItemStart })
        assertTrue(chunks.last().listComplete)
    }

    @Test
    fun `bullet led list does not count trailing subsection label as an item`() {
        val chunks = ScriptAwareChunker(minKeepChars = 1).chunk("doc", listOf(PageContent(1, listOf(
            Segment.ListBlock(1, listOf(
                Segment.ListItem("•", "Eliminate the hazard"),
                Segment.ListItem("•", "Use engineering controls"),
                Segment.ListItem("iii)", "Controlling Risk"),
            )),
        ))))

        assertEquals(2, chunks.single().listItemCount)
    }
    private val chunker = ScriptAwareChunker()

    private fun latinParagraph(sentences: Int) =
        (1..sentences).joinToString(" ") { "Sentence number $it describes a moderately long clause about revenue recognition policies." }

    private fun cjkParagraph(sentences: Int) =
        (1..sentences).joinToString("") { "第${it}条は収益認識に関する方針を説明しており、契約の識別と履行義務の配分について述べています。" }

    @Test
    fun `european text stays within max and overlaps`() {
        val text = latinParagraph(60)
        val chunks = chunker.chunkParagraph(text)
        assertTrue(chunks.size > 1)
        chunks.forEach { assertTrue("chunk too long: ${it.length}", it.length <= 1200 + 50) }
        // Overlap: the start of chunk i+1 should appear in chunk i.
        for (i in 0 until chunks.size - 1) {
            val head = chunks[i + 1].take(40)
            assertTrue("no overlap between chunk $i and ${i + 1}", chunks[i].contains(head))
        }
    }

    @Test
    fun `cjk text uses the smaller window`() {
        val text = cjkParagraph(40)
        val chunks = chunker.chunkParagraph(text)
        assertTrue(chunks.size > 3)
        chunks.forEach { assertTrue("cjk chunk too long: ${it.length}", it.length <= 400 + 60) }
        // Boundaries land on sentence terminators.
        chunks.dropLast(1).forEach { assertTrue(it.endsWith("。")) }
    }

    @Test
    fun `short paragraph is a single chunk`() {
        assertEquals(listOf("Hello world."), chunker.chunkParagraph("Hello   world."))
    }

    @Test
    fun `table chunks repeat the header`() {
        val header = "| Year | Revenue | Margin |"
        val rows = (1..80).map { "| ${2000 + it} | ${it * 1000} | ${it % 30}% |" }
        val chunks = chunker.chunkTable(Segment.Table(3, header, rows))
        assertTrue(chunks.size > 1)
        chunks.forEach { c ->
            assertTrue(c.startsWith(header))
            assertTrue(c.lines()[1].startsWith("| ---"))
        }
        val emitted = chunks.flatMap { it.lines().drop(2) }
        assertEquals(rows, emitted)
    }

    @Test
    fun `tiny same-page fragments merge and labels are dropped`() {
        val pages = listOf(
            PageContent(
                1,
                listOf(
                    Segment.Paragraph(1, "FORMWORK"),
                    Segment.Paragraph(1, "A. No separate payment will be made for concrete formwork."),
                    Segment.Paragraph(1, "CITY OF BAYTOWN"),
                    Segment.Paragraph(1, "03/2020"),
                ),
            ),
            PageContent(2, listOf(Segment.Paragraph(2, "03600-6"))),
        )
        val chunks = chunker.chunk(pages)
        assertEquals(1, chunks.size)
        assertTrue(chunks[0].text.contains("No separate payment"))
        assertTrue(chunks[0].text.contains("BAYTOWN"))
        assertTrue(chunks[0].text.contains("03/2020"))
        assertFalse(chunks.any { it.text == "03600-6" })
    }

    @Test
    fun `lowercase short line after heading is retained as its continuation`() {
        val chunks = ScriptAwareChunker().chunk("doc", listOf(PageContent(14, listOf(
            Segment.Heading(14, "2.5.5", "Personal Protective Equipment The provision should be considered when other controls are", 3),
            Segment.Paragraph(14, "impractical."),
            Segment.ListBlock(14, listOf(Segment.ListItem("•", "Eyes: safety glasses"))),
        ))))

        assertTrue(chunks.first().text.endsWith("controls are impractical."))
        assertTrue(chunks.any { "impractical" in it.text })
    }

    @Test
    fun `chunks carry page numbers and never cross pages`() {
        val pages = listOf(
            PageContent(1, listOf(Segment.Paragraph(1, latinParagraph(30)))),
            PageContent(2, listOf(Segment.Paragraph(2, cjkParagraph(20)), Segment.Table(2, "| A | B |", listOf("| 1 | 2 |")))),
        )
        val chunks = chunker.chunk(pages)
        assertTrue(chunks.any { it.pageNumber == 1 } && chunks.any { it.pageNumber == 2 })
        assertEquals(chunks.indices.toList(), chunks.map { it.chunkIndex })
        assertTrue(chunks.filter { it.pageNumber == 1 }.all { it.script == Script.LATIN })
        assertTrue(chunks.filter { it.pageNumber == 2 && !it.isTable }.all { it.script == Script.JAPANESE })
        assertEquals(1, chunks.count { it.isTable })
    }
}
