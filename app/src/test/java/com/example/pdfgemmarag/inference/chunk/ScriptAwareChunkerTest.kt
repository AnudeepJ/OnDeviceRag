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
