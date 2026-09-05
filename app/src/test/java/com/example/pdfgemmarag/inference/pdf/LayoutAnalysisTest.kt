package com.example.pdfgemmarag.inference.pdf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LayoutAnalysisTest {

    /** Builds a line of words laid out left to right at the given x starts (6pt per char). */
    private fun line(y: Float, vararg cells: Pair<Float, String>): LineBox = LineBox(
        cells.flatMap { (x, text) ->
            var cursor = x
            text.split(' ').map { w ->
                val box = Box(cursor, y, cursor + w.length * 6f, y + 10f)
                cursor += w.length * 6f + 6f // one space
                WordBox(w, box)
            }
        },
    )

    private fun page(n: Int, lines: List<LineBox>) = PageLayout(n, 600f, 800f, lines, hasImages = false, source = PageLayout.Source.TEXT)

    @Test
    fun `table rows with aligned columns become a markdown table`() {
        val lines = listOf(
            line(50f, 40f to "Introduction to the annual results"),
            line(80f, 40f to "Year", 200f to "Revenue", 400f to "Margin"),
            line(95f, 40f to "2022", 200f to "4.2M", 400f to "31%"),
            line(110f, 40f to "2023", 200f to "5.1M", 400f to "34%"),
            line(140f, 40f to "Revenue grew because of strong demand in Asia."),
        )
        val content = TableClusterer().analyse(page(1, lines))
        val tables = content.segments.filterIsInstance<Segment.Table>()
        assertEquals(1, tables.size)
        assertEquals("| Year | Revenue | Margin |", tables[0].header)
        assertEquals(listOf("| 2022 | 4.2M | 31% |", "| 2023 | 5.1M | 34% |"), tables[0].rows)
        val paragraphs = content.segments.filterIsInstance<Segment.Paragraph>()
        assertEquals(2, paragraphs.size)
        assertTrue(paragraphs[0].text.startsWith("Introduction"))
        assertTrue(paragraphs[1].text.startsWith("Revenue grew"))
    }

    @Test
    fun `plain prose lines merge into one paragraph and large gaps split`() {
        val lines = listOf(
            line(50f, 40f to "First line of the paragraph continues"),
            line(62f, 40f to "onto the second line without a gap."),
            line(120f, 40f to "A new paragraph after a big vertical gap."),
        )
        val content = TableClusterer().analyse(page(1, lines))
        val paragraphs = content.segments.filterIsInstance<Segment.Paragraph>()
        assertEquals(2, paragraphs.size)
        assertEquals("First line of the paragraph continues onto the second line without a gap.", paragraphs[0].text)
    }

    @Test
    fun `cjk lines join without spaces`() {
        val lines = listOf(
            LineBox(listOf(WordBox("東京都は日本の", Box(40f, 50f, 120f, 60f)))),
            LineBox(listOf(WordBox("首都です。", Box(40f, 62f, 100f, 72f)))),
        )
        val content = TableClusterer().analyse(page(1, lines))
        assertEquals("東京都は日本の首都です。", (content.segments.single() as Segment.Paragraph).text)
    }

    @Test
    fun `repeated headers footers and page numbers are stripped`() {
        val pages = (1..10).map { p ->
            page(
                p,
                listOf(
                    line(20f, 40f to "ACME Corp Annual Report 2023"),
                    line(50f, 40f to "Body ${"alpha beta gamma delta epsilon zeta eta theta iota kappa".split(' ')[p - 1]} paragraph."),
                    line(60f, 40f to "More ${"one two three four five six seven eight nine ten".split(' ')[p - 1]} text."),
                    line(780f, 300f to "Page $p of 10"),
                ),
            )
        }
        val stripped = HeaderFooterStripper().strip(pages)
        stripped.forEach { pg ->
            assertEquals(2, pg.lines.size)
            assertFalse(pg.lines.any { it.text.contains("ACME") || it.text.startsWith("Page") })
        }
    }

    @Test
    fun `unique first lines are kept`() {
        val pages = (1..5).map { p ->
            page(p, listOf(line(20f, 40f to "Chapter $p unique title ${"x".repeat(p)}"), line(50f, 40f to "Body ${"x".repeat(p)}")))
        }
        val stripped = HeaderFooterStripper().strip(pages)
        // Digits normalise to '#', but the trailing x-runs differ, so nothing repeats.
        assertTrue(stripped.all { it.lines.size == 2 })
    }
}
