package com.example.pdfgemmarag.inference.pdf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RowReflowTest {

    private val pageWidth = 595f
    private fun w(text: String, x: Float, y: Float, h: Float = 11f) = WordBox(text, Box(x, y, x + text.length * 5.5f, y + h))
    private fun row(y: Float, vararg cells: Pair<Float, String>) = LineBox(cells.map { (x, t) -> w(t, x, y) })
    private fun cell(x: Float, y: Float, t: String) = LineBox(listOf(w(t, x, y)))

    /** Mimics generators for which TextExtractor emits a table column as one vertical "line". */
    private fun column(x: Float, vararg cells: Pair<Float, String>) = LineBox(cells.map { (y, t) -> w(t, x, y) })

    private val expected = listOf(
        "Financial summary for the fiscal years shown below.",
        "Year Revenue Profit Margin",
        "2021 3.1M 0.4M 12.9%",
        "2022 3.7M 0.6M 16.2%",
        "2023 4.2M 0.9M 21.4%",
        "Figures are unaudited.",
    )

    @Test
    fun `vertical stacks are detected and horizontal lines are not`() {
        val r = RowReflow()
        assertTrue(r.isVerticalStack(column(60f, 140f to "2021", 156f to "2022", 172f to "2023")))
        assertFalse(r.isVerticalStack(row(100f, 60f to "Year", 180f to "Revenue", 300f to "Profit")))
        val superscript = LineBox(listOf(w("Water", 10f, 100f), w("2", 40f, 94f, 5f), w("is", 50f, 100f), w("wet", 65f, 100f)))
        assertFalse(r.isVerticalStack(superscript))
    }

    @Test
    fun `one-word-per-cell output (PDFNet 12 on emulator) is rebuilt into rows`() {
        // Exact geometry captured by Spike5ApryseExtractionTest on the emulator.
        val lines = listOf(
            row(71f, 60f to "Financial", 108f to "summary", 158f to "for", 175f to "the", 194f to "fiscal", 224f to "years", 254f to "shown", 289f to "below."),
            cell(60f, 111f, "Year"), cell(180f, 111f, "Revenue"), cell(300f, 111f, "Profit"),
            cell(60f, 131f, "2021"), cell(60f, 147f, "2022"), cell(60f, 163f, "2023"),
            row(211f, 60f to "Figures", 99f to "are", 118f to "unaudited."),
            cell(180f, 131f, "3.1M"), cell(180f, 147f, "3.7M"), cell(180f, 163f, "4.2M"),
            cell(300f, 131f, "0.4M"), cell(300f, 147f, "0.6M"), cell(300f, 163f, "0.9M"),
            cell(420f, 111f, "Margin"),
            cell(420f, 131f, "12.9%"), cell(420f, 147f, "16.2%"), cell(420f, 163f, "21.4%"),
        )
        val out = RowReflow().reflow(lines, pageWidth)
        assertEquals(expected, out.map { it.text })
        val table = TableClusterer().analyse(PageLayout(1, pageWidth, 842f, out, false, PageLayout.Source.TEXT))
            .segments.filterIsInstance<Segment.Table>().single()
        assertTrue(table.header, table.header.contains("Margin"))
        assertEquals(3, table.rows.size)
    }

    @Test
    fun `column-wise vertical lines are rebuilt into rows`() {
        val lines = listOf(
            row(80f, 60f to "Financial summary for the fiscal years shown below."),
            row(120f, 60f to "Year", 180f to "Revenue", 300f to "Profit"),
            row(120f, 420f to "Margin"),
            column(60f, 140f to "2021", 156f to "2022", 172f to "2023"),
            row(220f, 60f to "Figures are unaudited."),
            column(180f, 140f to "3.1M", 156f to "3.7M", 172f to "4.2M"),
            column(300f, 140f to "0.4M", 156f to "0.6M", 172f to "0.9M"),
            column(420f, 140f to "12.9%", 156f to "16.2%", 172f to "21.4%"),
        )
        assertEquals(expected, RowReflow().reflow(lines, pageWidth).map { it.text })
    }

    @Test
    fun `two-column body text is left alone`() {
        val lines = (0 until 12).map { row(100f + it * 14f, 40f to "Left column sentence number $it about apples.") } +
            (0 until 12).map { row(100f + it * 14f, 310f to "Right column sentence number $it about oranges.") }
        assertEquals(lines, RowReflow().reflow(lines, pageWidth))
    }

    @Test
    fun `single-column flow is restored from geometry`() {
        val lines = listOf(
            row(100f, 60f to "First body line is intentionally long enough."),
            row(800f, 60f to "Footer text"),
            row(114f, 60f to "Second wrapped body line follows the first."),
        )

        assertEquals(
            listOf(lines[0].text, lines[2].text, lines[1].text),
            RowReflow().reflow(lines, pageWidth).map { it.text },
        )
    }

    @Test
    fun `wide table labels paired with narrow values are rebuilt into rows`() {
        val lines = listOf(
            cell(60f, 110f, "Surfaces of columns piers and walls"),
            cell(470f, 110f, "1/2 in 10"),
            cell(60f, 130f, "Top surfaces of curbs and railings"),
            cell(470f, 130f, "3/16 in 10"),
            cell(60f, 150f, "Cross section of columns caps walls and beams"),
            cell(470f, 150f, "+1/2 -1/4"),
        )

        val out = RowReflow().reflow(lines, pageWidth)

        assertEquals(3, out.size)
        assertEquals("Surfaces of columns piers and walls 1/2 in 10", out[0].text)
        assertEquals("Top surfaces of curbs and railings 3/16 in 10", out[1].text)
        assertEquals("Cross section of columns caps walls and beams +1/2 -1/4", out[2].text)
    }

    @Test
    fun `centred values bridge wrapped description cells into logical rows`() {
        val lines = listOf(
            cell(135f, 220f, "PLUMB OR"),
            cell(235f, 220f, "Surfaces of columns, piers and"),
            cell(425f, 227f, "1/2 in 10"),
            cell(135f, 234f, "SPECIFIED BATTER"),
            cell(235f, 234f, "walls"),
            cell(235f, 266f, "Top surfaces of slabs"),
            cell(410f, 266f, "See Section 03345"),
            cell(135f, 286f, "SPECIFIED GRADE"),
            cell(235f, 284f, "Top surfaces of curbs and"),
            cell(422f, 291f, "3/16 in 10"),
            cell(235f, 298f, "railings"),
        )

        val out = RowReflow().reflow(lines, pageWidth).map { it.text }

        assertTrue(out.toString(), out.any {
            "Surfaces of columns, piers and" in it && "walls" in it && "1/2 in 10" in it
        })
        assertTrue(out.any { "Top surfaces of slabs See Section 03345" in it })
        assertTrue(out.toString(), out.any {
            "SPECIFIED GRADE" in it && "Top surfaces of curbs and" in it && "3/16 in 10" in it && "railings" in it
        })
    }

    @Test
    fun `two short same-baseline lines without a second aligned row are left alone`() {
        val lines = listOf(row(100f, 40f to "Title"), row(100f, 400f to "Draft"), row(130f, 40f to "A normal paragraph line follows here."))
        assertEquals(lines, RowReflow().reflow(lines, pageWidth))
    }
}
