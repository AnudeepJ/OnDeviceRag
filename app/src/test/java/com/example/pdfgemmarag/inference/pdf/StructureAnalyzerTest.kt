package com.example.pdfgemmarag.inference.pdf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StructureAnalyzerTest {
    private fun line(y: Float, text: String, x: Float = 40f, bold: Boolean = false) = LineBox(
        listOf(WordBox(text, Box(x, y, x + text.length * 6f, y + 10f), bold = bold)),
    )

    @Test
    fun `aggregates a wrapped heading before updating hierarchy`() {
        val page = PageLayout(
            1, 600f, 800f,
            listOf(
                line(40f, "2.05 CONCRETE MIX AND", bold = true),
                line(52f, "PERFORMANCE SPECIFICATIONS", x = 44f, bold = true),
                line(85f, "The mixture shall comply with the following requirements."),
            ),
            false, PageLayout.Source.TEXT,
        )
        val segments = StructureAnalyzer().analyse(page).segments
        val heading = segments.first() as Segment.Heading
        assertEquals("2.05", heading.number)
        assertEquals("CONCRETE MIX AND PERFORMANCE SPECIFICATIONS", heading.title)
    }

    @Test
    fun `enumerated clauses remain logical list items`() {
        val page = PageLayout(
            1, 600f, 800f,
            listOf(line(40f, "A. First requirement"), line(52f, "B. Second requirement")),
            false, PageLayout.Source.TEXT,
        )
        val list = StructureAnalyzer().analyse(page).segments.single() as Segment.ListBlock
        assertEquals(listOf("A.", "B."), list.items.map { it.label })
        assertTrue(list.items[0].text.contains("First"))
    }

    @Test
    fun `section cross reference does not become a specification boundary`() {
        val page = PageLayout(
            1, 600f, 800f,
            listOf(
                line(40f, "1.02 RELATED SECTIONS", bold = true),
                line(65f, "SECTION 01450 Testing Laboratory Services"),
                line(90f, "Requirements continue here."),
            ),
            false, PageLayout.Source.TEXT,
        )

        val headings = StructureAnalyzer().analyse(page).segments.filterIsInstance<Segment.Heading>()
        assertEquals(listOf("1.02"), headings.map { it.number })
    }

    @Test
    fun `styled title case section heading becomes a specification boundary`() {
        val page = PageLayout(
            1, 600f, 800f,
            listOf(
                line(40f, "SECTION 03310", bold = true),
                line(52f, "Cast-in-Place Concrete", bold = true),
                line(90f, "1.0 GENERAL", bold = true),
            ),
            false, PageLayout.Source.TEXT,
        )

        val heading = StructureAnalyzer().analyse(page).segments.first() as Segment.Heading
        assertEquals("03310", heading.specificationNumber)
        assertEquals("Cast-in-Place Concrete", heading.title)
    }

    @Test
    fun `unstyled section heading with part one sequence becomes a specification boundary`() {
        val page = PageLayout(
            1, 600f, 800f,
            listOf(
                line(220f, "SECTION 03310"),
                line(232f, "Structural Concrete"),
                line(270f, "1.0 GENERAL"),
                line(296f, "1.01 SECTION INCLUDES"),
            ),
            false, PageLayout.Source.TEXT,
        )

        val heading = StructureAnalyzer().analyse(page).segments.first() as Segment.Heading
        assertEquals("03310", heading.specificationNumber)
        assertEquals("Structural Concrete", heading.title)
    }

    @Test
    fun `unstyled section heading near page start becomes a specification boundary`() {
        val page = PageLayout(
            1, 600f, 800f,
            listOf(
                line(80f, "SECTION 03310"),
                line(92f, "Structural Concrete"),
                line(170f, "Introductory text continues."),
            ),
            false, PageLayout.Source.TEXT,
        )

        val heading = StructureAnalyzer().analyse(page).segments.first() as Segment.Heading
        assertEquals("03310", heading.specificationNumber)
    }

    @Test
    fun `uppercase standards ending with punctuation are not headings`() {
        val page = PageLayout(
            1, 600f, 800f,
            listOf(
                line(220f, "ASTM C33."),
                line(246f, "ACI - 305R."),
                line(272f, "Requirements continue here."),
            ),
            false, PageLayout.Source.TEXT,
        )

        val headings = StructureAnalyzer().analyse(page).segments.filterIsInstance<Segment.Heading>()
        assertTrue(headings.isEmpty())
    }
}
