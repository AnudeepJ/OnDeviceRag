package com.example.pdfgemmarag.inference.pdf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Heading kinds beyond CSI specifications: chapters, appendices, Roman labels, adjacent titles, generic list labels. */
class OutlineKindsTest {
    private fun line(y: Float, text: String, x: Float = 40f, bold: Boolean = false) = LineBox(
        listOf(WordBox(text, Box(x, y, x + text.length * 6f, y + 10f), bold = bold)),
    )

    private fun page(vararg lines: LineBox, number: Int = 1) =
        PageLayout(number, 600f, 800f, lines.toList(), false, PageLayout.Source.TEXT)

    @Test
    fun `bare chapter label adopts the title on the next line even across a blank line`() {
        val segments = StructureAnalyzer().analyse(
            page(
                line(40f, "Chapter 13"),
                line(70f, "EXCAVATION"),
                line(95f, "13.1 INTRODUCTION"),
                line(110f, "Generally speaking, an excavation is a hole in the ground."),
            ),
        ).segments
        val chapter = segments[0] as Segment.Heading
        assertEquals(Segment.Heading.KIND_CHAPTER, chapter.kind)
        assertEquals("13", chapter.printedNumber)
        assertEquals("EXCAVATION", chapter.title)
        assertEquals(1, chapter.level)
        val clause = segments[1] as Segment.Heading
        assertEquals(Segment.Heading.KIND_CLAUSE, clause.kind)
        assertEquals("13.1", clause.printedNumber)
    }

    @Test
    fun `roman chapter and appendix labels normalise their printed numbers`() {
        val segments = StructureAnalyzer().analyse(
            page(
                line(40f, "CHAPTER XIII - EXCAVATION", bold = true),
                line(60f, "Body text follows here."),
                line(90f, "Appendix B"),
                line(104f, "LIST OF STANDARDS"),
            ),
        ).segments.filterIsInstance<Segment.Heading>()
        assertEquals(listOf("13", "B"), segments.map { it.printedNumber })
        assertEquals(listOf(Segment.Heading.KIND_CHAPTER, Segment.Heading.KIND_APPENDIX), segments.map { it.kind })
        assertEquals("LIST OF STANDARDS", segments[1].title)
    }

    @Test
    fun `prose beginning with a chapter reference is not a boundary`() {
        val segments = StructureAnalyzer().analyse(
            page(
                line(40f, "Chapter 13 describes the excavation requirements that apply to every trench deeper than the limit set here."),
                line(52f, "Chapter 4"),
                line(64f, "the following paragraph continues in lower case."),
            ),
        ).segments
        assertTrue(segments.none { it is Segment.Heading && it.kind == Segment.Heading.KIND_CHAPTER })
    }

    @Test
    fun `contents page with chapter rows produces no chapter headings`() {
        val headings = StructureAnalyzer().analyse(
            page(
                line(40f, "Table of Contents"),
                line(60f, "Chapter Subject Page No."),
                line(72f, "1 Introduction 03"),
                line(84f, "2 Safety and Health Policy 07"),
                line(96f, "12 Confined Space 97"),
                line(108f, "13 Excavation 102"),
                line(120f, "14 Explosives 108"),
                line(132f, "28 List of Indian Standards 206"),
                number = 2,
            ),
        ).segments.filterIsInstance<Segment.Heading>()
        assertTrue(headings.isEmpty())
    }

    @Test
    fun `roman numeral and glyph bullets are list items`() {
        val segments = StructureAnalyzer().analyse(
            page(
                line(40f, "There are some basic steps about how the assessment should be undertaken like:"),
                line(52f, "I. Initiating the review"),
                line(64f, "II. Identify the hazard"),
                line(76f, "III. Identify all parties affected by the hazard and determine how they"),
                line(88f, "can be affected", x = 52f),
                line(100f, "IV. Evaluate or assess the risk"),
                line(112f, "V. Monitor & Review"),
                line(140f, "\uF0B7 All simple slope excavations 20 feet (6Mt) or less in depth"),
                line(152f, "\uF0B7 All benched excavations have a maximum allowable slope of 1:1."),
            ),
        ).segments
        val list = segments[1] as Segment.ListBlock
        assertEquals(listOf("I.", "II.", "III.", "IV.", "V."), list.items.take(5).map { it.label })
        assertTrue(list.items[2].text.endsWith("can be affected"))
        // Consecutive labelled lines form one block; glyph bullets are labels too.
        assertEquals(7, list.items.size)
        assertEquals("\uF0B7", list.items[5].label)
    }

    @Test
    fun `arabic conversion covers digits roman and letters`() {
        assertEquals("13", StructureAnalyzer.arabic("XIII"))
        assertEquals("4", StructureAnalyzer.arabic("iv"))
        assertEquals("7", StructureAnalyzer.arabic("007"))
        assertEquals("A", StructureAnalyzer.arabic("a"))
    }
}
