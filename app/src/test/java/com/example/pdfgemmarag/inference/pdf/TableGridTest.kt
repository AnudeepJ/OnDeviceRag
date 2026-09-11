package com.example.pdfgemmarag.inference.pdf

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.chat.AnswerQuestionUseCase
import com.example.pdfgemmarag.inference.chunk.ScriptAwareChunker
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Roadmap M4: unbordered grids with wrapped cells, captions, header-qualified row facts, matrix lookups. */
class TableGridTest {

    private fun line(y: Float, vararg cells: Pair<Float, String>): LineBox = LineBox(
        cells.flatMap { (x, text) ->
            var cursor = x
            text.split(' ').map { w ->
                val box = Box(cursor, y, cursor + w.length * 6f, y + 10f)
                cursor += w.length * 6f + 6f
                WordBox(w, box)
            }
        },
    )

    private fun page(lines: List<LineBox>) = PageLayout(1, 600f, 800f, lines, hasImages = false, source = PageLayout.Source.TEXT)

    /** The risk-description grid: a multi-line description next to a one-line range and a one-line control. */
    private val riskGrid = listOf(
        line(40f, 40f to "The priority of the control elements can be defined by the relative risk."),
        line(60f, 40f to "Risk", 100f to "Description", 460f to "Control"),
        line(74f, 40f to "1-4", 100f to "May be considered as Acceptable. However, a", 460f to "Low"),
        line(86f, 100f to "consistent watch may be made on its hazard."),
        line(100f, 40f to "5-12", 100f to "A planned approach may be developed to", 460f to "Medium"),
        line(112f, 100f to "control the hazard. Action must be documented"),
        line(124f, 100f to "and completion time limit may be decided."),
        line(138f, 40f to "13-20", 100f to "Risk requires an attention; action may be", 460f to "High"),
        line(150f, 100f to "followed up and completion time limit decided."),
        line(164f, 40f to "21-25", 100f to "Risk requires an immediate attention; action", 460f to "Very high"),
        line(176f, 100f to "must be documented and completion time decided."),
        line(196f, 40f to "Risk Assessment combines the consequences and likelihood of all incident outcomes."),
        line(210f, 40f to "Table 5.1 Relative risk and control level"),
    )

    @Test
    fun `wrapped description cells stay in their row and the control column is not interleaved`() {
        val content = TableClusterer().analyse(page(riskGrid))
        val table = content.segments.filterIsInstance<Segment.Table>().single()
        assertEquals("| Risk | Description | Control |", table.header)
        assertEquals(4, table.rows.size)
        assertEquals("| 21-25 | Risk requires an immediate attention; action must be documented and completion time decided. | Very high |", table.rows[3])
        assertTrue(table.rows[1].contains("| 5-12 | A planned approach may be developed to control the hazard. Action must be documented and completion time limit may be decided. | Medium |"))
        val paragraphs = content.segments.filterIsInstance<Segment.Paragraph>()
        assertTrue(paragraphs.any { it.text.startsWith("Risk Assessment combines") })
        assertTrue(paragraphs.none { it.text.contains("Very high") })
    }

    @Test
    fun `a caption line after the grid is attached and removed from prose`() {
        val content = TableClusterer().analyse(page(riskGrid.filterNot { it.text.startsWith("Risk Assessment") }))
        val table = content.segments.filterIsInstance<Segment.Table>().single()
        assertEquals("Table 5.1 Relative risk and control level", table.caption)
        assertEquals("5.1", table.tableNumber)
        assertFalse(content.segments.filterIsInstance<Segment.Paragraph>().any { it.text.contains("Table 5.1") })
    }

    @Test
    fun `header qualified row facts render column names with values`() {
        val facts = ScriptAwareChunker().tableRowFacts(
            "| Risk | Description | Control |\n| --- | --- | --- |\n| 21-25 | Risk requires an immediate attention | Very high |",
        )
        assertEquals("Risk: 21-25; Description: Risk requires an immediate attention; Control: Very high", facts)
        val matrix = ScriptAwareChunker().tableRowFacts(
            "|  | Insignificant | Minor | Moderate |\n| --- | --- | --- | --- |\n| Almost Certain | 5 | 10 | 15 |",
        )
        assertEquals("Almost Certain — Insignificant: 5; Minor: 10; Moderate: 15", matrix)
    }

    @Test
    fun `matrix cell lookup resolves row label times column header`() {
        val table = Citation(
            "t", "doc", 10, 1, 1.0,
            "|  | Insignificant | Minor | Moderate | Major | Catastrophic |\n| --- | --- | --- | --- | --- | --- |\n" +
                "| Almost Certain | 5 | 10 | 15 | 20 | 25 |\n| Likely | 4 | 8 | 12 | 16 | 20 |\n| Possible | 3 | 6 | 9 | 12 | 15 |",
            indexNamespace = "x", contentKind = "TABLE",
        )
        val lead = AnswerQuestionUseCase.buildTableLead(
            "In Table 1.1, what risk score results from Almost Certain likelihood and Catastrophic consequence?",
            listOf(table),
        )
        assertTrue(lead.decisive)
        assertEquals("Almost Certain × Catastrophic: 25 [Page 10]", lead.text)

        val possible = AnswerQuestionUseCase.buildTableLead(
            "Using Table 1.1, what is the assessed risk for Possible likelihood and Moderate consequence?",
            listOf(table),
        )
        assertEquals("Possible × Moderate: 9 [Page 10]", possible.text)
    }

    @Test
    fun `parenthesized table number is attached even when a score line sits between caption and grid`() {
        val lines = listOf(
            line(40f, 40f to "Table (1.1) Risk Level Assessment Consequences Insignificant Minor (2) Moderate Major (4) Catastrophic (5)"),
            line(54f, 40f to "(1)", 80f to "(3)", 120f to "hic", 160f to "(5)"),
            line(70f, 40f to "Almost Certain", 140f to "5", 200f to "10", 260f to "15", 320f to "20", 380f to "25"),
            line(84f, 40f to "Possible", 140f to "3", 200f to "6", 260f to "9", 320f to "12", 380f to "15"),
        )
        val content = StructureAnalyzer().analyse(page(lines))
        val table = content.segments.filterIsInstance<Segment.Table>().single()
        assertEquals("1.1", table.tableNumber)
        assertTrue(table.caption, table.caption.contains("1.1"))
        assertTrue(table.rows.joinToString(" ").contains("25") || table.header.contains("25"))
    }

    @Test
    fun `caption survives a three-line compound header before the detected grid`() {
        val content = StructureAnalyzer().analyse(page(listOf(
            line(40f, 40f to "Table 1.4 Assessed Risk Response"),
            line(52f, 40f to "Risk response categories"),
            line(64f, 40f to "Required management action"),
            line(76f, 40f to "Risk", 180f to "Required action"),
            line(90f, 40f to "Extreme", 180f to "Immediate action and management approval"),
            line(104f, 40f to "High", 180f to "Significant control measures"),
        )))

        val table = content.segments.filterIsInstance<Segment.Table>().single()
        assertEquals("1.4", table.tableNumber)
        assertTrue(table.caption.contains("Assessed Risk Response"))
    }

    @Test
    fun `an uppercase table caption is not a heading and stays attached to the grid`() {
        val lines = listOf(
            line(50f, 40f to "TABLE 5.1 RELATIVE RISK AND CONTROL LEVEL"),
            line(70f, 40f to "Risk", 100f to "Description", 460f to "Control"),
            line(84f, 40f to "21-25", 100f to "Risk requires an immediate attention", 460f to "Very high"),
        )
        val content = StructureAnalyzer().analyse(page(lines))
        val table = content.segments.filterIsInstance<Segment.Table>().single()
        assertEquals("TABLE 5.1 RELATIVE RISK AND CONTROL LEVEL", table.caption)
        assertEquals("5.1", table.tableNumber)
        assertTrue(content.segments.none { it is Segment.Heading && it.title.contains("RELATIVE RISK") })
    }

    @Test
    fun `prose after a grid is not absorbed as a wrapped cell`() {
        val lines = listOf(
            line(60f, 40f to "Class", 200f to "Strength", 400f to "Cement"),
            line(74f, 40f to "A", 200f to "4000", 400f to "564"),
            line(88f, 40f to "D", 200f to "5000", 400f to "658"),
            line(100f, 40f to "Cement content is given in pounds per cubic yard of concrete placed."),
        )
        val content = TableClusterer().analyse(page(lines))
        val table = content.segments.filterIsInstance<Segment.Table>().single()
        assertEquals(2, table.rows.size)
        assertTrue(content.segments.filterIsInstance<Segment.Paragraph>().single().text.startsWith("Cement content"))
    }
}

class RowKeyLeadTest {
    private fun table(text: String, caption: String = "") = Citation(
        "t", "doc", 28, 1, 1.0, text,
        indexNamespace = "x", contentKind = "TABLE", tableCaption = caption,
    )

    @Test
    fun `range row and named column answer from the grid alone`() {
        val table = table(
            "| Risk | Description | Control |\n| --- | --- | --- |\n" +
                "| 13-20 | Risk requires an attention | High |\n" +
                "| 21-25 | Risk requires an immediate attention | Very high |",
            "Table 5.1 Relative risk and control level",
        )
        val lead = AnswerQuestionUseCase.buildTableLead(
            "In the relative risk table, what control level applies to a risk value of 13-20?",
            listOf(table),
        )
        assertTrue(lead.decisive)
        assertTrue(lead.text, lead.text.contains("High"))
        assertFalse(lead.text.contains("Very high"))
        assertTrue(lead.text.contains("13-20"))
    }

    @Test
    fun `between and phrasing matches a hyphenated range row`() {
        val table = table(
            "| Risk | Description | Control |\n| --- | --- | --- |\n" +
                "| 21-25 | Risk requires an immediate attention | Very high |",
            "Table 5.1 Relative risk and control level",
        )
        val lead = AnswerQuestionUseCase.buildTableLead(
            "In the risk assessment matrix, what is the description and control level for a relative risk rating between 21 and 25?",
            listOf(table),
        )
        assertTrue(lead.decisive)
        assertTrue(lead.text.contains("immediate attention"))
        assertTrue(lead.text.contains("Very high"))
    }

    @Test
    fun `row label plus rating column is a unique cell`() {
        val table = table(
            "| Likelihood | Meaning | Rating |\n| --- | --- | --- |\n" +
                "| Almost Certain | Expected | 5 |\n| Possible | Could occur | 4 |",
            "Table 5.2 Likelihood",
        )
        val lead = AnswerQuestionUseCase.buildTableLead(
            "In the likelihood table, what rating is given to a Possible occurrence?",
            listOf(table),
        )
        assertTrue(lead.decisive)
        assertTrue(lead.text, lead.text.contains("Possible"))
        assertTrue(lead.text.contains("4"))
        assertFalse(lead.text.contains("Almost Certain"))
    }

    @Test
    fun `a generic Class header row does not answer a plywood class question`() {
        val schedule = table(
            "| Class | Length in Feet | in Inches |\n| --- | --- | --- |\n| Class | 10 | 120 |",
            "Table 2 Form ties",
        )
        val plywood = table(
            "| Product | Standard | Class |\n| --- | --- | --- |\n| Plywood | PS 1 | Class 1 |",
            "Table 1 Form materials",
        ).copy(chunkId = "t2", pageNumber = 12)
        val lead = AnswerQuestionUseCase.buildTableLead(
            "What standard and class must plywood form material meet",
            listOf(schedule, plywood),
        )
        if (lead.decisive) {
            assertTrue(lead.text, lead.text.contains("PS 1"))
            assertFalse(lead.text.contains("Length in Feet"))
        }
    }

    @Test
    fun `a class question names the letter cell as Class D`() {
        val table = table(
            "| Material | Examples | Class | Extinguisher |\n| --- | --- | --- | --- |\n" +
                "| Metal | Magnesium, Aluminum | D | Special metal extinguishers |",
            "Table 3 Fire extinguishers",
        )
        val lead = AnswerQuestionUseCase.buildTableLead(
            "In the fire-extinguisher table, what class covers metal fires and how should they be extinguished?",
            listOf(table),
        )
        assertTrue(lead.decisive)
        assertTrue(lead.text, lead.text.contains("Class D"))
        assertTrue(lead.text.contains("Special metal extinguishers"))
    }

    @Test
    fun `typed short class label selects that row`() {
        val table = table(
            "| Class | Strength | Cement |\n| --- | --- | --- |\n| A | 4000 | 564 |\n| D | 5000 | 658 |",
            "Table 3 Concrete classes",
        )
        val lead = AnswerQuestionUseCase.buildTableLead(
            "In the class table, what cement content is required for Class D?",
            listOf(table),
        )
        assertTrue(lead.decisive)
        assertTrue(lead.text.contains("658"))
        assertTrue(lead.text.contains("D"))
        assertFalse(lead.text.contains("564"))
    }

    @Test
    fun `the same row key in two tables is ambiguous without a caption hint`() {
        val first = table(
            "| Label | Rating |\n| --- | --- |\n| Possible | 4 |",
            "Table 1 Likelihood",
        )
        val second = table(
            "| Label | Rating |\n| --- | --- |\n| Possible | 2 |",
            "Table 2 Severity",
        ).copy(chunkId = "t2", pageNumber = 29)
        val lead = AnswerQuestionUseCase.buildTableLead(
            "In the table, what rating is given to a Possible occurrence?",
            listOf(first, second),
        )
        assertFalse(lead.decisive)
    }
}

class MatrixScaleLabelTest {
    @Test
    fun `scale suffixes in labels are ignored when matching the question`() {
        val table = Citation(
            "t", "doc", 10, 1, 1.0,
            "|  | Insignificant (1) | Minor (2) | Moderate (3) | Major (4) | Catastrophic (5) |\n| --- | --- | --- | --- | --- | --- |\n" +
                "| Almost Certain (5) | Moderate (5) | High (10) | High (15) | Catastrophic (20) | Catastrophic (25) |\n" +
                "| Possible (3) | Low (3) | Moderate (6) | Moderate (9) | High (12) | High (15) |",
            indexNamespace = "x", contentKind = "TABLE",
        )
        val lead = AnswerQuestionUseCase.buildTableLead(
            "In Table 1.1, what risk score results from Almost Certain likelihood and Catastrophic consequence?",
            listOf(table),
        )
        assertTrue(lead.decisive)
        assertEquals("Almost Certain (5) × Catastrophic (5): Catastrophic (25) [Page 10]", lead.text)
    }
}
