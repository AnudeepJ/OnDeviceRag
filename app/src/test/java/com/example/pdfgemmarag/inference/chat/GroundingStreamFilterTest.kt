package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GroundingStreamFilterTest {
    @Test
    fun `incomplete internal citation is discarded and marked as grounding failure`() {
        val source = citation.copy(text = "Values are 2 and 3.")
        val filter = GroundingStreamFilter("summarize", listOf(source))

        val output = filter.accept("Supported point [E") + filter.finish()

        assertEquals("Supported point ", output)
        assertTrue(filter.hadGroundingFailure)
    }


    @Test
    fun `table value remains grounded when its unit is in a separate header chunk`() {
        val header = citation.copy(text = "Minimum Cement Content (psi) Pounds per Cubic Yard")
        val row = citation.copy(chunkId = "c2", text = "| D | 5,000 | 658 (7 sacks) |")
        val filter = GroundingStreamFilter("Which class?", listOf(header, row))

        assertEquals("658 pounds", filter.accept("658 pounds") + filter.finish())
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `manifest validated numeric pointer is not mistaken for a unit bearing value`() {
        val source = citation.copy(text = "Other values are 17 days and 40 degrees.")
        val filter = GroundingStreamFilter("specification 03300", listOf(source), listOf("03300"))

        assertEquals(
            "Specification 03300 is applicable.",
            filter.accept("Specification 03300 is applicable.") + filter.finish(),
        )
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `retrieved citation identifiers are authoritative structural metadata`() {
        val source = citation.copy(specificationNumber = "03300", sectionNumber = "2.05")
        val filter = GroundingStreamFilter("ratio", listOf(source))

        assertEquals(
            "Specification 03300 section 2.05 is applicable.",
            filter.accept("Specification 03300 section 2.05 is applicable.") + filter.finish(),
        )
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `wrong generated structural pointers are corrected from unique citation metadata`() {
        val source = citation.copy(specificationNumber = "03300", sectionNumber = "2.05")
        val filter = GroundingStreamFilter("ratio", listOf(source))

        assertEquals(
            "Specification 03300 section 2.05 applies.",
            filter.accept("Specification 03310 section 2.06 applies.") + filter.finish(),
        )
        assertFalse(filter.hadGroundingFailure)
    }
    private val citation = Citation(
        "c1", "doc", 17, 1, 1.0,
        "Specification 03300 section 2.05 requires 5000 psi, ratio 0.55, range 10-15%, and 1/2 inch.",
        indexNamespace = "doc:v21:x", excerptId = "E1",
    )

    @Test
    fun `streams fragmented values and maps excerpt marker`() {
        val filter = GroundingStreamFilter("Summarize section 2.05", listOf(citation))
        val fragments = listOf("Use ", "50", "00 psi at ", "0.", "55 ", "[", "E1", "]")
        val output = buildString {
            fragments.forEach { append(filter.accept(it)) }
            append(filter.finish())
        }
        assertEquals("Use 5000 psi at 0.55 [Page 17]", output)
        assertEquals(listOf("c1"), filter.usedCitations.map { it.chunkId })
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `repairs unique numeric prefix without buffering whole answer`() {
        val filter = GroundingStreamFilter("ratio?", listOf(citation))
        val first = filter.accept("The ratio is ")
        val numeric = filter.accept("0.5.") + filter.finish()
        assertEquals("The ratio is ", first)
        assertEquals("0.55.", numeric)
    }

    @Test
    fun `unsupported identifier is not allowed through`() {
        val filter = GroundingStreamFilter("Which specification?", listOf(citation))
        val output = filter.accept("It is 030.") + filter.finish()
        assertTrue(output.contains("[unverified value]"))
        assertTrue(filter.hadGroundingFailure)
    }

    @Test
    fun `identifier asserted by question is not authoritative evidence`() {
        val source = citation.copy(text = "The retrieved requirement does not identify an ISO standard.")
        val filter = GroundingStreamFilter("Does ISO9999 apply?", listOf(source))

        val output = filter.accept("ISO9999 applies.") + filter.finish()

        assertEquals("ISO[unverified value] applies.", output)
        assertTrue(filter.hadGroundingFailure)
    }

    @Test
    fun `spaced identifier asserted by question is not authoritative evidence`() {
        val source = citation.copy(text = "The authority must initiate an inquiry.")
        val filter = GroundingStreamFilter("What does Rule 999 require?", listOf(source))

        val output = filter.accept("Rule 999 requires an inquiry.") + filter.finish()

        assertEquals("Rule [unverified value] requires an inquiry.", output)
        assertTrue(filter.hadGroundingFailure)
    }

    @Test
    fun `lexicon covers formats used by specifications`() {
        val values = GroundingStreamFilter.valueTokens(citation.text)
        assertTrue(values.containsAll(listOf("03300", "2.05", "5000", "0.55", "10-15", "1/2")))
    }

    @Test
    fun `list marker at line start is structural`() {
        val filter = GroundingStreamFilter("list requirements", listOf(citation))
        assertEquals("1. Item", filter.accept("1. Item") + filter.finish())
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `metadata values visible to model are grounded evidence`() {
        val enriched = citation.copy(
            specificationNumber = "03300",
            sectionNumber = "2.05",
            sectionPath = "03300 CAST-IN-PLACE CONCRETE > 2.05 CONCRETE MIX",
        )
        val filter = GroundingStreamFilter("which specification", listOf(enriched))

        val output = filter.accept("Specification 03300 section 2.05 [E1]") + filter.finish()

        assertEquals("Specification 03300 section 2.05 [Page 17]", output)
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `combined excerpt citations map to distinct source pages`() {
        val second = citation.copy(chunkId = "c2", pageNumber = 18, excerptId = "E2")
        val filter = GroundingStreamFilter("summarize", listOf(citation, second))

        val output = listOf("Supported ", "[E1, E", "2]").joinToString("") { filter.accept(it) } + filter.finish()

        assertEquals("Supported [Page 17][Page 18]", output)
        assertEquals(listOf("c1", "c2"), filter.usedCitations.map { it.chunkId })
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `numeric value is repaired using its following unit`() {
        val source = citation.copy(text = "Strengths are 2400 psi and 4000 psi; placement below 40 degrees requires protection.")
        val filter = GroundingStreamFilter("strength", listOf(source))

        val output = listOf("Strength is ", "40", " psi.").joinToString("") { filter.accept(it) } + filter.finish()

        assertEquals("Strength is 4000 psi.", output)
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `fragmented compound identifier remains allowed`() {
        val source = citation.copy(text = "Conform to ASTM C94 and ASTM C150.")
        val filter = GroundingStreamFilter("standard", listOf(source))

        val output = filter.accept("Use ASTM C") + filter.accept("94.") + filter.finish()

        assertEquals("Use ASTM C94.", output)
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `ordinary following word does not turn an allowed value into a unit mismatch`() {
        val enriched = citation.copy(specificationNumber = "03300")
        val filter = GroundingStreamFilter("which specification", listOf(enriched))

        val output = filter.accept("Specification 03300 is required.") + filter.finish()

        assertEquals("Specification 03300 is required.", output)
        assertFalse(filter.hadGroundingFailure)
    }

    @Test
    fun `hyphenated dimension survives numeric token boundary`() {
        val source = citation.copy(text = "Test a 6-inch diameter cylinder.")
        val filter = GroundingStreamFilter("diameter", listOf(source))

        val output = filter.accept("Use a 6-") + filter.accept("inch cylinder.") + filter.finish()

        assertEquals("Use a 6-inch cylinder.", output)
        assertFalse(filter.hadGroundingFailure)
    }
}
