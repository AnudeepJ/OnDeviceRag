package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Value-form matrix from the hardening plan: every supported form must survive at every streamed
 * token boundary, and synthetic unsupported forms must remain blocked with a stage reason.
 */
class GroundingValueFormsTest {

    private fun citation(text: String, id: String = "c1", excerpt: String = "E1") = Citation(
        id, "doc", 103, 1, 1.0, text, indexNamespace = "doc:v22:x", excerptId = excerpt,
    )

    /** Feeds [answer] through a fresh filter split at every character boundary and asserts identical output. */
    private fun assertSurvivesEverySplit(evidence: String, answer: String, expected: String = answer) {
        for (split in 0..answer.length) {
            val filter = GroundingStreamFilter("q", listOf(citation(evidence)))
            val output = filter.accept(answer.substring(0, split)) + filter.accept(answer.substring(split)) + filter.finish()
            assertEquals("split at $split", expected, output)
            assertFalse("split at $split flagged failure: ${filter.groundingReasons}", filter.hadGroundingFailure)
        }
    }

    @Test
    fun `compact metre survives when evidence is compact`() =
        assertSurvivesEverySplit("usually no more than 8m (25 ft) away from any worker", "No more than 8m (25 ft) away.")

    @Test
    fun `compact and spaced forms are interchangeable`() {
        assertSurvivesEverySplit("no more than 8m away", "No more than 8 m away.")
        assertSurvivesEverySplit("no more than 8 m away", "No more than 8m away.")
    }

    @Test
    fun `compact millimetres and mixed case tonnes survive`() {
        assertSurvivesEverySplit("Mid rail 600mm, top rail 1200mm and toe board 150mm", "Toe board 150mm, mid rail 600 mm.")
        assertSurvivesEverySplit("simple slope up to 6Mt (20 ft)", "Up to 6Mt (20 ft).")
    }

    @Test
    fun `standard identifier with year survives`() {
        assertSurvivesEverySplit("IS 3764:1992 Code of safety for excavation work", "See IS 3764:1992.")
        assertSurvivesEverySplit("IS 3764:1992 Code of safety for excavation work", "IS 3764 (1992) applies.")
    }

    @Test
    fun `ranges ratios and unicode fractions survive in restated forms`() {
        assertSurvivesEverySplit("21-25 Very high risk requires immediate attention", "A rating of 21-25 is Very high.")
        assertSurvivesEverySplit("21-25 Very high risk requires immediate attention", "Between 21 and 25 it is Very high.")
        assertSurvivesEverySplit("Type C soil maximum slope 1½:1", "Maximum slope is 1 1/2:1.")
        assertSurvivesEverySplit("Type C soil maximum slope 1½:1", "Maximum slope is 1½:1.")
        assertSurvivesEverySplit("withstand pressures up to 10,000 psi", "Up to 10000 psi.")
        assertSurvivesEverySplit("beams and columns: 2 1/2 inches", "Cover is 2 1/2 inches.")
    }

    @Test
    fun `heading identifier with hyphen grounds the spaced generated form`() {
        assertSurvivesEverySplit("Rule- 210. Inquiry into accidents.", "Rule 210 requires an inquiry.")
    }

    @Test
    fun `unsupported compact value is still blocked with a stage reason`() {
        val filter = GroundingStreamFilter("q", listOf(citation("no more than 8m away")))
        val output = filter.accept("No more than 9m away.") + filter.finish()

        assertEquals("No more than [unverified value]m away.", output)
        assertTrue(filter.hadGroundingFailure)
        assertTrue(filter.groundingReasons.toString(), filter.groundingReasons.any { it.startsWith("UNIT_MISMATCH") || it.startsWith("VALUE_ABSENT") })
    }

    @Test
    fun `value with the wrong unit is blocked when evidence distinguishes units`() {
        val filter = GroundingStreamFilter("q", listOf(citation("2400 psi and 4000 psi; protect below 40 degrees")))
        val output = filter.accept("Strength is 40 psi.") + filter.finish()

        assertEquals("Strength is 4000 psi.", output)
    }

    @Test
    fun `identifier from the question alone is blocked with identifier reason`() {
        val filter = GroundingStreamFilter("Does IS 9999 apply?", listOf(citation("No standard is named.")))
        val output = filter.accept("IS 9999 applies.") + filter.finish()

        assertEquals("IS [unverified value] applies.", output)
        assertTrue(filter.groundingReasons.any { it.startsWith("IDENTIFIER_MISMATCH") || it.startsWith("VALUE_ABSENT") })
    }

    @Test
    fun `identifier suffix letter is not treated as a unit`() {
        assertSurvivesEverySplit("Table 03210B Minimum concrete cover", "In Table 03210B the cover is listed.")
    }
}

class CitationMarkerToleranceTest {
    @Test
    fun `excerpt id with a list letter suffix resolves to the excerpt`() {
        val c = com.example.pdfgemmarag.core.model.Citation("c1", "doc", 97, 1, 1.0, "Confined space means a space that: a. bodily enter", indexNamespace = "x", excerptId = "E5")
        val filter = GroundingStreamFilter("q", listOf(c))
        val out = filter.accept("* bodily enter [E5a]") + filter.finish()
        assertEquals("* bodily enter [Page 97]", out)
        assertFalse(filter.hadGroundingFailure)
    }
}
