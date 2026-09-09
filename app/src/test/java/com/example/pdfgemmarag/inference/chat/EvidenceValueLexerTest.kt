package com.example.pdfgemmarag.inference.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EvidenceValueLexerTest {

    @Test
    fun `compact and spaced units both produce a measurement with the same normalized pair`() {
        val compact = EvidenceValueLexer.lex("no more than 8m (25 ft) away")
        val spaced = EvidenceValueLexer.lex("no more than 8 m (25 ft) away")

        assertTrue(compact.valueUnits.contains("8" to "m"))
        assertTrue(spaced.valueUnits.contains("8" to "m"))
        assertTrue(compact.valueUnits.contains("25" to "ft"))
        assertTrue(compact.values.containsAll(listOf("8", "25")))
    }

    @Test
    fun `mixed case compact unit is a unit but a single capital suffix is an identifier`() {
        val lexicon = EvidenceValueLexer.lex("up to 6Mt and Table 03210B")

        assertTrue(lexicon.valueUnits.contains("6" to "mt"))
        assertTrue(lexicon.values.contains("03210"))
        assertFalse(lexicon.valueUnits.any { it.first == "03210" })
        val id = lexicon.tokens.first { it.normalized == "03210" }
        assertEquals(EvidenceValueKind.COMPOUND_ID, id.kind)
    }

    @Test
    fun `ranges expose both ends and keep the range form`() {
        val lexicon = EvidenceValueLexer.lex("Risk 21-25 Very high; 13 to 20 High; 10–15% air")

        assertTrue(lexicon.values.containsAll(listOf("21-25", "21", "25", "13-20", "13", "20", "10-15", "10", "15")))
        assertEquals(EvidenceValueKind.RANGE, lexicon.tokens.first { it.normalized == "21-25" }.kind)
    }

    @Test
    fun `ratios and unicode fractions are folded and decomposed`() {
        val lexicon = EvidenceValueLexer.lex("Type B soil 1:1 up to 6 m; Type C 1½:1; cover 2 1/2 inches; ±1/2\" tolerance")

        assertTrue(lexicon.values.containsAll(listOf("1:1", "1 1/2:1", "1 1/2", "1/2", "2 1/2", "2", "±1/2")))
        assertEquals(EvidenceValueKind.RATIO, lexicon.tokens.first { it.normalized == "1 1/2:1" }.kind)
        assertTrue(lexicon.valueUnits.contains("2 1/2" to "inches"))
        assertTrue(lexicon.valueUnits.contains("6" to "m"))
    }

    @Test
    fun `standard identifiers keep their year and expose it as a value`() {
        val lexicon = EvidenceValueLexer.lex("IS 3764:1992 Code of safety for excavation work; ASTM C150; Rule- 210")

        assertTrue(lexicon.identifiers.toString(), lexicon.identifiers.containsAll(listOf("is3764:1992", "is3764", "astmc150", "rule210")))
        assertTrue(lexicon.values.toString(), lexicon.values.containsAll(listOf("1992", "3764", "210")))
        // The digits inside a letter-prefixed identifier are not free-standing evidence.
        assertFalse(lexicon.values.contains("150"))
    }

    @Test
    fun `thousands separators and percent are normalized`() {
        val lexicon = EvidenceValueLexer.lex("withstand 10,000 psi; 4-6% air; 5,000")

        assertTrue(lexicon.valueUnits.contains("10000" to "psi"))
        assertTrue(lexicon.values.containsAll(listOf("10000", "4-6", "4", "6", "5000")))
        assertEquals("10000", EvidenceValueLexer.normalizeNumber("10,000"))
        assertEquals("21-25", EvidenceValueLexer.normalizeNumber("21 – 25"))
        assertEquals("21-25", EvidenceValueLexer.normalizeNumber("21 to 25"))
    }

    @Test
    fun `existing specification lexicon is preserved`() {
        val values = EvidenceValueLexer.lex(
            "Specification 03300 section 2.05 requires 5000 psi, ratio 0.55, range 10-15%, and 1/2 inch.",
        ).values
        assertTrue(values.containsAll(listOf("03300", "2.05", "5000", "0.55", "10-15", "1/2")))
    }

    @Test
    fun `production grammar contains no document vocabulary`() {
        val source = java.io.File("src/main/java/com/example/pdfgemmarag/inference/chat/EvidenceValueLexer.kt").readText().lowercase()
        listOf("psi", "mm\"", "excavation", "concrete", "scaffold", "safety").forEach { word ->
            assertFalse("lexer must not whitelist '$word'", Regex("\"[^\"]*$word[^\"]*\"").containsMatchIn(source))
        }
    }
}
