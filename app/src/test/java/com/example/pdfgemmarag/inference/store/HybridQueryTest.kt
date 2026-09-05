package com.example.pdfgemmarag.inference.store

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HybridQueryTest {

    @Test
    fun `strips stopwords and keeps distinctive terms`() {
        val terms = HybridQuery.keywordTerms(
            "What city published these Division 3 concrete specifications, and what is the date?",
        )
        assertEquals(listOf("city", "published", "division", "concrete", "specifications", "date"), terms)
        assertFalse(terms.contains("what"))
        assertFalse(terms.contains("these"))
        assertFalse(terms.contains("and"))
    }

    @Test
    fun `formwork and plywood questions keep the nouns`() {
        assertTrue(HybridQuery.keywordTerms("Is concrete formwork paid as a separate bid item?").containsAll(listOf("concrete", "formwork", "paid", "separate")))
        assertTrue(HybridQuery.keywordTerms("What plywood standard and class is required for formwork?").containsAll(listOf("plywood", "standard", "class", "required", "formwork")))
    }

    @Test
    fun `build ORs each term parameter plus semanticSearch`() {
        val q = HybridQuery.build(listOf("city", "division"), similarityFloor = 0.3, vectorLimit = 12)
        assertEquals(
            "(getSearchStringParameter(0) OR getSearchStringParameter(1)) OR semanticSearch(getEmbeddingParameter(0), 0.3, 12)",
            q,
        )
    }

    @Test
    fun `no terms is semantic only`() {
        assertEquals(
            "semanticSearch(getEmbeddingParameter(0), 0.0, 8)",
            HybridQuery.build(emptyList(), 0.0, 8),
        )
    }

    @Test
    fun `keeps CJK tokens of any length`() {
        assertEquals(listOf("東京", "首都"), HybridQuery.keywordTerms("東京 is the 首都"))
    }
}
