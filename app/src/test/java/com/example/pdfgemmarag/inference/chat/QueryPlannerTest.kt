package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.SectionRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class QueryPlannerTest {
    private val first = section("a", "03300", "2.05", "CONCRETE MIX", 17)
    private val second = section("b", "03310", "2.02", "CONCRETE MIX", 60)
    private val manifest = DocumentStructureManifest(
        "doc", "doc:v21:test", DocumentStructureManifest.INDEX_VERSION, "sig", listOf(first, second),
    )

    @Test
    fun `duplicate exact titles ask for deterministic clarification`() {
        val plan = QueryPlanner().plan("Summarize concrete mix", manifest)
        assertEquals(QuestionIntent.AMBIGUOUS_SECTION, plan.intent)
        assertEquals(listOf("a", "b"), plan.candidateSections.map { it.sectionId })
        assertTrue(plan.clarification().contains("03300"))
        assertTrue(plan.clarification().contains("03310"))
    }

    @Test
    fun `combined specification and section resolve uniquely`() {
        val plan = QueryPlanner().plan("Summarize specification 03300 section 2.05 concrete mix", manifest)
        assertEquals(QuestionIntent.SECTION_SUMMARY, plan.intent)
        assertEquals("a", plan.resolvedSectionId)
        assertEquals("03300", plan.explicitSpecificationNumber)
        assertEquals("2.05", plan.explicitSectionNumber)
    }

    @Test
    fun `explicit specification number is not treated as document overview`() {
        val plan = QueryPlanner().plan(
            "Summarize section 2.05 concrete mix in specification 03300",
            manifest = null,
        )

        assertEquals(QuestionIntent.SECTION_SUMMARY, plan.intent)
    }

    @Test
    fun `document summary is classified as structural overview`() {
        val plan = QueryPlanner().plan("Summarize this document", manifest)

        assertEquals(QuestionIntent.DOCUMENT_OVERVIEW, plan.intent)
        assertNull(plan.resolvedSectionId)
    }

    @Test
    fun `generic fact does not pretend a section was resolved`() {
        val plan = QueryPlanner().plan("What is the required curing temperature?", manifest)
        assertEquals(QuestionIntent.FACT, plan.intent)
        assertNull(plan.resolvedSectionId)
    }

    @Test
    fun `fact question naming a specification does not pin to a one-word title`() {
        val heading = section("sealant-heading", "04100", "", "SEALANT", 10)
        val products = section("sealant-products", "04100", "2.02", "PREPACKAGED SEALANTS", 12)
        val other = section("c", "03300", "2.05", "CONCRETE MIX", 17)
        val fixture = DocumentStructureManifest(
            "doc", "doc:v21:test", DocumentStructureManifest.INDEX_VERSION, "sig",
            listOf(heading, products, other),
        )

        val plan = QueryPlanner().plan(
            "What minimum 28-day compressive strength is required for prepackaged sealant in specification 04100?",
            fixture,
        )

        assertEquals(QuestionIntent.FACT, plan.intent)
        assertEquals("04100", plan.explicitSpecificationNumber)
        assertNull(plan.resolvedSectionId)
    }

    @Test
    fun `table number that is not a document specification is not used as a spec filter`() {
        val plan = QueryPlanner().plan(
            "In Table 03210B, what minimum concrete cover is required for principal reinforcement?",
            manifest,
        )

        assertEquals(QuestionIntent.FACT, plan.intent)
        assertNull(plan.explicitSpecificationNumber)
        assertNull(plan.resolvedSectionId)
    }

    @Test
    fun `summarize still resolves a unique title after a specification is stripped`() {
        val heading = section("sealant-heading", "04100", "", "SEALANT", 10)
        val products = section("sealant-products", "04100", "2.02", "PREPACKAGED SEALANTS", 12)
        val fixture = DocumentStructureManifest(
            "doc", "doc:v21:test", DocumentStructureManifest.INDEX_VERSION, "sig",
            listOf(heading, products),
        )

        val plan = QueryPlanner().plan("Summarize sealant", fixture)

        assertEquals(QuestionIntent.SECTION_SUMMARY, plan.intent)
        assertEquals("sealant-heading", plan.resolvedSectionId)
    }

    @Test
    fun `bounded distance has no third party dependency behavior`() {
        assertEquals(1, QueryPlanner.boundedLevenshtein("concrete", "concret", 3))
        assertEquals(3, QueryPlanner.boundedLevenshtein("abc", "xyz", 2))
    }

    private fun section(id: String, spec: String, number: String, title: String, page: Int) = SectionRecord(
        id, spec, number, title, "$spec > $number $title", page, page, listOf("c$id"), 100,
    )
}
