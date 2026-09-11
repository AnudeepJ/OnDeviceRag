package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.SectionRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Chapter-style outlines: structural pointers, titled chapter requests, answer shapes, health, overview sampling. */
class StructuralPlanningTest {

    private fun node(
        id: String, kind: String, printed: String, number: String, title: String, path: String,
        start: Int, end: Int, level: Int, chunks: Int = 3, parent: String = "",
    ) = SectionRecord(
        id, "", number, title, path, start, end, (0 until chunks).map { "c$id$it" }, 100,
        level = level, kind = kind, printedNumber = printed, parentSectionId = parent,
    )

    private val ch12 = node("12", "CHAPTER", "12", "CHAPTER 12", "CONFINED SPACE", "CHAPTER 12 CONFINED SPACE", 97, 97, 1)
    private val ch12a = node("12a", "CLAUSE", "12.1", "12.1", "INTRODUCTION", "CHAPTER 12 CONFINED SPACE > 12.1 INTRODUCTION", 97, 99, 4, parent = "12")
    private val ch13 = node("13", "CHAPTER", "13", "CHAPTER 13", "EXCAVATION", "CHAPTER 13 EXCAVATION", 102, 102, 1)
    private val ch13a = node("13a", "CLAUSE", "13.1", "13.1", "INTRODUCTION", "CHAPTER 13 EXCAVATION > 13.1 INTRODUCTION", 102, 104, 4, parent = "13")
    private val ch13b = node("13b", "CLAUSE", "13.3", "13.3", "TEMPORARY PROTECTIVE STRUCTURE", "CHAPTER 13 EXCAVATION > 13.3 TEMPORARY PROTECTIVE STRUCTURE", 105, 107, 4, parent = "13")
    private val ch1 = node("1", "CHAPTER", "1", "CHAPTER 1", "INTRODUCTION", "CHAPTER 1 INTRODUCTION", 3, 6, 1)
    private val ch7 = node("7", "CHAPTER", "7", "CHAPTER 7", "TRAINING AND EDUCATION", "CHAPTER 7 TRAINING AND EDUCATION", 44, 51, 1)
    private val ch18 = node("18", "CHAPTER", "18", "CHAPTER 18", "CONSTRUCTION MACHINERY", "CHAPTER 18 CONSTRUCTION MACHINERY", 128, 145, 1)
    private val ch22 = node("22", "CHAPTER", "22", "CHAPTER 22", "PERSONAL PROTECTIVE EQUIPMENT", "CHAPTER 22 PERSONAL PROTECTIVE EQUIPMENT", 163, 174, 1)
    private val ch28 = node("28", "CHAPTER", "28", "CHAPTER 28", "LIST OF INDIAN STANDARDS", "CHAPTER 28 LIST OF INDIAN STANDARDS", 206, 208, 1)
    private val excavationBody = node("x", "HEADING", "", "", "EXCAVATION", "CHAPTER 3 SITE MANAGEMENT > EXCAVATION", 14, 14, 3)

    private val manifest = DocumentStructureManifest(
        "doc", "doc:v22:test", DocumentStructureManifest.INDEX_VERSION, "sig",
        listOf(ch1, excavationBody, ch7, ch12, ch12a, ch13, ch13a, ch13b, ch18, ch22, ch28),
    )

    @Test
    fun `explicit chapter number resolves to the chapter subtree for a summary`() {
        val plan = QueryPlanner().plan("Summarize Chapter 13 Excavation", manifest)
        assertEquals(QuestionIntent.SECTION_SUMMARY, plan.intent)
        assertEquals("13", plan.resolvedSectionId)
        assertTrue(plan.resolvedSubtree)
        assertEquals(StructuralPointer("CHAPTER", "13"), plan.structuralPointer)
    }

    @Test
    fun `roman chapter reference resolves like the arabic one`() {
        val plan = QueryPlanner().plan("Summarise chapter xiii", manifest)
        assertEquals("13", plan.resolvedSectionId)
    }

    @Test
    fun `titled chapter request resolves against chapter titles only`() {
        val plan = QueryPlanner().plan("Summarize the chapter on excavation.", manifest)
        assertEquals(QuestionIntent.SECTION_SUMMARY, plan.intent)
        assertEquals("13", plan.resolvedSectionId)
        assertTrue(plan.resolvedSubtree)
    }

    @Test
    fun `fact inside a chapter stays unscoped so children are searchable`() {
        val plan = QueryPlanner().plan("What does chapter 13 say about trench exits?", manifest)
        assertEquals(QuestionIntent.FACT, plan.intent)
        assertNull(plan.resolvedSectionId)
        assertEquals(StructuralPointer("CHAPTER", "13"), plan.structuralPointer)
    }

    @Test
    fun `unknown chapter number falls through instead of refusing`() {
        val plan = QueryPlanner().plan("Summarize chapter 40", manifest)
        assertFalse(plan.intent == QuestionIntent.AMBIGUOUS_SECTION)
        assertNull(plan.resolvedSectionId)
    }

    @Test
    fun `answer shapes are detected from wording alone`() {
        assertEquals(AnswerShape.DEFINITION, AnswerShape.of("What makes a space a confined space?"))
        assertEquals(AnswerShape.DEFINITION, AnswerShape.of("What is a trench?"))
        assertEquals(AnswerShape.DEFINITION, AnswerShape.of("What is an Emergency Action Plan?"))
        assertEquals(AnswerShape.NAVIGATION, AnswerShape.of("Which section covers the testing laboratory?"))
        assertEquals(AnswerShape.NAVIGATION, AnswerShape.of("Where in the document is curing discussed?"))
        assertEquals(AnswerShape.PROCEDURE, AnswerShape.of("What are the first aid procedures for an amputated body part?"))
        assertEquals(AnswerShape.TABLE, AnswerShape.of("In Table 1.1, what risk score results from Almost Certain likelihood?"))
        assertEquals(AnswerShape.DEFAULT, AnswerShape.of("What is the maximum distance allowed for a means of exit?"))
        assertEquals(AnswerShape.DEFAULT, AnswerShape.of("What is a scaffold required to support at minimum?"))
        assertEquals(AnswerShape.DEFINITION, QueryPlanner().plan("What makes a space a confined space?", manifest).shape)
    }

    @Test
    fun `descendants include the chapter and its clauses in order`() {
        val ids = manifest.descendantsOf(ch13).map { it.sectionId }
        assertEquals(listOf("13", "13a", "13b"), ids)
    }

    @Test
    fun `top level sections are the chapters and health is good when quartiles are covered`() {
        assertEquals(listOf("1", "7", "12", "13", "18", "22", "28"), manifest.topLevelSections().map { it.sectionId })
        val health = manifest.health()
        assertFalse(health.reasons.toString(), health.degraded)
        assertEquals(listOf(true, true, true, true), health.quartileCoverage)
    }

    @Test
    fun `manifest whose top level nodes sit only in the last quartile is degraded`() {
        val tail = DocumentStructureManifest(
            "doc", "doc:v22:test", DocumentStructureManifest.INDEX_VERSION, "sig",
            listOf(ch22, ch28, excavationBody.copy(startPage = 200, endPage = 200)),
        )
        val health = tail.health()
        assertTrue(health.degraded)
        assertTrue(health.reasons.any { "coverage" in it })
    }

    @Test
    fun `overview sampling reserves slots per quartile instead of the final appendix`() {
        val many = (1..28).map { n ->
            val id = "k%02d".format(n)
            node(id, "CHAPTER", "$n", "CHAPTER $n", "TITLE $n", "CHAPTER $n TITLE $n", 3 + (n - 1) * 7, 3 + n * 7 - 1, 1)
        }
        val wide = DocumentStructureManifest("doc", "doc:v22:test", DocumentStructureManifest.INDEX_VERSION, "sig", many)
        val ids = AnswerQuestionUseCase.overviewChunkIds(wide, maxSections = 8, chunksPerSection = 2)
        val pages = many.filter { s -> s.orderedChunkIds.any(ids::contains) }.map { it.startPage }
        assertEquals(8, pages.size)
        val last = wide.lastPage
        (0 until 4).forEach { q ->
            val from = 1 + q * last / 4
            val to = if (q == 3) last else (q + 1) * last / 4
            assertTrue("quartile $q ($from..$to) unrepresented: $pages", pages.any { it in from..to })
        }
    }
}
