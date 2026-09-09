package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.store.HybridQuery
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DefinitionLeadTest {
    private fun chunk(index: Int, text: String, page: Int = 102, kind: String = "PARAGRAPH") =
        Citation("c$index", "doc", page, index, 1.0, text, indexNamespace = "doc:v22:x", sectionId = "s", contentKind = kind)

    @Test
    fun `subject extraction covers the definition phrasings`() {
        assertEquals("trench", AnswerQuestionUseCase.definitionSubject("What is a trench?"))
        assertEquals("confined space", AnswerQuestionUseCase.definitionSubject("What makes a space a confined space?"))
        assertEquals("emergency action plan", AnswerQuestionUseCase.definitionSubject("What is an Emergency Action Plan?"))
        assertEquals("hazard", AnswerQuestionUseCase.definitionSubject("What does hazard mean?"))
        assertEquals("risk", AnswerQuestionUseCase.definitionSubject("Define risk."))
    }

    @Test
    fun `unique defining sentence is returned verbatim with its page`() {
        val intro = chunk(696, "Generally speaking, an excavation is a hole in the ground as the result of removing material. A trench is an excavation in which the depth exceeds (is bigger than) the width. Working in trenches is hazardous.")
        val other = chunk(707, "Have a means of exit provided from the inside of the trench.", page = 103)
        val lead = AnswerQuestionUseCase.buildDefinitionLead("What is a trench?", listOf(other, intro))
        assertTrue(lead.decisive)
        assertEquals("A trench is an excavation in which the depth exceeds (is bigger than) the width. [Page 102]", lead.text)
        assertEquals(listOf("c696"), lead.citations.map { it.chunkId })
    }

    @Test
    fun `definition that introduces a list carries the following criteria`() {
        val define = chunk(660, "12.1 INTRODUCTION Confined space means a space that: The term 'confined space' has any of following defining features as, any space;", page = 97)
        val items = chunk(661, "a. In which a person can bodily enter;\nb. Contains material that has the potential to engulf an entrant\nc. Contains or has a potential to contain a hazardous atmosphere", page = 97, kind = "LIST")
        val lead = AnswerQuestionUseCase.buildDefinitionLead("What makes a space a confined space?", listOf(define, items))
        assertTrue(lead.decisive)
        assertTrue(lead.text.startsWith("Confined space means a space that:"))
        assertTrue(lead.text.contains("bodily enter"))
        assertTrue(lead.text.contains("hazardous atmosphere"))
        assertEquals(listOf("c660", "c661"), lead.citations.map { it.chunkId })
    }

    @Test
    fun `competing definitions leave the answer to grounded generation`() {
        val a = chunk(1, "A hazard is a source of potential harm.")
        val b = chunk(2, "A hazard means any substance, object or situation that can cause an accident.", page = 8)
        val lead = AnswerQuestionUseCase.buildDefinitionLead("What is a hazard?", listOf(a, b))
        assertFalse(lead.decisive)
    }

    @Test
    fun `non definition question yields no subject`() {
        assertNull(AnswerQuestionUseCase.definitionSubject("What is the maximum distance allowed for a means of exit?"))
    }

    @Test
    fun `prefix stems and coverage rerank recover morphological variants`() {
        assertEquals("trench*", HybridQuery.prefixTerm("trenches"))
        assertEquals("requir*", HybridQuery.prefixTerm("required"))
        assertEquals("general*", HybridQuery.prefixTerm("generally"))
        assertEquals("depth*", HybridQuery.prefixTerm("depth"))
        assertNull(HybridQuery.prefixTerm("03300"))

        val terms = HybridQuery.keywordTerms("At what trench depth is a protective system generally required? between 21 and 25")
        assertTrue(terms.toString(), terms.containsAll(listOf("trench", "depth", "protective", "system", "generally", "required", "21", "25")))

        val good = "In general, trenches that are 1.2 metres deep or greater require a protective system." to 0.70
        val weak = "Designing a protective system requires consideration of many factors." to 0.76
        val reranked = HybridQuery.rerank(listOf(weak, good), terms.take(6), 0.35, { it.second }, { it.first })
        assertEquals(good.first, reranked.first().first.first)
    }
}

class StandardReferenceLeadTest {
    private fun chunk(index: Int, text: String, page: Int = 206, kind: String = "LIST") =
        Citation("c$index", "doc", page, index, 1.0, text, indexNamespace = "doc:v22:x", sectionId = "s", contentKind = kind)

    @Test
    fun `unique standard line naming the topic is copied verbatim`() {
        val list = chunk(1340, "9. IS 3696 (Part 1):1987 Safety code for scaffolds\n10. IS 3764:1992 Code of safety for excavation work\n11. IS 4081:1986 Safety code for blasting")
        val prose = chunk(1000, "Excavation work must follow the applicable standards and be supervised.", page = 102, kind = "PARAGRAPH")
        val lead = AnswerQuestionUseCase.buildStandardReferenceLead("Which Indian Standard covers the code of safety for excavation work?", listOf(prose, list))
        assertTrue(lead.decisive)
        assertEquals("10. IS 3764:1992 Code of safety for excavation work [Page 206]", lead.text)
    }

    @Test
    fun `question without a standard cue never triggers the lead`() {
        val list = chunk(1340, "10. IS 3764:1992 Code of safety for excavation work")
        assertFalse(AnswerQuestionUseCase.buildStandardReferenceLead("What must be checked before excavation work?", listOf(list)).decisive)
    }

    @Test
    fun `tied lines leave the choice to grounded generation`() {
        val list = chunk(1340, "IS 3696:1987 Safety code for scaffolds and ladders\nIS 4014:1967 Code for steel tubular scaffolding")
        assertFalse(AnswerQuestionUseCase.buildStandardReferenceLead("Which standard is the safety code for scaffolding?", listOf(list)).decisive)
    }
}

class AnchorRerankTest {
    @Test
    fun `acronym anchors outweigh two generic shared words`() {
        val question = "What are the five basic steps of HIRA?"
        val terms = HybridQuery.keywordTerms(question)
        val anchors = HybridQuery.anchorTerms(question)
        assertEquals(setOf("hira"), anchors)
        val hira = "There are some basic steps about how the risk assessment should be undertaken like: I. Initiating the HIRA II. Identify the hazard" to 0.80
        val investigation = "The five steps to be followed in the investigation are: What happened? How it happened?" to 0.80
        val filler = (1..6).map { "Unrelated chunk number $it about site safety." to 0.5 }
        val ranked = HybridQuery.rerank(listOf(investigation, hira) + filler, terms, 0.35, { it.second }, { it.first }, anchors)
        assertEquals(hira.first, ranked.first().first.first)
    }
}
