package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.TableRecord
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Trust-boundary regressions for source-derived answers that bypass Gemma generation. */
class DeterministicAnswerSafetyTest {
    private fun citation(index: Int, text: String, kind: String = "PARAGRAPH") = Citation(
        "c$index", "doc", 1, index, 1.0, text,
        sectionId = "s", contentKind = kind, excerptId = "E1",
    )

    @Test
    fun `definition keeps decimal value and unit`() {
        val lead = AnswerQuestionUseCase.buildDefinitionLead(
            "What is a widget?",
            listOf(citation(1, "A widget is a component measuring 1.5 metres in length.")),
        )

        assertTrue(lead.decisive)
        assertTrue(lead.text, lead.text.contains("1.5 metres"))
    }

    @Test
    fun `numeric citation suffix is rejected rather than repaired`() {
        val filter = GroundingStreamFilter("Describe it", listOf(citation(1, "Supported text.")))

        val answer = filter.accept("Claim [E10]") + filter.finish()

        assertTrue(answer, filter.hadGroundingFailure)
        assertFalse(answer, answer.contains("[Page 1]"))
    }

    @Test
    fun `known missing list continuation prevents decisive partial answer`() {
        val intro = citation(1, "The widget setup steps are:")
        val first = citation(2, "1. Start the widget.", "LIST").copy(continuesToChunkIndex = 3)

        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "List all widget setup steps",
            listOf(intro, first),
        )

        assertFalse(lead.text, lead.decisive)
    }

    @Test
    fun `resolved table with missing rows does not use another table`() {
        val other = citation(2, "| Class | Limit |\n| D | 99 |", "TABLE").copy(
            tableId = "other", tableNumber = "5.2", tableCaption = "Table 5.2 Widget load limit",
        )

        val lead = AnswerQuestionUseCase.buildTableLead(
            "In Table 5.1 what is the widget load limit for class D?",
            listOf(other),
            resolvedTableId = "target",
        )

        assertFalse(lead.text, lead.decisive)
    }

    @Test
    fun `conditional answer must cover the requested property`() {
        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What is the inspection interval for Type B equipment?",
            listOf(citation(1, "Type B equipment must remain below 20 metres when operating.")),
        )

        assertFalse(lead.text, lead.decisive)
    }

    @Test
    fun `short conditional property question matches an inflected source term`() {
        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What is the slope for Type B?",
            listOf(citation(1, "Type B material less than 20 feet deep may be sloped at 1:1.")),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text.contains("1:1"))
    }

    @Test
    fun `short wrong property does not trigger conditional answer`() {
        val lead = AnswerQuestionUseCase.buildConditionalValueLead(
            "What is the inspection for Type B?",
            listOf(citation(1, "Type B material must remain below 20 metres when operating.")),
        )

        assertFalse(lead.text, lead.decisive)
    }

    @Test
    fun `table number does not resolve to a longer prefix`() {
        val wrong = TableRecord("wrong", "1.10", "Table 1.10 Limits", 1, 1, listOf("c1"))
        val manifest = DocumentStructureManifest("doc", "ns", 23, "sig", emptyList(), listOf(wrong))

        val plan = QueryPlanner().plan("In Table 1.1 what is the limit?", manifest)

        assertNull(plan.resolvedTableId)
    }
}
