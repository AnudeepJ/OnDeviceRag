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
    fun `unique measured speed sentence preserves both conditioned limits`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What is the maximum speed for vehicles transporting explosives on rough roads and elsewhere?",
            listOf(citation(1, "The speed of the vehicle shall not exceed 25 km/h on rough roads and 40 km/h elsewhere.")),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text.contains("25 km/h"))
        assertTrue(lead.text.contains("40 km/h"))
    }

    @Test
    fun `unique measured mesh sentence preserves area and side limits`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What are the maximum allowable mesh opening dimensions and length for safety nets?",
            listOf(citation(1, "Mesh openings shall not exceed 36 square inches and shall not be longer than 6 inches on any side.")),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text.contains("36 square inches"))
        assertTrue(lead.text.contains("6 inches"))
    }

    @Test
    fun `tied measured sentences decline deterministic answer`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What is the maximum cable length?",
            listOf(
                citation(1, "The maximum cable length is 20 metres."),
                citation(2, "The maximum cable length is 30 metres."),
            ),
        )

        assertFalse(lead.text, lead.decisive)
    }

    @Test
    fun `multi class voltage comparison declines single sentence answer`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What are the voltage protection levels and differences between Class A, Class B, and Class C hard hats?",
            listOf(citation(1, "Class B provides high-voltage shock and burn protection up to 20,000 volts.")),
        )

        assertFalse(lead.text, lead.decisive)
    }

    @Test
    fun `mixed solids and color question declines numeric-only sentence`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What minimum solids content and color requirement apply to membrane forming curing compound?",
            listOf(citation(1, "Forms may be removed after 2 days and curing compound applied.")),
        )

        assertFalse(lead.text, lead.decisive)
    }

    @Test
    fun `general trench depth selects the explicit protective system threshold`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "At what trench depth is a protective system generally required?",
            listOf(
                citation(1, "In general, trenches that are 1.2 metres (4 feet) deep or greater require a protective system unless the excavation is made entirely in stable rock."),
                citation(2, "All simple slope excavations 20 feet (6Mt) or less in depth will have a maximum allowable slope of 3/4:1."),
            ),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text.contains("1.2 metres (4 feet)"))
    }

    @Test
    fun `coverage rate recognizes gallons per square feet over a nearby day value`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What minimum coverage rate applies when spraying curing compound on unformed surfaces?",
            listOf(
                citation(1, "Forms may be removed after 2 days and curing compound applied."),
                citation(
                    2,
                    "Unformed Surfaces: Cure by membrane curing compound method. Apply a uniform coating at the rate of coverage recommended by the manufacturer. Do not apply less than 1 gallon per 180 square feet of area.",
                ),
            ),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text.contains("1 gallon"))
        assertTrue(lead.text.contains("180 square feet"))
    }

    @Test
    fun `compressive strength percentage outranks nearby removal days`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What minimum compressive strength is required before removing formwork that supports the weight of concrete?",
            listOf(
                citation(1, "Do not remove forms supporting weight of concrete in less than 4 days."),
                citation(2, "The minimum concrete compressive strength for removal of formwork supporting weight of concrete is 75 percent of specified minimum 28-day strength."),
            ),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text.contains("75 percent"))
    }

    @Test
    fun `curing day question preserves temperature and duration`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What temperature and duration make a calendar day count as a curing day?",
            listOf(
                citation(
                    1,
                    "A curing day is any calendar day in which the temperature is above 50 degrees F for at least 19 hours. Colder days may be counted if air temperature is maintained above 50 degrees F.",
                ),
            ),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text.contains("50 degrees F"))
        assertTrue(lead.text.contains("19 hours"))
    }

    @Test
    fun `unitless water cement ratio beats nearby drying days`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What is the maximum water-cement ratio for concrete for liquid-containing structures?",
            listOf(
                citation(1, "Shrinkage tests are performed at 21 or 28 days of drying."),
                citation(2, "Maximum allowable water-cement ratios are as follows. Concrete for liquid-containing structures: 0.45."),
            ),
        )

        assertTrue(lead.text, lead.decisive)
        assertTrue(lead.text.contains("0.45"))
    }

    @Test
    fun `white pigmented condition declines numeric only solids sentence`() {
        val lead = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What minimum solids content applies, and when must it be white-pigmented?",
            listOf(citation(1, "Minimum solids content: 30 percent.")),
        )

        assertFalse(lead.text, lead.decisive)
    }

    @Test
    fun `table number does not resolve to a longer prefix`() {
        val wrong = TableRecord("wrong", "1.10", "Table 1.10 Limits", 1, 1, listOf("c1"))
        val manifest = DocumentStructureManifest(
            "doc", "ns", DocumentStructureManifest.INDEX_VERSION, "sig", emptyList(), listOf(wrong),
        )

        val plan = QueryPlanner().plan("In Table 1.1 what is the limit?", manifest)

        assertNull(plan.resolvedTableId)
    }
}
