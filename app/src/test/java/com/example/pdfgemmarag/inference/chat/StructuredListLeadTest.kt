package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Complete-list answers across label styles, chunk boundaries and pages (roadmap M3). */
class StructuredListLeadTest {

    private fun chunk(
        index: Int, text: String, page: Int = 25, kind: String = "PARAGRAPH",
        continuesTo: Int = -1, continuesFrom: Int = -1,
        listId: String = "", itemStart: Int = 0, itemCount: Int = 0,
        totalItems: Int = 0, complete: Boolean = false,
    ) = Citation(
        "c$index", "doc", page, index, 1.0, text, indexNamespace = "doc:v22:x", sectionId = "s",
        contentKind = kind, continuesToChunkIndex = continuesTo, continuesFromChunkIndex = continuesFrom,
        listId = listId, listItemStart = itemStart, listItemCount = itemCount,
        listTotalItems = totalItems, listComplete = complete,
    )

    @Test
    fun `roman numbered list with a plain colon introduction returns every requested item`() {
        val intro = chunk(10, "There are some basic steps about how the risk assessment should be undertaken like:")
        val list = chunk(
            11,
            "I. Initiating the review\nII. Identify the hazard\nIII. Identify all parties affected by the hazard and determine how they can be affected\nIV. Evaluate or assess the risk\nV. Monitor & Review",
            kind = "LIST",
        )
        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "What are the five basic steps of the risk assessment?",
            listOf(list, intro),
        )
        assertTrue(lead.decisive)
        assertEquals(5, lead.text.lines().size)
        assertTrue(lead.text.contains("V. Monitor & Review [Page 25]"))
    }

    @Test
    fun `of subject selects the matching five item list when a nearby five step list competes`() {
        val hiraIntro = chunk(168, "There are some basic steps about how the risk assessment should be undertaken like:")
        val hiraList = chunk(
            169,
            "I. Initiating the HIRA II. Identify the hazard III. Identify all parties affected by the hazard and determine how they can be affected IV. Evaluate or assess the risk V. Monitor & Review",
            kind = "LIST", listId = "hira", itemCount = 5, totalItems = 5, complete = true,
        )
        val hierarchyIntro = chunk(179, "The Five Steps of Hierarchy to assess the hazard should include as followed:", page = 26)
        val hierarchyList = chunk(
            180,
            "A. Look for the hazards B. Decide who might be harmed, and how C. Evaluate the risks D. Record findings E. Review the assessment",
            page = 26, kind = "LIST", listId = "hierarchy", itemCount = 5, totalItems = 5, complete = true,
        )
        val approachIntro = chunk(170, "Step 1: Initiating the HIRA - Approaches to risk assessment are based upon:")
        val approachList = chunk(
            171,
            "1. Observation 2. Identification 3. Consideration 4. Review 5. External factors 6. Stress 7. Safeguards 8. PPE",
            kind = "LIST", listId = "approaches", itemCount = 8, totalItems = 8, complete = true,
        )

        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "What are the five basic steps of HIRA?",
            listOf(hierarchyIntro, hierarchyList, approachIntro, approachList, hiraIntro, hiraList),
        )

        assertTrue(lead.decisive)
        assertEquals(5, lead.text.lines().size)
        assertTrue(lead.text.contains("Initiating the HIRA"))
        assertFalse(lead.text.contains("Look for the hazards"))
    }

    @Test
    fun `structured list missing same-page tail is not decisive`() {
        val intro = chunk(10, "Before starting the work the following checks are required:")
        val first = chunk(
            11, "1. Inspect equipment", kind = "LIST", listId = "checks",
            itemStart = 0, itemCount = 1, totalItems = 6, complete = false,
        )

        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "Which checks are required before starting the work?", listOf(intro, first),
        )

        assertFalse(lead.decisive)
    }

    @Test
    fun `list continuing onto the next page is followed through the continuation edge`() {
        val intro = chunk(10, "Before starting the work the following checks are required:")
        val first = chunk(11, "\u2022 Locate all buried services\n\u2022 Identify overhead power lines", kind = "LIST", continuesTo = 12)
        val second = chunk(12, "\u2022 Test for hazardous gas before entering\n\u2022 Remove water from the excavation", page = 26, kind = "LIST", continuesFrom = 11)
        val unrelated = chunk(13, "\u2022 Unrelated later list item", page = 27, kind = "LIST")

        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "Which checks are required before starting the work?",
            listOf(intro, first, second, unrelated),
        )
        assertTrue(lead.decisive)
        assertEquals(4, lead.text.lines().size)
        assertTrue(lead.text.contains("[Page 26]"))
        assertFalse(lead.text.contains("Unrelated"))
    }

    @Test
    fun `fewer items than the requested count is not a decisive answer`() {
        val intro = chunk(10, "The basic steps are:")
        val list = chunk(11, "1. Only one step", kind = "LIST")
        val lead = AnswerQuestionUseCase.buildNumberedListLead("What are the five basic steps?", listOf(intro, list))
        assertFalse(lead.decisive)
    }

    @Test
    fun `a table reference is not mistaken for a requested item count`() {
        val intro = chunk(10, "The following classes are listed:")
        val list = chunk(11, "A. First class\nB. Second class\nC. Third class", kind = "LIST")
        val lead = AnswerQuestionUseCase.buildNumberedListLead("Which classes are listed in Table 1.1?", listOf(intro, list))
        assertTrue(lead.decisive)
        assertEquals(3, lead.text.lines().size)
    }

    @Test
    fun `implicit enumeration follows a prominent retrieved source into its adjacent list`() {
        val seed = chunk(9, "Emergency planning overview", page = 196).copy(chunkId = "seed", score = 2.0)
        val intro = chunk(10, "Emergencies that may need to be planned for include (but are not limited to):", page = 196)
            .copy(score = 0.2, retrievalProvenance = "ADJACENT", sourceChunkId = "seed")
        val list = chunk(
            11,
            "A. Serious injuries or Poisoning B. Fire or Explosion C. Chemical leak or spill D. Electrocution E. Structural collapse F. Flood or Earthquake",
            page = 196,
            kind = "LIST",
        ).copy(score = 0.2, retrievalProvenance = "ADJACENT", sourceChunkId = "seed")
        val distractorIntro = chunk(20, "A poorly managed construction site can lead to accidents from:", page = 23)
        val distractorList = chunk(21, "A. Falling tools B. Wrong equipment C. Collisions", page = 23, kind = "LIST")

        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "What emergencies must a construction-site emergency plan consider?",
            listOf(seed, intro, list, distractorIntro, distractorList),
        )

        assertTrue(lead.decisive)
        assertEquals(6, lead.text.lines().size)
        assertTrue(lead.text.contains("F. Flood or Earthquake [Page 196]"))
    }

    @Test
    fun `generic first aid list cannot answer a specific amputation procedure`() {
        val intro = chunk(10, "The following first aid arrangements shall be provided:")
        val list = chunk(11, "a. Provide Correct Training", kind = "LIST")

        val lead = AnswerQuestionUseCase.buildNumberedListLead(
            "What are the first aid procedures for handling an amputated body part?",
            listOf(intro, list),
        )

        assertFalse(lead.decisive)
    }

    @Test
    fun `numbered subprocedure copies all bullets until the next numbered subject`() {
        val first = chunk(
            20,
            "1. Abdominal injury • Call for help 2. Amputation • Apply pressure • Place the amputated part in a clean plastic bag",
            page = 185, kind = "LIST", complete = false,
        )
        val tail = chunk(
            21, "• Put the bag in a container with ice • Go to hospital", page = 185,
            kind = "LIST", complete = true,
        )
        val boundary = chunk(22, "3. Altitude sickness • Descend", page = 186, kind = "LIST")

        val lead = AnswerQuestionUseCase.buildScopedNumberedProcedureLead(
            "What are the first aid procedures for handling an amputated body part?",
            listOf(first, tail, boundary),
        )

        assertTrue(lead.decisive)
        assertEquals(4, lead.text.lines().size)
        assertTrue(lead.text.contains("clean plastic bag"))
        assertFalse(lead.text.contains("Altitude"))
    }

    @Test
    fun `control hierarchy selects the source I through V sequence`() {
        val first = chunk(
            30,
            "I. Elimination - remove it II. Substitution - replace it III. Engineering control: isolate it",
            page = 31, kind = "LIST",
        )
        val second = chunk(35, "IV. Administrative controls: procedures V. Personal Protective Equipment: PPE", page = 31, kind = "LIST")

        val lead = AnswerQuestionUseCase.buildHierarchyLead(
            "What is the priority order of the hierarchy of risk control measures?",
            listOf(first, second),
        )

        assertTrue(lead.decisive)
        assertEquals(5, lead.text.lines().size)
        assertTrue(lead.text.contains("V. Personal Protective Equipment"))
    }
}
