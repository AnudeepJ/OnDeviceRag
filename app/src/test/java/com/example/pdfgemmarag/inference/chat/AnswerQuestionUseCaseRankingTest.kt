package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AnswerQuestionUseCaseRankingTest {
    @Test
    fun `adjacent expansion follows rank and query-relative score rather than absolute score`() {
        assertTrue(AnswerQuestionUseCase.shouldExpandAdjacentSeed(rank = 0, score = 0.82, bestScore = 0.82))
        assertTrue(AnswerQuestionUseCase.shouldExpandAdjacentSeed(rank = 5, score = 0.66, bestScore = 0.82))
        assertFalse(AnswerQuestionUseCase.shouldExpandAdjacentSeed(rank = 6, score = 0.65, bestScore = 0.82))
        assertFalse(AnswerQuestionUseCase.shouldExpandAdjacentSeed(rank = 4, score = 0.30, bestScore = 0.82))
    }

    @Test
    fun `same-page adjacent evidence crosses a heading boundary only with query support`() {
        val seed = citation(316, "7.2.1 EMPLOYEES").copy(
            sectionId = "employees",
            contentKind = "HEADING",
            pageNumber = 45,
        )
        val relevant = citation(315, "Safety and health training should preferably be not less than 48 hours").copy(
            sectionId = "training",
            pageNumber = 45,
        )
        val unrelated = citation(317, "Protective clothing must be kept clean").copy(
            sectionId = "ppe",
            pageNumber = 45,
        )
        val terms = setOf("training", "duration", "safety", "health")

        assertTrue(AnswerQuestionUseCase.isEligibleAdjacentNeighbor(seed, relevant, terms))
        assertFalse(AnswerQuestionUseCase.isEligibleAdjacentNeighbor(seed, unrelated, terms))
        assertFalse(
            AnswerQuestionUseCase.isEligibleAdjacentNeighbor(
                seed,
                relevant.copy(pageNumber = 44),
                terms,
            ),
        )
    }

    @Test
    fun `corrupt manifest fallback is restricted to facts with a matching trusted namespace`() {
        assertTrue(
            AnswerQuestionUseCase.canFallbackWithoutManifest(
                QuestionIntent.FACT, "doc", "doc:v21:published",
            ),
        )
        assertFalse(
            AnswerQuestionUseCase.canFallbackWithoutManifest(
                QuestionIntent.SECTION_SUMMARY, "doc", "doc:v21:published",
            ),
        )
        assertFalse(
            AnswerQuestionUseCase.canFallbackWithoutManifest(
                QuestionIntent.FACT, "doc", "another:v21:published",
            ),
        )
    }

    @Test
    fun `overview samples top-level sections across the document`() {
        val sections = (1..20).map { index ->
            com.example.pdfgemmarag.inference.store.SectionRecord(
                sectionId = "s$index",
                specificationNumber = index.toString().padStart(5, '0'),
                sectionNumber = "",
                title = "Topic $index",
                path = "Topic $index",
                startPage = index,
                endPage = index,
                orderedChunkIds = listOf("c$index-a", "c$index-b", "c$index-c"),
                tokenCount = 100,
                level = 1,
            )
        }
        val manifest = com.example.pdfgemmarag.inference.store.DocumentStructureManifest(
            "doc", "doc:v21:test", 21, "sig", sections,
        )

        val ids = AnswerQuestionUseCase.overviewChunkIds(manifest, maxSections = 5, chunksPerSection = 2)

        assertEquals(10, ids.size)
        assertTrue(ids.take(5).all { it.endsWith("-a") })
        assertTrue(ids.drop(5).all { it.endsWith("-b") })
        assertEquals("c1-a", ids.first())
        assertEquals("c20-b", ids.last())
    }

    @Test
    fun `overview lead exposes distinct cited document roots immediately`() {
        val roots = listOf(
            citation(1, "CAST-IN-PLACE CONCRETE").copy(
                specificationNumber = "03300", sectionId = "s1", sectionTitle = "CAST-IN-PLACE CONCRETE",
                contentKind = "HEADING", pageNumber = 13,
            ),
            citation(2, "STRUCTURAL CONCRETE").copy(
                specificationNumber = "03310", sectionId = "s2", sectionTitle = "STRUCTURAL CONCRETE",
                contentKind = "HEADING", pageNumber = 51,
            ),
        )

        val lead = AnswerQuestionUseCase.buildOverviewLead(roots)

        assertTrue(lead.text.contains("Specification 03300"))
        assertTrue(lead.text.contains("[Page 13]"))
        assertEquals(listOf("s1", "s2"), lead.citations.map { it.sectionId })
    }
    @Test
    fun `manifest scoped fallback ranks the requested facts first`() {
        val unrelated = citation(1, "Admixtures shall comply with the referenced standard")
        val target = citation(2, "Maximum allowable water cement ratios are 0.45, 0.40, and 0.55")

        val ranked = AnswerQuestionUseCase.rankDirectScope(
            "List the maximum water cement ratios in specification 03310",
            listOf(unrelated, target),
        )

        assertEquals(2, ranked.first().chunkIndex)
        assertEquals(true, ranked.first().score > ranked.last().score)
    }

    @Test
    fun `section lookup resolves a uniquely matching printed cross reference`() {
        val source = citation(4, "References: 1. Section 01440 - Inspection Services 2. Section 01450 – Testing Laboratory Services")

        val answer = AnswerQuestionUseCase.buildSectionPointerAnswer(
            "What section covers the testing laboratory services?",
            listOf(source),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("Section 01450"))
        assertTrue(answer.text.contains("Testing Laboratory Services"))
    }

    @Test
    fun `list question resolves adjacent enumerated value block without generation`() {
        val heading = citation(10, "1. Maximum allowable ratios shall be as follows:")
        val values = citation(11, "a. Liquid-containing: 0.45. b. Brackish water: 0.40. c. Other: 0.55. 2. Continue procedure.")

        val answer = AnswerQuestionUseCase.buildEnumeratedValueAnswer(
            "List the maximum allowable ratios",
            listOf(heading, values),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("0.45"))
        assertTrue(answer.text.contains("0.40"))
        assertTrue(answer.text.contains("0.55"))
        assertEquals(false, answer.text.contains("Continue procedure"))
    }

    @Test
    fun `numbered rule list keeps adjacent items associated with its introduction`() {
        val introduction = citation(
            3,
            "There are two golden rules for developing a safe and productive environment:",
        )
        val first = citation(4, "1) Correct unsafe conditions immediately.")
            .copy(contentKind = "LIST")
        val second = citation(5, "2) Leave a laboratory in better condition than when you found it.")
            .copy(contentKind = "LIST")

        val answer = AnswerQuestionUseCase.buildNumberedListLead(
            "What are the two golden rules for laboratory safety?",
            listOf(introduction, first, second),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("Correct unsafe conditions immediately"))
        assertTrue(answer.text.contains("Leave a laboratory in better condition"))
        assertEquals(listOf("c4", "c5"), answer.citations.map { it.chunkId })
    }

    @Test
    fun `numbered items in one compact list chunk remain individually answerable`() {
        val introduction = citation(3, "The two safety rules are:")
        val compactList = citation(
            4,
            "1) Correct unsafe conditions immediately. 2) Leave the laboratory in better condition.",
        ).copy(contentKind = "LIST")

        val answer = AnswerQuestionUseCase.buildNumberedListLead(
            "List the two safety rules.",
            listOf(introduction, compactList),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("1) Correct unsafe conditions immediately"))
        assertTrue(answer.text.contains("2) Leave the laboratory"))
        assertEquals(listOf("c4"), answer.citations.map { it.chunkId })
    }

    @Test
    fun `summary preserves a unique adjacent multi-value block`() {
        val heading = citation(10, "E. Maximum allowable ratios shall be as follows:")
        val values = citation(11, "a. First: 0.45. b. Second: 0.40. c. Other: 0.55. 2. Continue procedure.")

        val lead = AnswerQuestionUseCase.buildEnumeratedSummaryLead(listOf(heading, values))

        assertEquals(false, lead.decisive)
        assertTrue(lead.text.contains("0.45"))
        assertTrue(lead.text.contains("0.40"))
        assertTrue(lead.text.contains("0.55"))
        assertEquals(false, lead.text.contains("Continue procedure"))
    }

    @Test
    fun `enumerated ratios do not hijack a nearby slump question`() {
        val heading = citation(10, "E. Water-Cement Ratios: Maximum allowable water-cement ratios shall be as follows:")
        val values = citation(11, "a. Liquid-containing: 0.45. b. Brackish water: 0.40. c. Other: 0.55.")

        val answer = AnswerQuestionUseCase.buildEnumeratedValueAnswer(
            "What are the minimum and maximum slump values for Portland cement concrete and concrete dosed with superplasticizer",
            listOf(heading, values),
        )

        assertEquals(false, answer.decisive)
        assertTrue(answer.text.isEmpty())
    }

    @Test
    fun `tolerance question returns one uniquely matching table row`() {
        val table = citation(
            358,
            "| Grade | Top surfaces of curbs and railings | 3/16” in 10’ | | --- | --- | --- | " +
                "| Drawing Dimensions | Cross section of columns, caps, walls, beams, and similar members | ±1/2”, -1/4” | " +
                "|  | Thickness of deck slabs | ±1/4”, -1/8” |",
        ).copy(contentKind = "TABLE")

        val answer = AnswerQuestionUseCase.buildTableLead(
            "Got it. What about the cross-sectional dimensions? How much tolerance do we have on the column thickness?",
            listOf(
                table,
                table.copy(
                    chunkId = "duplicate",
                    pageNumber = 12,
                    score = 0.9,
                    text = table.text.replace("±", "+").replace("”", "\""),
                ),
            ),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("Cross section of columns"))
        assertTrue(answer.text.contains("+1/2\""))
        assertTrue(answer.text.contains("-1/4\""))
        assertEquals(false, answer.text.contains("deck slabs"))
    }

    @Test
    fun `variation question returns exact value from flattened schedule paragraph`() {
        val paragraph = citation(
            357,
            "Variation In Maximum From Plumb of Specified Surfaces of columns, piers and walls 1/2\" in 10’ Batter Level or Top surfaces of slabs",
        )

        val answer = AnswerQuestionUseCase.buildTableLead(
            "What is the maximum allowable variation from plumb for the new piers?",
            listOf(paragraph),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("1/2\" in 10’"))
        assertEquals(false, answer.text.contains("Top surfaces of slabs"))
    }

    @Test
    fun `curb tolerance is not hijacked by plumb paragraph through function words`() {
        val paragraph = citation(
            357,
            "Variation In Maximum From Plumb of Specified Surfaces of columns, piers and walls 1/2\" in 10’",
        )
        val table = citation(
            358,
            "| Grade | Top surfaces of curbs and railings | 3/16” in 10’ |",
        ).copy(contentKind = "TABLE")

        val answer = AnswerQuestionUseCase.buildTableLead(
            "Okay, and what is the tolerance for the level of the top surfaces of the curbs?",
            listOf(paragraph, table),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text.contains("3/16” in 10’"))
        assertEquals(false, answer.text.contains("From Plumb"))
    }

    @Test
    fun `flattened table keeps parent scope attached to repeated row label`() {
        val header = citation(
            239,
            "Minimum Surface Cover (in inches) Slabs and Joists - Top and bottom bars for dry conditions",
        )
        val flattened = citation(
            240,
            "No. 5 bars and smaller: 1 1/2 No. 6 through No. 18 bars: 2 " +
                "Beams and Columns - For dry conditions - Stirrups, spirals and ties: 1 1/2 " +
                "Principal reinforcement: 2 Exposed to earth, water, sewage or weather " +
                "Stirrups and ties: 2 Principal reinforcement: 2 1/2 Walls - For dry conditions",
        )

        val answer = AnswerQuestionUseCase.buildTableLead(
            "In Table 03210B, what minimum concrete cover is required for principal reinforcement " +
                "in beams and columns exposed to earth, water, sewage or weather?",
            listOf(header, flattened),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text, answer.text.contains("Principal reinforcement — 2 1/2 inches"))
        assertEquals(listOf("c240"), answer.citations.map { it.chunkId })
        assertEquals(false, answer.text.contains("No. 5 bars"))
    }

    @Test
    fun `flattened table declines equally matching sibling rows`() {
        val flattened = citation(
            240,
            "Beams - Principal reinforcement: 2 Columns - Principal reinforcement: 3",
        )

        val answer = AnswerQuestionUseCase.buildTableLead(
            "In the table, what is the principal reinforcement value?",
            listOf(flattened),
        )

        assertEquals(false, answer.decisive)
        assertTrue(answer.text.isEmpty())
    }

    @Test
    fun `flattened signed tolerance row is returned verbatim`() {
        val flattened = citation(
            233,
            "Uniform spacing of bars (but the required number of bars shall not be reduced): ±2 " +
                "Uniform spacing of stirrups and ties (but the required number of stirrups and ties shall not be reduced): ±1",
        )
        val nearbyCoverTable = citation(
            240,
            "Principal reinforcement: 2 Exposed to earth, water, sewage or weather Stirrups and ties: 2 " +
                "Principal reinforcement: 2 1/2",
        ).copy(pageNumber = 33)

        val answer = AnswerQuestionUseCase.buildTableLead(
            "What placement tolerance applies to uniform spacing of stirrups and ties for reinforcement?",
            listOf(nearbyCoverTable, flattened),
        )

        assertTrue(answer.decisive)
        assertTrue(answer.text, answer.text.contains("±1"))
        assertEquals(false, answer.text.contains("±2"))
    }

    @Test
    fun `flattened shortcut declines a multi row values request`() {
        val flattened = citation(
            435,
            "Portland Cement Concrete: 2 to 4 Concrete to be dosed with superplasticizer: 1 to 3",
        )

        val answer = AnswerQuestionUseCase.buildTableLead(
            "What are the minimum and maximum slump values for Portland cement concrete and " +
                "concrete dosed with superplasticizer?",
            listOf(flattened),
        )

        assertEquals(false, answer.decisive)
        assertTrue(answer.text.isEmpty())
    }

    private fun citation(index: Int, text: String) = Citation(
        chunkId = "c$index",
        docHash = "doc",
        pageNumber = 62,
        chunkIndex = index,
        score = 0.0,
        text = text,
        sectionPath = "SECTION 03310 pencils",
    )
}
