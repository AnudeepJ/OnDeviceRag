package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AnswerQuestionUseCaseRankingTest {
    @Test
    fun `named table category resolves a letter-labelled list row`() {
        val rowList = citation(
            59,
            "E. Rare Will only occur in exceptional circumstances D. Unlikely Not likely to occur within the foreseeable future C. Possible May occur",
        ).copy(contentKind = "LIST", pageNumber = 11)

        val lead = AnswerQuestionUseCase.buildLabelledListTableRowLead(
            "In Table 1.3, which probability category is D and how is it described?",
            listOf(rowList),
        )

        assertTrue(lead.decisive)
        assertEquals("Category D. Unlikely Not likely to occur within the foreseeable future [Page 11]", lead.text)
    }

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
        assertTrue(lead.decisive)
    }

    @Test
    fun `quantified scaffold questions prefer words in the returned sentence`() {
        val load = citation(
            529,
            "Scaffolds shall be designed to support at least 4 times the anticipated weight of men and material.",
        ).copy(pageNumber = 78)
        val dimensions = citation(
            531,
            "Minimum height of the first scaffold ledger shall be 2.2 meters. " +
                "The mid rail and top rail shall be at height 600 mm and 1200 mm respectively and toe boards 150mm shall be attached. " +
                "Wall scaffolding shall be secured every 10 meters and 8 meters. " +
                "Minimum overlap of vertical and horizontal members shall be 600 mm with two couplers.",
        ).copy(pageNumber = 78)

        val dimensionAnswer = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What are the tube-and-coupler scaffold mid rail, top rail and toe board dimensions?",
            listOf(dimensions, load),
        )
        val loadAnswer = AnswerQuestionUseCase.buildQuantifiedSentenceLead(
            "What minimum load must a scaffold be able to support?",
            listOf(dimensions, load),
        )

        assertTrue(dimensionAnswer.text, dimensionAnswer.text.contains("600 mm") && dimensionAnswer.text.contains("1200 mm") && dimensionAnswer.text.contains("150mm"))
        assertTrue(loadAnswer.text, loadAnswer.text.contains("4 times"))
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

    @Test
    fun `flattened min max table returns every requested row`() {
        val flattened = citation(
            420,
            "Concrete Type Minimum Slump Maximum Slump Portland Cement Concrete 2” 4” " +
                "Concrete to be dosed with superplasticizer: 1” 3” " +
                "Normal Weight Concrete after dosing with superplasticizer 4” 9”",
        )

        val answer = AnswerQuestionUseCase.buildTableLead(
            "What are the minimum and maximum slump values for Portland cement concrete and " +
                "concrete dosed with superplasticizer?",
            listOf(flattened),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text, answer.text.contains("Portland Cement Concrete: minimum 2”"))
        assertTrue(answer.text, answer.text.contains("Concrete to be dosed with superplasticizer: minimum 1”"))
        assertFalse(answer.text.contains("Normal Weight"))
    }

    @Test
    fun `printed table number scopes a caption and its local continuation`() {
        val candidates = listOf(
            citation(52, "Table (1.1) Risk Level Assessment"),
            citation(53, "Almost Certain Moderate 5 High 10 High 15 Catastrophic 20 Catastrophic 25"),
            citation(54, "Possible Low 3 Moderate 6 Moderate 9 High 12 High 15"),
            citation(80, "Unrelated requirement from another section").copy(sectionId = "other"),
        ).map { if (it.chunkIndex == 80) it else it.copy(sectionId = "risk") }

        val scoped = AnswerQuestionUseCase.scopeExplicitInlineTable(
            "In Table 1.1, what score results from Possible and Moderate?",
            candidates,
        )

        assertEquals(listOf(52, 53, 54), scoped.map { it.chunkIndex })
    }

    @Test
    fun `flattened risk matrix resolves row and consequence intersection`() {
        val headerAndFirst = citation(
            53,
            "Table 1.1 Insignificant Minor Moderate Major Catastrophic Almost Moderate (5) High (10) High (15) Catastrophic (20) Certain Catastrophic (25)",
        )
        val remaining = citation(
            54,
            "Likely Moderate (4) Moderate (8) High (12) Catastrophic (18) Catastrophic (20) " +
                "Possible Low (3) Moderate (6) Moderate (9) High (12) High (15)",
        )

        val certain = AnswerQuestionUseCase.buildTableLead(
            "In Table 1.1, what risk score results from Almost Certain likelihood and Catastrophic consequence?",
            listOf(headerAndFirst, remaining),
        )
        val possible = AnswerQuestionUseCase.buildTableLead(
            "Using Table 1.1, what is the assessed risk for Possible likelihood and Moderate consequence?",
            listOf(headerAndFirst, remaining),
        )

        assertTrue(certain.text, certain.decisive && certain.text.contains("25"))
        assertTrue(possible.text, possible.decisive && possible.text.contains("9"))
    }

    @Test
    fun `flattened risk matrix tolerates scores emitted after their descriptions`() {
        val malformed = citation(
            54,
            "(5) Likelihood Likely Moderate (4) Moderate High (12) Catastrophic Catastrophic " +
                "(4) (8) (18) (20) Possible Low (3) Moderate Moderate High (12) High (15) (3) (6) (9)",
        )

        val possible = AnswerQuestionUseCase.buildTableLead(
            "Using Table 1.1, what is the assessed risk for Possible likelihood and Moderate consequence?",
            listOf(malformed),
        )

        assertTrue(possible.text, possible.decisive)
        assertTrue(possible.text, possible.text.contains("Possible × Moderate: 9"))
    }

    @Test
    fun `risk response row includes an immediately following action list`() {
        val table = citation(
            62,
            "| | If an incident occurred | - Consider alternatives to doing the activity |\n" +
                "| Extreme | If an incident occurred | activity |\n" +
                "| | permanent injury or death | - Significant control measures will need to be implemented |",
        ).copy(contentKind = "TABLE", pageNumber = 12)
        val continuation = citation(63, "- Immediate action required by Management")
            .copy(contentKind = "LIST", pageNumber = 12)

        val answer = AnswerQuestionUseCase.buildTableLead(
            "What actions does Table 1.4 prescribe for an Extreme assessed risk?",
            listOf(table, continuation),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text.contains("alternatives"))
        assertTrue(answer.text.contains("Significant control measures"))
        assertTrue(answer.text.contains("Immediate action"))
    }

    @Test
    fun `boolean attribute row reverse lookup returns the option with yes`() {
        val header = citation(101, "Feature Safety Glasses Face Shield Splash Goggles")
            .copy(pageNumber = 19)
        val table = citation(
            102,
            "| Coverage area | Eyes | Eyes face nose and mouth | Eyes orbital bones |\n" +
                "| --- | --- | --- | --- |\n" +
                "| Seal around eyes | No | No | Yes |",
        ).copy(contentKind = "TABLE", pageNumber = 19)

        val answer = AnswerQuestionUseCase.buildTableLead(
            "Which eye-protection option in Table 1 seals around the eyes?",
            listOf(header, table),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text, answer.text.contains("Splash Goggles"))
        assertFalse(answer.text, answer.text.contains("Flying debris"))
    }

    @Test
    fun `typed table row matches when class and power are reordered in the question`() {
        val table = citation(
            177,
            "| Laser Class | Fire Hazard Distance | Skin Burn Distance | Flash Blindness Distance | Eye Hazard Distance |\n" +
                "| --- | --- | --- | --- | --- |\n" +
                "| Class 2 0.99 mW, 532 nm | 0.6 | 0.9 | 67 m | 14 m |",
        ).copy(contentKind = "TABLE", pageNumber = 34)

        val answer = AnswerQuestionUseCase.buildTableLead(
            "For the 0.99 mW Class 2 laser in Table 5, what are the flash-blindness and eye-hazard distances?",
            listOf(table),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text, answer.text.contains("67 m"))
        assertTrue(answer.text, answer.text.contains("14 m"))
    }

    @Test
    fun `typed first data row remains addressable when the header is missing`() {
        val table = citation(
            177,
            "| Class 2 0.99 mW, 532 nm | 0.6 | 0.9 | 67 m | 14 m |\n" +
                "| --- | --- | --- | --- | --- |\n" +
                "| Class 3R 4.99 mW, 532 nm | 1.4 | 2.1 | 149 m | 32 m |",
        ).copy(contentKind = "TABLE", pageNumber = 34)

        val answer = AnswerQuestionUseCase.buildTableLead(
            "For the 0.99 mW Class 2 laser in Table 5, what are the flash-blindness and eye-hazard distances?",
            listOf(table),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text, answer.text.contains("67 m"))
        assertTrue(answer.text, answer.text.contains("14 m"))
    }

    @Test
    fun `short typed first row remains addressable when the header is missing`() {
        val table = citation(
            174,
            "| 3B | Required | Required | Suggested | 5-500 mW | serious injury |\n" +
                "| --- | --- | --- | --- | --- | --- |\n" +
                "| 4 | Required | Required | Suggested | > 500 mW | fire hazard |",
        ).copy(contentKind = "TABLE", pageNumber = 33)

        val answer = AnswerQuestionUseCase.buildTableLead(
            "For a Class 3B laser, what energy range and procedural requirements appear in Table 4?",
            listOf(table),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text, answer.text.contains("5-500 mW"))
        assertTrue(answer.text, answer.text.contains("Required"))
    }

    @Test
    fun `paired never and always directive is preserved`() {
        val source = citation(
            124,
            "Do not mix incompatible chemicals. Never add water to acid. Always add acid to water.",
        ).copy(pageNumber = 24)

        val answer = AnswerQuestionUseCase.buildPairedDirectiveLead(
            "Should water be added to acid, or acid to water?",
            listOf(source),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text, answer.text.contains("Never add water to acid"))
        assertTrue(answer.text, answer.text.contains("Always add acid to water"))
    }

    @Test
    fun `flattened electrical fire row returns explicit class and methods`() {
        val row = citation(
            157,
            "Short circuit, hot electrical Powder type: Carbon dioxide Electrical components, lightning E discharge, etc.",
        ).copy(pageNumber = 31)

        val answer = AnswerQuestionUseCase.buildTableLead(
            "What class and extinguisher types does Table 3 specify for electrical fires?",
            listOf(row, citation(156, "| Flammable liquids | B | Foam spray |")),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text, answer.text.contains("Class E"))
        assertTrue(answer.text, answer.text.contains("Carbon dioxide"))
    }

    @Test
    fun `numbered severity row returns its full meaning`() {
        val rows = citation(
            57,
            "1. Insignificant No Treatment required 2. Minor Minor injury requiring First Aid Treatment " +
                "3. Moderate Injury requiring Medical Treatment or lost time " +
                "4. Major Serious injury requiring special Medical Treatment and/or Hospitalization " +
                "5. Severe Loss of life or permanent disability",
        )

        val answer = AnswerQuestionUseCase.buildTableLead(
            "According to Table 1.2, what does severity level 4 mean?",
            listOf(rows),
        )

        assertTrue(answer.text, answer.decisive)
        assertTrue(answer.text.contains("Serious injury"))
        assertTrue(answer.text.contains("Hospitalization"))
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
