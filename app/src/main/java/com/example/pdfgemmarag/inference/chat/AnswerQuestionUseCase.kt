package com.example.pdfgemmarag.inference.chat

import android.os.SystemClock
import android.util.Log
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.GenerationStats
import com.example.pdfgemmarag.core.model.QaPair
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import com.example.pdfgemmarag.inference.llm.ContextAssembler
import com.example.pdfgemmarag.inference.llm.ContextSelector
import com.example.pdfgemmarag.inference.llm.GemmaEngine
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.ManifestIntegrityError
import com.example.pdfgemmarag.inference.store.HybridQuery
import com.example.pdfgemmarag.inference.store.SectionRecord

/** Retrieve -> assemble -> generate for one question against one document. */
class AnswerQuestionUseCase(
    private val requireEmbedder: suspend () -> EmbeddingGemmaEmbedder,
    private val store: AppSearchVectorStore,
    private val engine: GemmaEngine,
    private val selector: ContextSelector = ContextSelector(),
    private val planner: QueryPlanner = QueryPlanner(),
    private val topK: Int = 12,
    private val similarityFloor: Double = 0.3,
    private val keywordWeight: Double = 0.05,
) {

    interface Listener {
        fun onRetrieved(citations: List<Citation>)
        fun onToken(text: String)
        fun onDone(stats: GenerationStats)
        fun onError(message: String)
    }

    /** Runs retrieval synchronously and starts generation; returns the handle used for cancellation. */
    suspend fun start(
        generationId: Long,
        docHash: String,
        activeIndexNamespace: String,
        question: String,
        history: List<QaPair>,
        listener: Listener,
    ): GemmaEngine.Generation? {
        val t0 = SystemClock.elapsedRealtime()
        val retrievalQuestion = rewriteForRetrieval(question, history)
        val manifestResult = runCatching { store.loadManifest(docHash) }
        val manifest = manifestResult.getOrNull()
        Log.i(
            TAG,
            "manifest hash=${docHash.take(12)} loaded=${manifest != null} sections=${manifest?.sections?.size ?: 0} " +
                "error=${manifestResult.exceptionOrNull()?.javaClass?.simpleName ?: "none"}",
        )
        val inherited = history.lastOrNull()?.sourceSectionId?.ifBlank { null }
        var plan = planner.plan(question, manifest, inherited)
        var queryVec: FloatArray? = null
        var manifestFallback = false
        var rawCandidates = emptyList<Citation>()
        var sourceCompleteness = EvidenceCompleteness.UNKNOWN

        if (manifestResult.exceptionOrNull() is ManifestIntegrityError) {
            if (!canFallbackWithoutManifest(plan.intent, docHash, activeIndexNamespace)) {
                return deterministic(generationId, t0, REPAIR_MESSAGE, listener)
            }
            manifestFallback = true
            Log.w(TAG, "manifest corrupt; using trusted namespace for unscoped FACT retrieval")
        }
        if (plan.intent == QuestionIntent.AMBIGUOUS_SECTION) {
            return deterministic(
                generationId, t0, plan.clarification(), listener,
                ambiguityCitations(manifest, plan),
            )
        }
        if (plan.intent == QuestionIntent.SECTION_SUMMARY && plan.resolvedSectionId == null && manifest != null) {
            queryVec = requireEmbedder().embedQuery(retrievalQuestion)
            plan = planner.plan(question, manifest, inherited, queryVec)
            if (plan.intent == QuestionIntent.AMBIGUOUS_SECTION) {
                return deterministic(
                    generationId, t0, plan.clarification(), listener,
                    ambiguityCitations(manifest, plan),
                )
            }
        }

        val implicitListIds = manifest?.let { implicitSubsectionListIds(plan, it) }.orEmpty()
        val ranked = when {
            plan.intent == QuestionIntent.FACT && implicitListIds.isNotEmpty() && manifest != null -> {
                try {
                    store.getChunks(manifest, implicitListIds).also {
                        rawCandidates = it
                        sourceCompleteness = EvidenceCompleteness.COMPLETE
                    }
                } catch (t: ManifestIntegrityError) {
                    Log.e(TAG, "implicit subsection list fetch failed", t)
                    return deterministic(generationId, t0, REPAIR_MESSAGE, listener)
                }
            }
            plan.intent == QuestionIntent.DOCUMENT_OVERVIEW && manifest != null -> {
                val ids = overviewChunkIds(manifest)
                if (ids.isEmpty()) return deterministic(generationId, t0, REPAIR_MESSAGE, listener)
                try {
                    store.getChunks(manifest, ids).also { rawCandidates = it }
                } catch (t: ManifestIntegrityError) {
                    Log.e(TAG, "document overview fetch failed", t)
                    return deterministic(generationId, t0, REPAIR_MESSAGE, listener)
                }
            }
            plan.intent == QuestionIntent.SECTION_SUMMARY && plan.resolvedSectionId != null && manifest != null -> {
                val section = manifest.sections.firstOrNull { it.sectionId == plan.resolvedSectionId }
                    ?: return deterministic(generationId, t0, REPAIR_MESSAGE, listener)
                // A chapter or part is summarised from its whole subtree in reading order; a
                // clause is summarised from its own chunks. The context budget bounds the prompt.
                val ids = if (plan.resolvedSubtree) {
                    manifest.boundedSubtreeChunkIds(section, MAX_SUBTREE_CHUNKS)
                } else section.orderedChunkIds
                Log.i(TAG, "section summary ${section.kind} '${section.title.take(40)}' subtree=${plan.resolvedSubtree} chunks=${ids.size}")
                try {
                    store.getChunks(manifest, ids).also {
                        rawCandidates = it
                        val allIds = if (plan.resolvedSubtree) {
                            manifest.descendantsOf(section).flatMap { node -> node.orderedChunkIds }.distinct()
                        } else section.orderedChunkIds
                        sourceCompleteness = if (ids.size == allIds.size) EvidenceCompleteness.COMPLETE
                        else EvidenceCompleteness.PARTIAL
                    }
                } catch (t: ManifestIntegrityError) {
                    Log.e(TAG, "section direct fetch failed", t)
                    return deterministic(generationId, t0, REPAIR_MESSAGE, listener)
                }
            }
            else -> {
                val vector = queryVec ?: requireEmbedder().embedQuery(retrievalQuestion)
                val terms = HybridQuery.keywordTerms(retrievalQuestion)
                Log.i(TAG, "ask hash=${docHash.take(12)} q='${question.take(120)}' terms=$terms dim=${vector.size}")
                var primary = store.search(
                    docHash, retrievalQuestion, vector, topK, similarityFloor, keywordWeight,
                    sectionId = plan.resolvedSectionId,
                    specificationNumber = plan.explicitSpecificationNumber,
                    indexNamespace = manifest?.indexNamespace ?: activeIndexNamespace.takeIf { manifestFallback },
                )
                if (primary.isEmpty() && plan.inheritedSectionId != null &&
                    plan.resolvedSectionId == plan.inheritedSectionId
                ) {
                    Log.i(TAG, "inherited section produced no hits; retrying standard unscoped search")
                    primary = store.search(
                        docHash, retrievalQuestion, vector, topK, similarityFloor, keywordWeight,
                        indexNamespace = manifest?.indexNamespace ?: activeIndexNamespace.takeIf { manifestFallback },
                    )
                }
                if (primary.isEmpty() && plan.explicitSpecificationNumber != null && manifest != null) {
                    // Some compatible AppSearch schema upgrades expose newly stored metadata but
                    // do not backfill its searchable posting list. The manifest is authoritative:
                    // direct-fetch only that specification, then rank its text locally. This also
                    // avoids silently searching a different specification when the index is stale.
                    val scopedIds = manifest.sections
                        .filter { it.specificationNumber.equals(plan.explicitSpecificationNumber, true) }
                        .flatMap { it.orderedChunkIds }
                    if (scopedIds.isNotEmpty()) {
                        primary = runCatching { store.getChunks(manifest, scopedIds) }
                            .map { rankDirectScope(retrievalQuestion, it).take(topK) }
                            .onFailure { Log.w(TAG, "manifest-scoped compatibility fetch failed: ${it.message}") }
                            .getOrDefault(emptyList())
                    }
                }
                rawCandidates = primary
                if (plan.resolvedTableId != null && manifest != null) {
                    primary = mergeResolvedTable(plan.resolvedTableId, manifest, primary)
                }
                if (plan.shape == AnswerShape.DEFINITION) {
                    // The sentence that defines the subject outranks later usages of the term, so
                    // its neighbours (the criteria list that usually follows) are expanded too.
                    primary = promoteDefiningSentences(question, primary)
                }
                expandStructuralNeighbors(primary, manifest, retrievalQuestion)
            }
        }
        // An explicit table identity is a hard evidence boundary. If its direct fetch failed, an
        // unrelated hybrid hit must not answer in its place.
        val inlineTableScope = scopeExplicitInlineTable(question, ranked)
        val answerEvidence = plan.resolvedTableId?.let { tableId ->
            val canonicalSource = if (inlineTableScope.size < ranked.size) {
                inlineTableScope
            } else {
                ranked.filter { it.tableId == tableId }
            }
            canonicalTableEvidence(manifest, tableId, canonicalSource)
        } ?: inlineTableScope
        val trace = EvidenceTrace(rawCandidates.ifEmpty { ranked }, answerEvidence, sourceCompleteness)
        if (plan.intent == QuestionIntent.DOCUMENT_OVERVIEW) {
            val overview = buildOverviewLead(answerEvidence)
            if (overview.decisive) {
                return deterministic(
                    generationId, t0, overview.text, listener, overview.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "OVERVIEW_LEAD", plan.intent,
                    trace,
                )
            }
        }
        if (plan.intent == QuestionIntent.FACT && plan.shape == AnswerShape.DEFINITION && manifest != null) {
            val definition = buildDefinitionLead(question, answerEvidence)
            Log.i(TAG, "definition lead subject='${definitionSubject(question)}' decisive=${definition.decisive} candidates=${answerEvidence.size}")
            if (definition.decisive) {
                // A definition that introduces its criteria with a colon needs the following list
                // chunk even when retrieval did not surface it.
                val complete = completeDefinition(definition, answerEvidence, manifest, question)
                return deterministic(
                    generationId, t0, complete.text, listener, complete.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "DEFINITION_LEAD", plan.intent,
                    trace,
                )
            }
        }
        if (plan.intent == QuestionIntent.FACT) {
            val standard = buildStandardReferenceLead(question, answerEvidence)
            if (standard.decisive) {
                return deterministic(
                    generationId, t0, standard.text, listener, standard.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "STANDARD_REFERENCE_LEAD", plan.intent,
                    trace,
                )
            }
            val pairedDirective = buildPairedDirectiveLead(question, answerEvidence)
            if (pairedDirective.decisive) {
                return deterministic(
                    generationId, t0, pairedDirective.text, listener, pairedDirective.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "PAIRED_DIRECTIVE_LEAD", plan.intent,
                    trace,
                )
            }
            val scopedProcedure = buildScopedNumberedProcedureLead(question, answerEvidence)
            if (scopedProcedure.decisive) {
                return deterministic(
                    generationId, t0, scopedProcedure.text, listener, scopedProcedure.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "SCOPED_PROCEDURE_LEAD", plan.intent,
                    trace.copy(completeness = EvidenceCompleteness.COMPLETE),
                )
            }
            val hierarchy = buildHierarchyLead(question, answerEvidence)
            if (hierarchy.decisive) {
                return deterministic(
                    generationId, t0, hierarchy.text, listener, hierarchy.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "HIERARCHY_LEAD", plan.intent,
                    trace.copy(completeness = EvidenceCompleteness.COMPLETE),
                )
            }
            val numberedList = buildNumberedListLead(question, answerEvidence)
            if (numberedList.decisive) {
                return deterministic(
                    generationId, t0, numberedList.text, listener, numberedList.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "LIST_LEAD", plan.intent,
                    trace.copy(completeness = EvidenceCompleteness.COMPLETE),
                )
            }
            val enumerated = buildEnumeratedValueAnswer(question, answerEvidence)
            if (enumerated.text.isNotEmpty()) {
                return deterministic(
                    generationId, t0, enumerated.text, listener, enumerated.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "ENUMERATED_VALUES_LEAD", plan.intent,
                    trace,
                )
            }
            // Navigation is answered structurally only when the user asked where something is;
            // a definition or requirement question must never be turned into a section pointer.
            if (plan.shape == AnswerShape.NAVIGATION) {
                val pointer = buildSectionPointerAnswer(question, answerEvidence)
                if (pointer.text.isNotEmpty()) {
                    return deterministic(
                        generationId, t0, pointer.text, listener, pointer.citations,
                        plan.resolvedSectionId.orEmpty(), manifestFallback, "SECTION_POINTER_LEAD", plan.intent,
                        trace,
                    )
                }
            }
            val exactTableRow = buildTableLead(question, answerEvidence, plan.resolvedTableId)
            if (exactTableRow.decisive) {
                return deterministic(
                    generationId, t0, exactTableRow.text, listener, exactTableRow.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "TABLE_ROW_LEAD", plan.intent,
                    trace,
                )
            }
            val conditional = buildConditionalValueLead(question, answerEvidence)
            if (conditional.decisive) {
                return deterministic(
                    generationId, t0, conditional.text, listener, conditional.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "CONDITIONAL_VALUE_LEAD", plan.intent,
                    trace,
                )
            }
            val quantified = buildQuantifiedSentenceLead(question, answerEvidence)
            if (quantified.decisive) {
                return deterministic(
                    generationId, t0, quantified.text, listener, quantified.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "QUANTIFIED_SENTENCE_LEAD", plan.intent,
                    trace,
                )
            }
        }
        // A follow-up ("What about Type C?") is only meaningful with the previous question; the
        // model receives the same combined text that retrieval used.
        val promptQuestion = if (retrievalQuestion != question) retrievalQuestion else question
        val selected = selector.select(promptQuestion, answerEvidence, plan.intent, plan.shape)
        Log.i(TAG, "planned ${plan.intent} retrieved ${answerEvidence.size} -> ${selected.excerpts.size} chunks (~${selected.approxTokens} tokens) in ${SystemClock.elapsedRealtime() - t0} ms")
        selected.excerpts.forEachIndexed { i, c ->
            Log.i(TAG, "  cite[$i] p${c.pageNumber} c${c.chunkIndex} score=${"%.3f".format(c.score)} '${c.text.take(80).replace('\n', ' ')}'")
        }

        if (selected.excerpts.isEmpty()) {
            Log.w(TAG, "EMPTY retrieval — returning canned refusal (index empty or query matched nothing)")
            val msg = "The document does not appear to contain information about that."
            listener.onToken(msg)
            listener.onDone(
                GenerationStats(
                    generationId, 0, SystemClock.elapsedRealtime() - t0, msg.length, 0.0,
                    0, 0, engine.backendName, false,
                    sourceSectionId = plan.resolvedSectionId.orEmpty(),
                    manifestFallback = manifestFallback,
                    rawCandidateChunks = trace.raw.size,
                    rawCandidateChunkIds = trace.raw.joinToString(",") { it.chunkId },
                    rawCandidatePages = trace.raw.map { it.pageNumber }.distinct().joinToString(","),
                    candidateChunks = trace.expanded.size,
                    candidateChunkIds = trace.expanded.joinToString(",") { it.chunkId },
                    candidateProvenance = trace.expanded.joinToString(",") {
                        "${it.chunkId}:${it.retrievalProvenance}:${it.sourceChunkId}"
                    },
                    evidenceCompleteness = trace.completeness.name,
                ),
            )
            return null
        }

        var firstToken = -1L
        var firstVisibleToken = -1L
        var chars = 0
        val evidenceLead = when (plan.intent) {
            QuestionIntent.FACT -> buildTableLead(question, answerEvidence, plan.resolvedTableId)
            QuestionIntent.SECTION_SUMMARY -> buildEnumeratedSummaryLead(answerEvidence)
            QuestionIntent.DOCUMENT_OVERVIEW -> buildOverviewLead(answerEvidence)
            QuestionIntent.AMBIGUOUS_SECTION -> SummaryLead.EMPTY
        }
        if (evidenceLead.decisive) {
            return deterministic(
                generationId, t0, evidenceLead.text, listener, evidenceLead.citations,
                plan.resolvedSectionId.orEmpty(), manifestFallback,
                trace = trace,
            )
        }
        if (evidenceLead.text.isNotEmpty()) {
            firstVisibleToken = SystemClock.elapsedRealtime()
            chars += evidenceLead.text.length + 2
            listener.onToken(evidenceLead.text + "\n\n")
        }
        val validatedPointers = buildList {
            plan.explicitSpecificationNumber?.takeIf { value ->
                manifest?.sections?.any { it.specificationNumber.equals(value, true) } == true
            }?.let(::add)
            plan.explicitSectionNumber?.takeIf { value ->
                manifest?.sections?.any { it.sectionNumber.equals(value, true) } == true
            }?.let(::add)
        }
        val filter = GroundingStreamFilter(question, selected.excerpts, validatedPointers)
        // History is used above to rewrite follow-up retrieval. Do not replay model answers into
        // generation: an incorrect answer would become apparent source material on the next turn.
        return engine.generate(
            systemInstruction = GemmaEngine.SYSTEM_INSTRUCTION,
            history = emptyList(),
            userMessage = selected.prompt,
            // A 75-word summary can exceed 160 model tokens once citations and Markdown are
            // included. Keep enough headroom to finish the last sentence instead of displaying a
            // syntactically valid but visibly truncated answer.
            maxOutputTokens = AnswerPolicy.maxOutputTokens(plan.intent, question),
            sink = object : GemmaEngine.TokenSink {
                override fun onToken(text: String) {
                    if (firstToken < 0) firstToken = SystemClock.elapsedRealtime()
                    val safe = filter.accept(text)
                    if (safe.isNotEmpty()) {
                        if (firstVisibleToken < 0) firstVisibleToken = SystemClock.elapsedRealtime()
                        chars += safe.length
                        listener.onToken(safe)
                    }
                }

                override fun onDone(cancelled: Boolean, benchmark: GemmaEngine.TurnBenchmark?) {
                    val end = SystemClock.elapsedRealtime()
                    val tail = filter.finish()
                    if (tail.isNotEmpty()) {
                        if (firstVisibleToken < 0) firstVisibleToken = SystemClock.elapsedRealtime()
                        chars += tail.length
                        listener.onToken(tail)
                    }
                    // Only cited excerpts cross Binder; text is fetched lazily with its explicit namespace.
                    listener.onRetrieved(
                        (evidenceLead.citations + filter.usedCitations)
                            .distinctBy { it.indexNamespace to it.chunkId }
                            .map { it.copy(text = "") },
                    )
                    if (filter.hadGroundingFailure) {
                        Log.w(TAG, "grounding rejected: ${filter.groundingReasons.joinToString("; ")}")
                    }
                    val genMs = if (firstToken > 0) (end - firstToken).coerceAtLeast(1) else 1
                    listener.onDone(
                        GenerationStats(
                            generationId = generationId,
                            timeToFirstTokenMs = if (firstToken > 0) firstToken - t0 else -1,
                            totalMs = end - t0,
                            outputChars = chars,
                            approxTokensPerSecond = benchmark?.decodeTokensPerSecond?.takeIf { it > 0 }
                                ?: (ContextAssembler.estimateTokens("x".repeat(chars)) * 1000.0 / genMs),
                            retrievedChunks = selected.excerpts.size,
                            contextTokensApprox = selected.approxTokens,
                            backend = engine.backendName,
                            cancelled = cancelled,
                            visibleTimeToFirstTokenMs = if (firstVisibleToken > 0) firstVisibleToken - t0 else -1,
                            groundingFailure = filter.hadGroundingFailure,
                            sourceSectionId = plan.resolvedSectionId.orEmpty(),
                            manifestFallback = manifestFallback,
                            groundingReasons = filter.groundingReasons.joinToString(";"),
                            prefillTokensPerSecond = benchmark?.prefillTokensPerSecond ?: 0.0,
                            decodeTokensPerSecond = benchmark?.decodeTokensPerSecond ?: 0.0,
                            prefillTokens = benchmark?.prefillTokens ?: 0,
                            intent = plan.intent.name,
                            answeredBy = "",
                            candidateChunks = answerEvidence.size,
                            candidateChunkIds = answerEvidence.joinToString(",") { it.chunkId },
                            candidateProvenance = answerEvidence.joinToString(",") {
                                "${it.chunkId}:${it.retrievalProvenance}:${it.sourceChunkId}"
                            },
                            selectedChunkIds = selected.excerpts.joinToString(",") { it.chunkId },
                            selectedPages = selected.excerpts.map { it.pageNumber }.distinct().joinToString(","),
                            answerCitationChunkIds = (evidenceLead.citations + filter.usedCitations)
                                .distinctBy { it.indexNamespace to it.chunkId }
                                .joinToString(",") { it.chunkId },
                            evidenceComplete = trace.completeness == EvidenceCompleteness.COMPLETE,
                            rawCandidateChunks = trace.raw.size,
                            rawCandidateChunkIds = trace.raw.joinToString(",") { it.chunkId },
                            rawCandidatePages = trace.raw.map { it.pageNumber }.distinct().joinToString(","),
                            evidenceCompleteness = trace.completeness.name,
                            selectionComplete = selected.completeCoverage,
                        ),
                    )
                }

                override fun onError(t: Throwable) {
                    Log.e(TAG, "generation failed", t)
                    val tail = filter.finish()
                    if (tail.isNotEmpty()) listener.onToken(tail)
                    listener.onError(t.message ?: t.javaClass.simpleName)
                }
            },
        )
    }

    companion object {
        private const val TAG = "AnswerQuestionUseCase"
        private val FOLLOW_UP_WORDS = listOf("it", "that", "this", "they", "those", "these", "there", "same")
        private val FOLLOW_UP_PREFIXES = listOf("and ", "what about ", "how about ")
        private val DECIMAL = Regex("(?<![\\d.])\\d+\\.\\d+(?!\\d)")
        private const val REPAIR_MESSAGE =
            "The document index needs repair before I can safely answer. Please re-index the document."
        private const val MAX_STRUCTURAL_NEIGHBOR_DISTANCE = 3
        private const val MAX_ADJACENT_PRIMARY_SEEDS = 6
        private const val ADJACENT_SEED_RELATIVE_FLOOR = 0.60
        private const val ADJACENT_SCORE_FACTOR = 0.85
        private const val MIN_CROSS_SECTION_NEIGHBOR_TERMS = 2
        private val LIMIT_OR_DURATION_QUERY = Regex(
            "(?i)\\b(?:minimum|maximum|duration|not\\s+less\\s+than|not\\s+more\\s+than)\\b",
        )
        private const val MAX_SUBTREE_CHUNKS = 80
        private const val TABLE_FETCH_BOOST = 2.0

        internal fun canFallbackWithoutManifest(
            intent: QuestionIntent,
            docHash: String,
            activeIndexNamespace: String,
        ): Boolean = intent == QuestionIntent.FACT && activeIndexNamespace.isNotBlank() &&
            (activeIndexNamespace == docHash || activeIndexNamespace.startsWith("$docHash:"))

        internal fun isEligibleAdjacentNeighbor(
            seed: Citation,
            neighbor: Citation,
            queryTerms: Set<String>,
        ): Boolean {
            if (neighbor.pageNumber != seed.pageNumber) return false
            if (neighbor.sectionId == seed.sectionId) return true
            val searchable = neighbor.sectionPath + " " + neighbor.text
            val matches = queryTerms.count { term ->
                Regex("(?<![\\p{L}\\p{N}])${Regex.escape(term)}(?![\\p{L}\\p{N}])", RegexOption.IGNORE_CASE)
                    .containsMatchIn(searchable)
            }
            return matches >= MIN_CROSS_SECTION_NEIGHBOR_TERMS
        }

        internal fun shouldExpandAdjacentSeed(rank: Int, score: Double, bestScore: Double): Boolean =
            rank < MAX_ADJACENT_PRIMARY_SEEDS &&
                (bestScore <= 0.0 || score >= bestScore * ADJACENT_SEED_RELATIVE_FLOOR)

        /** Selects a small, document-wide structural sample without vector search. */
        internal fun overviewChunkIds(
            manifest: DocumentStructureManifest,
            maxSections: Int = 8,
            chunksPerSection: Int = 4,
        ): List<String> {
            val sections = manifest.sections.filter { it.orderedChunkIds.isNotEmpty() && it.title.isNotBlank() }
            if (sections.isEmpty()) {
                return pageStratifiedChunkIds(manifest, maxSections * chunksPerSection)
            }
            val specificationGroups = sections
                .filter { it.specificationNumber.isNotBlank() }
                .groupBy { it.specificationNumber }
                .values
                .sortedBy { group -> group.minOf { it.startPage } }
            if (specificationGroups.size >= 2) {
                val perSpecification = evenlySpaced(specificationGroups, maxSections)
                    .map { group ->
                        group.sortedWith(compareBy<SectionRecord> { it.level }.thenBy { it.startPage })
                            .flatMap { it.orderedChunkIds }
                            .distinct()
                            .take(chunksPerSection)
                    }
                // Round-robin keeps every specification represented if the context budget fills
                // before all secondary excerpts fit.
                return (0 until chunksPerSection)
                    .flatMap { offset -> perSpecification.mapNotNull { it.getOrNull(offset) } }
                    .distinct()
            }
            val health = manifest.health()
            var representatives = manifest.topLevelSections().filter { it.orderedChunkIds.isNotEmpty() }
            // A document with one root heading still needs breadth: include its immediate children.
            if (representatives.size < 3) {
                val topLevel = representatives.map { it.level }.filter { it > 0 }.minOrNull()
                    ?: sections.map { it.level }.filter { it > 0 }.minOrNull()
                val childLevel = sections.map { it.level }.filter { it > (topLevel ?: 0) }.minOrNull()
                if (childLevel != null) representatives = (representatives + sections.filter { it.level == childLevel }).distinct()
            }
            // Coverage, not rank: reserve slots per document quartile so a long final appendix
            // cannot monopolise the sample. A degraded outline falls back to page-stratified
            // sampling over every titled section.
            if (health.degraded && manifest.pages.isNotEmpty()) {
                return pageStratifiedChunkIds(manifest, maxSections * chunksPerSection)
            }
            val pool = if (health.degraded && representatives.size < maxSections) sections else representatives
            val sampled = quartileSample(pool.sortedBy { it.startPage }, manifest.lastPage, maxSections)
            return sampled.flatMap { it.orderedChunkIds.take(chunksPerSection) }.distinct()
        }

        /** Genuine degraded fallback over source pages, including pages with no recognised heading. */
        internal fun pageStratifiedChunkIds(manifest: DocumentStructureManifest, limit: Int): List<String> {
            val populated = manifest.pages.filter { it.orderedChunkIds.isNotEmpty() }
            if (populated.isEmpty() || limit <= 0) return emptyList()
            val pageSlots = minOf(populated.size, limit)
            val sampledPages = evenlySpaced(populated, pageSlots)
            val firstPass = sampledPages.mapNotNull { it.orderedChunkIds.firstOrNull() }
            if (firstPass.size >= limit) return firstPass.take(limit)
            val remaining = sampledPages.flatMap { it.orderedChunkIds.drop(1) } +
                populated.filter { it !in sampledPages }.flatMap { it.orderedChunkIds }
            return (firstPass + remaining).distinct().take(limit)
        }

        /** Even slots per quartile of the page range, then evenly spaced within each quartile. */
        internal fun quartileSample(sorted: List<SectionRecord>, lastPage: Int, limit: Int): List<SectionRecord> {
            if (sorted.size <= limit) return sorted
            val perQuartile = (limit + 3) / 4
            val last = lastPage.coerceAtLeast(1)
            val out = LinkedHashSet<SectionRecord>()
            for (quartile in 0 until 4) {
                val from = 1 + quartile * last / 4
                val to = if (quartile == 3) last else (quartile + 1) * last / 4
                val inRange = sorted.filter { it.startPage in from..to }
                out += evenlySpaced(inRange, perQuartile)
            }
            // Fill any unused slots from the remaining sections in reading order.
            if (out.size < limit) {
                out += evenlySpaced(sorted.filter { it !in out }, limit - out.size)
            }
            return out.sortedBy { it.startPage }.take(limit)
        }

        private fun <T> evenlySpaced(values: List<T>, limit: Int): List<T> {
            if (values.size <= limit || limit <= 0) return values.take(limit.coerceAtLeast(0))
            if (limit == 1) return listOf(values.first())
            return (0 until limit).map { slot ->
                values[(slot * (values.lastIndex).toDouble() / (limit - 1)).toInt()]
            }.distinct()
        }

        internal data class SummaryLead(
            val text: String,
            val citations: List<Citation>,
            val decisive: Boolean = false,
        ) {
            companion object { val EMPTY = SummaryLead("", emptyList()) }
        }

        /** Immediately visible, source-derived outline while the compact model adds key details. */
        internal fun buildOverviewLead(candidates: List<Citation>): SummaryLead {
            val roots = candidates
                .filter { it.contentKind == "HEADING" }
                .distinctBy { it.specificationNumber.ifBlank { it.sectionId } }
                .take(8)
            if (roots.isEmpty()) return SummaryLead.EMPTY
            val text = buildString {
                append("Document scope:\n")
                roots.forEach { citation ->
                    // Printed identity comes from the heading itself: "Specification 03300",
                    // "CHAPTER 13", "APPENDIX A". Absent identifiers are simply omitted.
                    val identity = citation.specificationNumber
                        .takeIf(String::isNotBlank)
                        ?.let { "Specification $it — " }
                        ?: citation.sectionNumber
                            .takeIf { it.isNotBlank() && !it.equals("null", true) && it.any(Char::isLetter) }
                            ?.let { "$it — " }
                            .orEmpty()
                    val title = citation.sectionTitle.ifBlank {
                        citation.text.lineSequence().firstOrNull().orEmpty().trim()
                    }.takeUnless { it.equals("null", true) }.orEmpty()
                    append("- ").append(identity).append(title.ifBlank { "Untitled section" })
                        .append(" [Page ").append(citation.pageNumber).append("]\n")
                }
            }.trimEnd()
            return SummaryLead(text, roots, decisive = true)
        }

        /** Exact table rows plus their structural neighbours, ranked by query words and numbers. */
        internal fun buildTableLead(
            question: String,
            candidates: List<Citation>,
            resolvedTableId: String? = null,
        ): SummaryLead {
            val scoped = if (resolvedTableId.isNullOrBlank()) {
                candidates
            } else if (candidates.any { it.retrievalProvenance == "CANONICAL_CELL" }) {
                // canonicalTableEvidence already applied the exact printed-table boundary and
                // retained nearby source fragments that can repair incomplete geometric cells.
                candidates
            } else {
                candidates.filter { it.tableId == resolvedTableId }
            }
            val query = normalizeForEvidenceMatch(question)
            val toleranceAsk = TOLERANCE_QUERY_HINT.containsMatchIn(query)
            // Lists of ratios and numbered requirements are structured too; that alone must not
            // cause an unrelated nearby table to be streamed as a supposedly relevant answer.
            if (!TABLE_QUERY_HINT.containsMatchIn(query) && !toleranceAsk) return SummaryLead.EMPTY
            if (toleranceAsk) {
                val queryEvidence = evidenceWords(query)
                val matches = scoped.flatMap { citation ->
                    TOLERANCE_PARAGRAPH.findAll(citation.text).map { match ->
                        val label = match.groupValues[1]
                            .replace(Regex("\\s+"), " ")
                            .trim()
                            .replace(LEADING_REPEATED_WORD, "\$1")
                        val value = match.groupValues[2].trim()
                        Triple(citation, listOf(label, value), evidenceWords(label).count(queryEvidence::contains))
                    }.toList()
                }.filter { it.third >= 2 }
                val groups = matches.groupBy { (_, cells, _) -> canonicalTableRow(cells) }
                    .values.sortedByDescending { group -> group.maxOf { it.third } }
                val bestGroup = groups.firstOrNull()
                val bestScore = bestGroup?.maxOfOrNull { it.third }
                val best = bestGroup?.maxByOrNull { it.first.score }
                if (best != null && groups.drop(1).none { group -> group.maxOf { it.third } == bestScore }) {
                    return SummaryLead(
                        "${best.second.joinToString(" — ")} [Page ${best.first.pageNumber}]",
                        listOf(best.first),
                        decisive = true,
                    )
                }
            }
            // Prefer an intact grid whenever extraction preserved one. The flattened fallback
            // deliberately uses looser parsing and can mistake header scale values for row cells.
            val intactTables = scoped.filter { it.contentKind == "TABLE" || '|' in it.text }
            val intactBooleanAttribute = buildBooleanAttributeLead(query, scoped)
            if (intactBooleanAttribute.decisive) return intactBooleanAttribute
            val intactTypedRow = buildHeaderlessTypedRowLead(query, intactTables)
            if (intactTypedRow.decisive) return intactTypedRow
            val intactMatrixCell = buildMatrixCellLead(query, intactTables)
            if (intactMatrixCell.decisive) return intactMatrixCell
            val flatRiskMatrix = buildFlatRiskMatrixLead(query, scoped)
            if (flatRiskMatrix.decisive) return flatRiskMatrix
            val numberedRow = buildNumberedInlineRowLead(query, scoped)
            if (numberedRow.decisive) return numberedRow
            val labelledListRow = buildLabelledListTableRowLead(query, scoped)
            if (labelledListRow.decisive) return labelledListRow
            val inlineMinMax = buildInlineMinMaxRowsLead(query, scoped)
            if (inlineMinMax.decisive) return inlineMinMax
            // Apryse can preserve the reading order of a visually ruled table while flattening
            // its cells into a paragraph. Resolve only a high-confidence row: the row label must
            // occur in the question, and the preceding scope window must beat every competing
            // label/value path by a safe margin. This keeps parent scopes such as structure type
            // and exposure condition attached to repeated child labels.
            val flattened = buildFlattenedTableLead(query, scoped)
            if (flattened.decisive) return flattened
            // Some extractors preserve a table as pipe-delimited text even when geometry was too
            // weak for the conservative TABLE label. The delimiters are still explicit evidence.
            val tables = scoped.filter { it.contentKind == "TABLE" || '|' in it.text }
            if (tables.isEmpty()) return SummaryLead.EMPTY
            val queryNumbers = NUMBER_TOKEN.findAll(query).map { it.value }.toSet()
            if (toleranceAsk) {
                val evidenceWords = evidenceWords(query)
                val rows = tables.flatMap { citation ->
                    markdownRows(citation).map { cells ->
                        val text = cells.joinToString(" ")
                        Triple(citation, cells, evidenceWords(text).count(evidenceWords::contains))
                    }
                }.filter { (_, cells, score) ->
                    score >= 2 && NUMBER_TOKEN.containsMatchIn(cells.joinToString(" "))
                }.sortedByDescending { it.third }
                // The same schedule can legitimately be repeated in a specification package.
                // Treat identical rows as corroboration; only equal-scoring *different* rows are
                // ambiguous. Prefer the highest retrieval score as the displayed citation.
                val groups = rows.groupBy { (_, cells, _) -> canonicalTableRow(cells) }
                    .values.sortedByDescending { group -> group.maxOf { it.third } }
                val bestGroup = groups.firstOrNull()
                val bestScore = bestGroup?.maxOfOrNull { it.third }
                val best = bestGroup?.maxByOrNull { it.first.score }
                if (best != null && groups.drop(1).none { group -> group.maxOf { it.third } == bestScore }) {
                    return SummaryLead(
                        "${best.second.joinToString(" — ")} [Page ${best.first.pageNumber}]",
                        listOf(best.first),
                        decisive = true,
                    )
                }
                // A tolerance/variation question must not inherit an unrelated nearby table merely
                // because it contains many numbers. Let grounded generation use normal context.
                return SummaryLead.EMPTY
            }
            val matrixCell = buildMatrixCellLead(query, tables)
            if (matrixCell.decisive) return matrixCell
            val riskResponse = buildRiskResponseLead(query, scoped)
            if (riskResponse.decisive) return riskResponse
            val fireClass = buildFireClassLead(query, scoped)
            if (fireClass.decisive) return fireClass
            val rowKey = buildRowKeyLead(query, tables)
            if (rowKey.decisive) return rowKey
            if (queryNumbers.size >= 2) {
                val exactRows = tables.flatMap { citation ->
                    citation.text.lines().mapNotNull { line ->
                        val normalized = normalizeForEvidenceMatch(line)
                        line.takeIf { '|' in it && queryNumbers.all { number -> number in normalized } }
                            ?.let { citation to it }
                    }
                }
                if (exactRows.size == 1) {
                    val (citation, row) = exactRows.single()
                    val cells = row.trim().trim('|').split('|').map(String::trim)
                        .filter { it.isNotBlank() && !it.matches(Regex("-+")) }
                    if (cells.size >= 2) {
                        val label = if (Regex("\\bclass\\b").containsMatchIn(query)) "Class ${cells.first()}" else cells.first()
                        return SummaryLead(
                            "$label — ${cells.drop(1).joinToString(" — ")} [Page ${citation.pageNumber}]",
                            listOf(citation),
                            decisive = true,
                        )
                    }
                }
            }
            // Non-decisive rows already exist in the selected prompt. Do not expose diagnostic
            // source dumps in the user answer; grounded generation will format the result.
            return SummaryLead.EMPTY
        }

        /** Rebuilds complete rows from pre-split canonical cells for deterministic table lookup. */
        internal fun canonicalTableEvidence(
            manifest: DocumentStructureManifest?,
            tableId: String,
            source: List<Citation>,
        ): List<Citation> {
            val table = manifest?.tables?.firstOrNull { it.tableId == tableId } ?: return source
            if (table.cells.isEmpty()) return source
            val rows = table.cells.groupBy { it.rowIndex }.toSortedMap().mapNotNull { (rowIndex, cells) ->
                val ordered = cells.sortedBy { it.columnIndex }
                val width = maxOf(
                    table.columnHeaders.size,
                    (ordered.maxOfOrNull { it.columnIndex } ?: -1) + 1,
                )
                if (width < 2) return@mapNotNull null
                val headers = (0 until width).map { column ->
                    table.columnHeaders.getOrNull(column)
                        ?: ordered.firstOrNull { it.columnIndex == column }?.columnHeaderPath?.lastOrNull().orEmpty()
                }
                val values = (0 until width).map { column ->
                    ordered.firstOrNull { it.columnIndex == column }?.text.orEmpty()
                }
                val sourceIds = ordered.flatMap { it.sourceChunkIds }.distinct()
                val base = source.firstOrNull { it.chunkId in sourceIds }
                    ?: source.firstOrNull { it.pageNumber == ordered.first().pageNumber }
                    ?: return@mapNotNull null
                fun row(values: List<String>) = values.joinToString(" | ", prefix = "| ", postfix = " |") {
                    it.replace("|", "\\|")
                }
                base.copy(
                    chunkId = "${sourceIds.firstOrNull() ?: base.chunkId}:row$rowIndex",
                    text = row(headers) + "\n" + row(headers.map { "---" }) + "\n" + row(values),
                    tableId = table.tableId,
                    tableNumber = table.tableNumber,
                    tableCaption = table.caption,
                    retrievalProvenance = "CANONICAL_CELL",
                    sourceChunkId = sourceIds.firstOrNull() ?: base.chunkId,
                )
            }
            return (source + rows).ifEmpty { source }
        }

        /** Uses an exact printed table number as a local boundary when PDF geometry missed it. */
        internal fun scopeExplicitInlineTable(question: String, candidates: List<Citation>): List<Citation> {
            val number = EXPLICIT_TABLE_NUMBER.find(question)?.groupValues?.get(1) ?: return candidates
            val caption = Regex(
                "(?i)\\btable\\s*\\(?\\s*${Regex.escape(number)}\\s*\\)?(?![0-9A-Za-z.-])",
            )
            val anchors = candidates.filter { caption.containsMatchIn(it.tableCaption + " " + it.text) }
                .distinctBy { it.chunkIndex }
                .sortedBy { it.chunkIndex }
            if (anchors.isEmpty() || anchors.last().chunkIndex - anchors.first().chunkIndex > INLINE_TABLE_RADIUS) {
                return candidates
            }
            val anchor = anchors.first()
            val scoped = candidates.filter { candidate ->
                candidate.sectionId == anchor.sectionId &&
                    kotlin.math.abs(candidate.chunkIndex - anchor.chunkIndex) <= INLINE_TABLE_RADIUS
            }
            return scoped.ifEmpty { candidates }
        }

        /** The noun phrase a definition question asks about: "what is a trench" -> "trench". */
        internal fun definitionSubject(question: String): String? {
            if (AnswerShape.of(question) != AnswerShape.DEFINITION) return null
            val text = question.trim().trimEnd('?', '.', ' ').lowercase()
            for (pattern in DEFINITION_SUBJECT_PATTERNS) {
                val match = pattern.find(text) ?: continue
                val subject = match.groupValues[1].trim().replace(Regex("\\s+"), " ")
                if (subject.length in 2..60) return subject
            }
            return null
        }

        /**
         * Returns the source's own defining sentence for the asked subject when exactly one such
         * sentence exists in the evidence: `<subject> is|are|means|refers to|is defined as ...`.
         * A definition that introduces a list (ends with a colon) carries the items that follow it.
         */
        internal fun buildDefinitionLead(question: String, candidates: List<Citation>): SummaryLead {
            val subject = definitionSubject(question) ?: return SummaryLead.EMPTY
            val subjectPattern = definingSentencePattern(subject)
            val byIndex = candidates.associateBy { it.chunkIndex }
            val matches = candidates.flatMap { citation ->
                subjectPattern.findAll(citation.text).map { match -> citation to match.value.trim() }.toList()
            }.distinctBy { (_, sentence) -> normalizeForEvidenceMatch(sentence).replace(Regex("\\W+"), " ").trim() }
            if (matches.size != 1) return SummaryLead.EMPTY
            val (citation, sentence) = matches.single()
            val text = StringBuilder(sentence)
            val citations = arrayListOf(citation)
            if (sentence.endsWith(":")) {
                // The defining criteria follow the colon: the rest of this chunk and a list chunk after it.
                val rest = citation.text.substringAfter(sentence, "").trim().take(600)
                if (rest.isNotBlank()) text.append('\n').append(rest)
                val next = byIndex[citation.chunkIndex + 1]
                if (next != null && next.pageNumber == citation.pageNumber && next.contentKind == "LIST") {
                    text.append('\n').append(splitListItems(next).joinToString("\n"))
                    citations += next
                }
            }
            return SummaryLead("${text.toString().trim()} [Page ${citation.pageNumber}]", citations, decisive = true)
        }

        /**
         * "Which standard/code covers X": copies the one evidence line that names a standard
         * identifier (`IS 3764:1992`, `ASTM C150`, `EN 1991-1-4`) and shares at least two content
         * words with the question. Copying the line keeps the identifier and its year exact instead
         * of asking the model to re-type them.
         */
        internal fun buildStandardReferenceLead(question: String, candidates: List<Citation>): SummaryLead {
            val query = normalizeForEvidenceMatch(question)
            if (!STANDARD_QUERY_HINT.containsMatchIn(query)) return SummaryLead.EMPTY
            val queryWords = evidenceWords(query) - STANDARD_QUERY_STOP
            if (queryWords.size < 2) return SummaryLead.EMPTY
            // "what plywood standard" names the thing being classified. Require that subject in
            // the source sentence so a broad, higher-scored standards paragraph cannot win merely
            // because it repeats surrounding section words such as "concrete formwork".
            val requiredSubject = STANDARD_SUBJECT.find(query)?.groupValues?.get(1)
            val matches = candidates.flatMap { citation ->
                evidenceSentences(citation.text).mapNotNull { sentence ->
                    val bounded = sentence.replace(DANGLING_LIST_PREFIX, "").trim()
                    if (!STANDARD_ID.containsMatchIn(bounded)) return@mapNotNull null
                    val words = evidenceWords(bounded)
                    if (requiredSubject != null && requiredSubject !in words) return@mapNotNull null
                    val hits = words.count(queryWords::contains)
                    if (hits < 2) null else Triple(citation, bounded, hits)
                }
            }.groupBy { (_, line, _) -> normalizeForEvidenceMatch(line).replace(Regex("\\s+"), " ") }
                .values.map { group -> group.maxByOrNull { it.first.score }!! }
                .sortedByDescending { it.third }
            val best = matches.firstOrNull() ?: return SummaryLead.EMPTY
            if (matches.drop(1).any { it.third == best.third }) return SummaryLead.EMPTY
            return SummaryLead("${best.second} [Page ${best.first.pageNumber}]", listOf(best.first), decisive = true)
        }

        private val STANDARD_QUERY_HINT = Regex("\\b(?:standards?|codes?|norms?)\\b")
        private val STANDARD_SUBJECT = Regex("\\bwhat\\s+(?:the\\s+)?([\\p{L}\\p{N}-]{3,30})\\s+standards?\\b")
        // Sentence extraction can leave the next alphabetic list marker (for example `D.`)
        // attached to the selected line. Numeric tokens at the end are often the actual value
        // (`Class 1.`), so they must never be treated as disposable list markers here.
        private val DANGLING_LIST_PREFIX = Regex("\\s+[A-Z]\\.$")
        private val STANDARD_QUERY_STOP = setOf("standard", "code", "norm", "which", "cover", "covers", "applicable", "apply", "applies")
        /** Letter-prefixed standard identifiers with optional year or part suffix; grammar only, no issuer list. */
        private val STANDARD_ID = Regex("\\b[A-Z]{2,6}(?:\\s?[A-Z]{1,3})?[\\s-]?\\d{1,6}(?:[-–/.]\\d{1,4})*(?:\\s?[:(]\\s?\\d{4}\\)?)?\\b")

        /** Re-orders hits so chunks holding a defining sentence for the asked subject lead the list. */
        internal fun promoteDefiningSentences(question: String, hits: List<Citation>): List<Citation> {
            val subject = definitionSubject(question) ?: return hits
            val pattern = definingSentencePattern(subject)
            val best = hits.maxOfOrNull { it.score } ?: return hits
            return hits.map { hit ->
                if (pattern.containsMatchIn(hit.text)) hit.copy(score = best + DEFINITION_PROMOTION) else hit
            }.sortedByDescending { it.score }
        }

        internal fun definingSentencePattern(subject: String): Regex = Regex(
            "(?im)(?:^|(?<=[.;:\\n]\\s{0,3})|(?<=\\b[A-Z]{3,40}\\s))(?:the\\s+term\\s+)?['\"‘’“”]?(?:an?\\s+|the\\s+)?" + Regex.escape(subject) +
                "s?['\"‘’“”]?\\s+(?:is|are|means|refers\\s+to|is\\s+defined\\s+as|can\\s+be\\s+defined\\s+as|shall\\s+mean)\\b[^\\n]{3,400}?(?:(?<!\\d)[.:]|$)",
        )

        private const val DEFINITION_PROMOTION = 0.25

        private val DEFINITION_SUBJECT_PATTERNS = listOf(
            Regex("what makes (?:a|an|the) [\\p{L}\\s'’-]{1,40}? (?:a|an) ([\\p{L}\\s'’-]{2,60})$"),
            Regex("what (?:is|are) (?:the )?(?:definition|meaning) of (?:a |an |the )?([\\p{L}\\s'’-]{2,60})$"),
            Regex("what does (?:a |an |the )?([\\p{L}\\s'’-]{2,60}) mean$"),
            Regex("what is meant by (?:a |an |the )?([\\p{L}\\s'’-]{2,60})$"),
            Regex("define (?:a |an |the )?([\\p{L}\\s'’-]{2,60})$"),
            Regex("what (?:is|are) (?:a |an |the )?([\\p{L}\\s'’-]{2,60})$"),
        )

        /**
         * Row-label × column-header lookup in a grid ("Almost Certain likelihood and Catastrophic
         * consequence" -> the cell where that row and column meet). Labels are matched as whole
         * phrases inside the question; exactly one cell may qualify.
         */
        internal fun buildMatrixCellLead(query: String, tables: List<Citation>): SummaryLead {
            if (!MATRIX_INTERSECTION_QUERY.containsMatchIn(query)) return SummaryLead.EMPTY
            data class Cell(val citation: Citation, val row: String, val column: String, val value: String)
            fun phraseIn(label: String): Boolean {
                // "Almost Certain (5)" is asked about as "Almost Certain"; the scale in brackets is presentation.
                val normalized = normalizeForEvidenceMatch(label.replace(Regex("\\([^)]*\\)"), " "))
                    .replace(Regex("[^a-z0-9 ]"), " ").replace(Regex("\\s+"), " ").trim()
                if (normalized.length < 4) return false
                return Regex("(?<![a-z0-9])" + Regex.escape(normalized) + "(?![a-z0-9])").containsMatchIn(query)
            }
            val cells = tables.flatMap { citation ->
                val rows = markdownGrid(citation)
                if (rows.size < 2) return@flatMap emptyList()
                val header = rows.first()
                rows.drop(1).flatMap { row ->
                    val rowLabel = row.firstOrNull().orEmpty()
                    if (!phraseIn(rowLabel)) return@flatMap emptyList()
                    (1 until minOf(header.size, row.size)).mapNotNull { k ->
                        val column = header[k]
                        val value = row[k]
                        if (value.isBlank() || !phraseIn(column)) null else Cell(citation, rowLabel, column, value)
                    }
                }
            }.distinctBy { normalizeForEvidenceMatch("${it.row}|${it.column}|${it.value}") }
            val cell = cells.singleOrNull() ?: return SummaryLead.EMPTY
            return SummaryLead(
                "${cell.row} × ${cell.column}: ${cell.value} [Page ${cell.citation.pageNumber}]",
                listOf(cell.citation),
                decisive = true,
            )
        }

        /** Reverse lookup for `Which option ...?` rows that contain one explicit `Yes`. */
        internal fun buildBooleanAttributeLead(query: String, candidates: List<Citation>): SummaryLead {
            if (!WHICH_OPTION_QUERY.containsMatchIn(query)) return SummaryLead.EMPTY
            val queryWords = evidenceWords(query)
            candidates.filter { it.contentKind == "TABLE" || '|' in it.text }.forEach { table ->
                val rows = markdownGrid(table)
                val row = rows.firstOrNull { cells ->
                    val labelWords = evidenceWords(cells.firstOrNull().orEmpty())
                    labelWords.size >= 2 && labelWords.all(queryWords::contains) &&
                        cells.drop(1).count { it.equals("yes", true) } == 1
                } ?: return@forEach
                val optionCount = row.size - 1
                val selectedIndex = row.drop(1).indexOfFirst { it.equals("yes", true) }
                if (optionCount < 1 || selectedIndex < 0) return@forEach
                val prelude = candidates.firstOrNull { candidate ->
                    candidate.pageNumber == table.pageNumber && candidate.chunkIndex < table.chunkIndex &&
                        FEATURE_HEADER.containsMatchIn(candidate.text)
                } ?: return@forEach
                val words = prelude.text.replace(FEATURE_HEADER, " ").trim().split(Regex("\\s+"))
                if (words.size < optionCount || words.size % optionCount != 0) return@forEach
                val width = words.size / optionCount
                val options = words.chunked(width).map { it.joinToString(" ") }
                val option = options.getOrNull(selectedIndex) ?: return@forEach
                return SummaryLead(
                    "$option — ${row.first()}: Yes [Page ${table.pageNumber}]",
                    listOf(prelude, table),
                    decisive = true,
                )
            }
            return SummaryLead.EMPTY
        }

        /** Reads a typed first data row when PDF extraction omitted the table's real header. */
        internal fun buildHeaderlessTypedRowLead(query: String, tables: List<Citation>): SummaryLead {
            val hits = tables.flatMap { citation ->
                markdownRows(citation).mapNotNull { cells ->
                    val label = cells.firstOrNull().orEmpty()
                    val typed = TYPED_CLASS.find(label)
                    val shortTyped = label.trim().takeIf { it.matches(Regex("(?i)^[a-z0-9]{1,3}$")) }
                    val decimal = DECIMAL_VALUE.find(label)?.value
                    val strongIdentity = (typed != null && decimal != null && decimal in query) ||
                        (shortTyped != null && queryHasRowKey(query, shortTyped))
                    if (cells.size < 3 || !strongIdentity || !queryHasRowKey(query, label)
                    ) null else citation to cells
                }
            }.distinctBy { (_, cells) -> canonicalTableRow(cells) }
            val hit = hits.singleOrNull() ?: return SummaryLead.EMPTY
            return SummaryLead(
                "${hit.second.first()} — ${hit.second.drop(1).filter(String::isNotBlank).joinToString(" — ")} " +
                    "[Page ${hit.first.pageNumber}]",
                listOf(hit.first),
                decisive = true,
            )
        }

        /** Preserves a source's consecutive Never/Always safety instruction as one answer. */
        internal fun buildPairedDirectiveLead(question: String, candidates: List<Citation>): SummaryLead {
            val queryWords = evidenceWords(question)
            if (queryWords.size < 2) return SummaryLead.EMPTY
            val matches = candidates.mapNotNull { citation ->
                val directive = PAIRED_DIRECTIVE.find(citation.text)?.value?.replace(Regex("\\s+"), " ")?.trim()
                    ?: return@mapNotNull null
                val overlap = evidenceWords(directive).count(queryWords::contains)
                if (overlap < 2) null else Triple(citation, directive, overlap)
            }.sortedWith(compareByDescending<Triple<Citation, String, Int>> { it.third }
                .thenByDescending { it.first.score })
            val best = matches.firstOrNull() ?: return SummaryLead.EMPTY
            if (matches.getOrNull(1)?.third == best.third &&
                normalizeForEvidenceMatch(matches[1].second) != normalizeForEvidenceMatch(best.second)
            ) return SummaryLead.EMPTY
            return SummaryLead(
                "${best.second} [Page ${best.first.pageNumber}]",
                listOf(best.first),
                decisive = true,
            )
        }

        /** Reassembles an assessed-risk action row split vertically across a table and its next list. */
        internal fun buildRiskResponseLead(query: String, candidates: List<Citation>): SummaryLead {
            if (!RISK_RESPONSE_QUERY.containsMatchIn(query)) return SummaryLead.EMPTY
            val level = RISK_RESPONSE_LEVELS.firstOrNull { phraseInQuery(query, it) }
                ?: return SummaryLead.EMPTY
            val levelMatch = Regex("(?i)(?<![a-z])${Regex.escape(level)}(?![a-z])")
            val source = candidates.firstOrNull { citation ->
                (citation.contentKind == "TABLE" || '|' in citation.text) &&
                    levelMatch.containsMatchIn(citation.text) &&
                    ("-" in citation.text || "action" in citation.text.lowercase())
            } ?: return SummaryLead.EMPTY
            val levelStart = levelMatch.find(source.text)?.range?.first ?: return SummaryLead.EMPTY
            // In the extracted Table 1.4 continuation, the first Extreme action is emitted in
            // the visual row immediately before the cell containing the "Extreme" label.
            val precedingAlternative = if (level == "extreme") {
                Regex("(?i)-?\\s*consider alternatives?\\b").find(source.text)?.range?.first
            } else null
            val start = precedingAlternative?.takeIf { it < levelStart } ?: levelStart
            val body = source.text.substring(start)
                .replace('|', ' ')
                .replace(Regex("(?:^|\\s)-{3,}(?=\\s|$)"), " ")
                .replace(Regex("\\s+"), " ")
                .trim()
            val trailing = candidates
                .filter { it.chunkIndex in (source.chunkIndex + 1)..(source.chunkIndex + 2) }
                .sortedBy { it.chunkIndex }
                .takeWhile { it.contentKind == "LIST" || it.continuesFromChunkIndex >= 0 }
            val continuation = trailing.joinToString(" ") { it.text.replace(Regex("\\s+"), " ").trim() }
            val text = listOf(body, continuation).filter(String::isNotBlank).joinToString(" ")
            if (!RISK_ACTION_EVIDENCE.containsMatchIn(text)) return SummaryLead.EMPTY
            return SummaryLead(
                "$text [Page ${source.pageNumber}]",
                listOf(source) + trailing,
                decisive = true,
            )
        }

        /** Recovers a fire class and extinguisher method from a visually flattened table row. */
        internal fun buildFireClassLead(query: String, candidates: List<Citation>): SummaryLead {
            if (!FIRE_CLASS_QUERY.containsMatchIn(query)) return SummaryLead.EMPTY
            val matches = candidates.mapNotNull { citation ->
                if (!Regex("(?i)\\belectrical\\b").containsMatchIn(citation.text)) return@mapNotNull null
                val classCode = Regex("(?<![A-Za-z])([A-E])(?![A-Za-z])").find(citation.text)?.groupValues?.get(1)
                    ?: return@mapNotNull null
                val methods = FIRE_EXTINGUISHER_METHODS.findAll(citation.text)
                    .map { it.value.trim().replaceFirstChar(Char::uppercase) }
                    .distinctBy { it.lowercase() }
                    .toList()
                if (methods.isEmpty()) null else Triple(citation, classCode, methods)
            }.distinctBy { (citation, code, methods) -> "$code|${methods.joinToString()}|${citation.text}" }
            val match = matches.singleOrNull() ?: return SummaryLead.EMPTY
            return SummaryLead(
                "Electrical fires — Class ${match.second} — ${match.third.joinToString("; ")} [Page ${match.first.pageNumber}]",
                listOf(match.first),
                decisive = true,
            )
        }

        /**
         * Unique first-column key (or typed short label such as `class D`) plus an optional
         * named column. Range labels match `21-25`, `21 to 25` and `21 and 25`. Two tables that
         * share a key stay unresolved unless the question names one caption.
         */
        internal fun buildRowKeyLead(query: String, tables: List<Citation>): SummaryLead {
            data class Hit(
                val citation: Citation,
                val rowLabel: String,
                val column: String,
                val value: String,
                val cells: List<String>,
                val captionScore: Int,
            )
            val queryWords = evidenceWords(query)
            val hits = tables.flatMap { citation ->
                val rows = markdownGrid(citation)
                if (rows.size < 2) return@flatMap emptyList()
                val header = rows.first()
                val captionScore = evidenceWords(
                    normalizeForEvidenceMatch(citation.tableCaption + " " + header.joinToString(" ")),
                ).count(queryWords::contains)
                rows.drop(1).flatMap { row ->
                    val rowLabel = row.firstOrNull().orEmpty()
                    if (!queryHasRowKey(query, rowLabel)) return@flatMap emptyList()
                    val named = (1 until minOf(header.size, row.size)).mapNotNull { k ->
                        val column = header[k]
                        val value = row[k]
                        if (value.isBlank() || !queryNamesColumn(query, column)) null
                        else Hit(citation, rowLabel, column, value, row, captionScore)
                    }
                    if (named.isNotEmpty()) named
                    else listOf(
                        Hit(citation, rowLabel, "", row.drop(1).filter(String::isNotBlank).joinToString(" — "), row, captionScore),
                    )
                }
            }
            if (hits.isEmpty()) return SummaryLead.EMPTY
            val bestCaption = hits.maxOf { it.captionScore }
            val preferred = if (bestCaption > 0) hits.filter { it.captionScore == bestCaption } else hits
            val uniqueRows = preferred.distinctBy {
                normalizeForEvidenceMatch("${it.citation.chunkId}|${it.rowLabel}")
            }
            val uniqueValues = preferred.distinctBy { normalizeForEvidenceMatch("${it.rowLabel}|${it.column}|${it.value}") }
            val hit = uniqueRows.singleOrNull() ?: uniqueValues.singleOrNull() ?: return SummaryLead.EMPTY
            val typed = hit.cells.firstOrNull { it.matches(Regex("^[A-Za-z]\\d?$")) }
            val classLabel = if (typed != null && CLASS_WORD.containsMatchIn(query)) "Class $typed. " else ""
            val body = if (uniqueRows.size == 1) {
                "${hit.rowLabel} — ${hit.cells.drop(1).filter(String::isNotBlank).joinToString(" — ")}"
            } else if (hit.column.isNotBlank()) {
                "${hit.rowLabel} — ${hit.column}: ${hit.value}"
            } else {
                "${hit.rowLabel} — ${hit.value}"
            }
            return SummaryLead("$classLabel$body [Page ${hit.citation.pageNumber}]", listOf(hit.citation), decisive = true)
        }

        private fun queryHasRowKey(query: String, label: String): Boolean {
            val q = rowKeyNormalize(query)
            val forms = rowKeyForms(label)
            val longest = forms.maxByOrNull { it.length }.orEmpty()
            // A row whose only label is the word `class`/`type` is a header, not a key. `Class D`
            // still matches through the short typed-label path below.
            if (longest in GENERIC_ROW_KEYS) return false
            if (longest.length >= 4) {
                if (forms.any { form ->
                    Regex("(?<![a-z0-9])" + Regex.escape(form) + "(?![a-z0-9])").containsMatchIn(q)
                }) return true
                val typed = TYPED_CLASS.find(label)
                if (typed != null && Regex(
                        "(?i)\\b${Regex.escape(typed.groupValues[1])}\\s+${Regex.escape(typed.groupValues[2])}\\b",
                    ).containsMatchIn(query)
                ) {
                    val measurements = Regex("(?<![\\p{L}\\p{N}])\\d+\\.\\d+(?![\\p{L}\\p{N}])")
                        .findAll(label).map { it.value }.toList()
                    if (measurements.any { value -> value in query }) return true
                }
                val ends = RANGE_ENDS.find(rowKeyNormalize(label)) ?: return false
                return Regex("(?<![a-z0-9])${ends.groupValues[1]}(?![a-z0-9])").containsMatchIn(q) &&
                    Regex("(?<![a-z0-9])${ends.groupValues[2]}(?![a-z0-9])").containsMatchIn(q)
            }
            val key = rowKeyNormalize(label)
            if (key.isEmpty() || key.length > 3) return false
            return Regex("\\b(?:class|type|grade|group)\\s+" + Regex.escape(key) + "\\b").containsMatchIn(q)
        }

        private fun queryNamesColumn(query: String, header: String): Boolean {
            val normalized = rowKeyNormalize(header.replace(Regex("\\([^)]*\\)"), " "))
            if (normalized.length < 3) return false
            if (Regex("(?<![a-z0-9])" + Regex.escape(normalized) + "(?![a-z0-9])").containsMatchIn(rowKeyNormalize(query))) {
                return true
            }
            return normalized.split(' ').any { word ->
                word.length >= 4 && Regex("(?<![a-z0-9])" + Regex.escape(word) + "(?![a-z0-9])")
                    .containsMatchIn(rowKeyNormalize(query))
            }
        }

        private fun rowKeyNormalize(value: String): String = normalizeForEvidenceMatch(value)
            .replace(Regex("\\([^)]*\\)"), " ")
            .replace(Regex("[^a-z0-9]+"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()

        private fun rowKeyForms(label: String): Set<String> {
            val n = rowKeyNormalize(label)
            val ends = RANGE_ENDS.find(n) ?: return setOf(n)
            val a = ends.groupValues[1]
            val b = ends.groupValues[2]
            return setOf(n, "$a $b", "$a-$b", "$a to $b", "$a and $b")
        }

        /**
         * Copies the one evidence sentence that names a typed class (`type B`, `class D`) together
         * with a conditioned value (a ratio, measurement or range plus if/when/less-than/deep).
         * Compact models otherwise keep the value and drop the qualifier. Ambiguous matches decline.
         */
        internal fun buildConditionalValueLead(question: String, candidates: List<Citation>): SummaryLead {
            val labels = TYPED_CLASS.findAll(question).map { match ->
                match.groupValues[1].lowercase() to match.groupValues[2].lowercase()
            }.distinct().toList()
            if (labels.size != 1) return SummaryLead.EMPTY
            val (kind, label) = labels.single()
            val mention = Regex("(?i)\\b" + Regex.escape(kind) + "\\s+" + Regex.escape(label) + "\\b")
            val requestedTerms = HybridQuery.keywordTerms(question).toSet() - setOf(kind, label)
            val bareFollowUp = FOLLOW_UP_PREFIXES.any { question.trim().lowercase().startsWith(it) }
            fun matchesRequestedProperty(text: String): Boolean {
                if (bareFollowUp) return true
                val words = evidenceWords(text)
                val overlap = requestedTerms.count { term ->
                    val stem = HybridQuery.prefixTerm(term)?.removeSuffix("*") ?: term
                    words.any { it == stem || (stem.length >= 4 && it.startsWith(stem)) }
                }
                return overlap >= minOf(2, requestedTerms.size)
            }
            fun hasConditionedValue(text: String): Boolean {
                if (!CONDITION_CUE.containsMatchIn(text)) return false
                return EvidenceValueLexer.lex(text).tokens.any { token ->
                    token.kind == EvidenceValueKind.RATIO ||
                        token.kind == EvidenceValueKind.MEASUREMENT ||
                        token.kind == EvidenceValueKind.RANGE ||
                        token.kind == EvidenceValueKind.FRACTION
                }
            }

            data class ConditionalMatch(val citations: List<Citation>, val text: String)
            val directMatches = candidates.flatMap { citation ->
                evidenceSentences(citation.text).mapNotNull { sentence ->
                    if (!mention.containsMatchIn(sentence)) return@mapNotNull null
                    if (!hasConditionedValue(sentence)) return@mapNotNull null
                    // A substantive question must match its requested property/subject, not merely
                    // the same class. Prefix matching covers ordinary inflections (slope/sloped).
                    // A bare follow-up carries too little independent wording, so unique typed
                    // evidence may still answer it.
                    if (!matchesRequestedProperty(sentence)) return@mapNotNull null
                    ConditionalMatch(listOf(citation), sentence)
                }
            }

            // PDF layout extraction commonly separates a typed row heading from its value block:
            //   c721 "Excavations Made in Type B ..."
            //   c722 "All simple slope excavations ... maximum allowable slope of 1:1 ..."
            // Bind only the immediately following chunk on the same page. This preserves the row
            // association and prevents nearby Type A/C value blocks from being mixed into the answer.
            val byIndex = candidates.associateBy { it.chunkIndex }
            val adjacentMatches = candidates.mapNotNull { heading ->
                if (!mention.containsMatchIn(heading.text)) return@mapNotNull null
                val value = byIndex[heading.chunkIndex + 1] ?: return@mapNotNull null
                if (value.pageNumber != heading.pageNumber) return@mapNotNull null
                val conditioned = evidenceSentences(value.text).filter(::hasConditionedValue)
                if (conditioned.isEmpty()) return@mapNotNull null
                val text = conditioned.joinToString(" ")
                if (!matchesRequestedProperty(heading.text + " " + text)) return@mapNotNull null
                ConditionalMatch(listOf(heading, value), text)
            }

            val matches = (directMatches + adjacentMatches).distinctBy { match ->
                normalizeForEvidenceMatch(match.text).replace(Regex("\\s+"), " ")
            }
            if (matches.size != 1) return SummaryLead.EMPTY
            val match = matches.single()
            val page = match.citations.last().pageNumber
            return SummaryLead("${kind.replaceFirstChar(Char::uppercase)} ${label.uppercase()}: ${match.text} [Page $page]", match.citations, decisive = true)
        }

        /** Copies one uniquely matched sentence that states an explicit measured limit. */
        internal fun buildQuantifiedSentenceLead(question: String, candidates: List<Citation>): SummaryLead {
            val query = normalizeForEvidenceMatch(question)
            if (!QUANTIFIED_QUERY_HINT.containsMatchIn(query)) return SummaryLead.EMPTY
            if (QUANTIFIED_COMPARISON_HINT.containsMatchIn(query)) return SummaryLead.EMPTY
            if (QUANTIFIED_NON_NUMERIC_ATTRIBUTE.containsMatchIn(query)) return SummaryLead.EMPTY
            if (LOAD_MULTIPLIER_QUERY.containsMatchIn(query)) {
                val multiplierMatches = candidates.flatMap { citation ->
                    evidenceSentences(citation.text).mapNotNull { sentence ->
                        sentence.takeIf {
                            MULTIPLIER_VALUE.containsMatchIn(it) && SUPPORT_OR_CAPACITY.containsMatchIn(it)
                        }?.let { citation to it }
                    }
                }.distinctBy { normalizeForEvidenceMatch(it.second) }
                val match = multiplierMatches.singleOrNull()
                if (match != null) {
                    return SummaryLead(
                        "${match.second.trim()} [Page ${match.first.pageNumber}]",
                        listOf(match.first),
                        decisive = true,
                    )
                }
            }
            val requested = HybridQuery.keywordTerms(query).toSet() - QUANTIFIED_QUERY_STOP
            data class Match(val citation: Citation, val sentence: String, val score: Int)
            val matches = candidates.flatMap { citation ->
                val sentences = evidenceSentences(citation.text)
                sentences.mapIndexedNotNull { index, sentence ->
                    val valueCount = QUANTIFIED_VALUE.findAll(sentence).count() +
                        if (RATIO_QUERY_HINT.containsMatchIn(query)) DECIMAL_VALUE.findAll(sentence).count() else 0
                    if (valueCount == 0) return@mapIndexedNotNull null
                    if (TEMPERATURE_DURATION_QUERY.containsMatchIn(query) && valueCount < 2) {
                        return@mapIndexedNotNull null
                    }
                    val scopeStart = (index - 2).coerceAtLeast(0)
                    val scopedWords = evidenceWords(sentences.subList(scopeStart, index + 1).joinToString(" "))
                    val ownWords = evidenceWords(sentence)
                    val ownOverlap = requested.count { term ->
                        val stem = HybridQuery.prefixTerm(term)?.removeSuffix("*") ?: term
                        ownWords.any { it == stem || (stem.length >= 4 && it.startsWith(stem)) }
                    }
                    val overlap = requested.count { term ->
                        val stem = HybridQuery.prefixTerm(term)?.removeSuffix("*") ?: term
                        scopedWords.any { it == stem || (stem.length >= 4 && it.startsWith(stem)) }
                    }
                    val requiredOverlap = if (valueCount >= 2) 2 else 3
                    if (overlap < minOf(requiredOverlap, requested.size)) return@mapIndexedNotNull null
                    val typedValueBonus = if (
                        COVERAGE_RATE_QUERY.containsMatchIn(query) && COVERAGE_RATE_VALUE.containsMatchIn(sentence)
                    ) 500 else 0
                    Match(citation, sentence, typedValueBonus + ownOverlap * 20 + overlap * 10 + valueCount)
                }
            }.distinctBy { normalizeForEvidenceMatch(it.sentence).replace(Regex("\\s+"), " ") }
                .sortedByDescending { it.score }
            val best = matches.firstOrNull() ?: return SummaryLead.EMPTY
            if (matches.getOrNull(1)?.score == best.score) return SummaryLead.EMPTY
            return SummaryLead(
                "${best.sentence.trim()} [Page ${best.citation.pageNumber}]",
                listOf(best.citation),
                decisive = true,
            )
        }

        private fun evidenceSentences(text: String): List<String> {
            // Protect list ordinals before splitting punctuation: `10. IS 3764...` and
            // `C. Plywood...` are one evidence sentence, while physical line breaks are real
            // boundaries even when the preceding OCR text omitted terminal punctuation.
            val protected = LIST_ITEM_PREFIX.replace(text) { match ->
                match.groupValues[1] + match.groupValues[2] + ".\u00a0"
            }
            return protected.split(SENTENCE_BOUNDARY)
                .map { it.replace('\u00a0', ' ').trim() }
                .filter { it.length >= 12 }
        }

        /** Resolves explicit "which section" asks from a uniquely matching printed cross-reference. */
        internal fun buildSectionPointerAnswer(question: String, candidates: List<Citation>): SummaryLead {
            val query = normalizeForEvidenceMatch(question)
            if (!SECTION_LOOKUP_HINT.containsMatchIn(query)) return SummaryLead.EMPTY
            val queryWords = WORD_TOKEN.findAll(query).map { it.value }.filterNot { it in TABLE_STOP_WORDS }.toSet()
            val matches = candidates.flatMap { citation ->
                SECTION_POINTER.findAll(citation.text).map { match ->
                    val number = match.groupValues[1]
                    val title = match.groupValues[2].trim()
                    val titleWords = WORD_TOKEN.findAll(title.lowercase()).map { it.value }.toSet()
                    Triple(citation, number to title, titleWords.count(queryWords::contains))
                }.toList()
            }.filter { it.third >= 2 }
                .sortedByDescending { it.third }
            val best = matches.firstOrNull() ?: return SummaryLead.EMPTY
            if (matches.any { it.second.first != best.second.first && it.third == best.third }) return SummaryLead.EMPTY
            return SummaryLead(
                "Section ${best.second.first} – ${best.second.second} [Page ${best.first.pageNumber}]",
                listOf(best.first),
                decisive = true,
            )
        }

        /** Exact adjacent "as follows" + multi-value blocks are safe to return without sampling. */
        internal fun buildEnumeratedValueAnswer(question: String, candidates: List<Citation>): SummaryLead {
            val query = normalizeForEvidenceMatch(question)
            if (!LIST_LOOKUP_HINT.containsMatchIn(query)) return SummaryLead.EMPTY
            val queryWords = WORD_TOKEN.findAll(query).map { it.value }.filterNot { it in TABLE_STOP_WORDS }.toSet()
            val byIndex = candidates.associateBy { it.chunkIndex }
            val matches = candidates.mapNotNull { heading ->
                if (!AS_FOLLOWS.containsMatchIn(heading.text)) return@mapNotNull null
                val headingWords = WORD_TOKEN.findAll(heading.text.lowercase()).map { it.value }.toSet()
                // Three lexical anchors prevent a nearby structured list from hijacking a different
                // "what are" question (for example, slump values beside water-cement ratios).
                if (headingWords.count(queryWords::contains) < 3) return@mapNotNull null
                val values = byIndex[heading.chunkIndex + 1] ?: return@mapNotNull null
                if (DECIMAL_VALUE.findAll(values.text).count() < 2) return@mapNotNull null
                heading to values
            }
            if (matches.size != 1) return SummaryLead.EMPTY
            val (heading, values) = matches.single()
            val nextRequirement = NEXT_NUMBERED_REQUIREMENT.find(values.text)?.range?.first ?: values.text.length
            val body = values.text.substring(0, nextRequirement).trim()
            return SummaryLead(
                "${heading.text.trim()}\n$body [Page ${values.pageNumber}]",
                listOf(heading, values),
                decisive = true,
            )
        }

        /**
         * Returns a uniquely matched, physically adjacent numbered list without asking the model to
         * paraphrase or omit its items. PDF extraction commonly stores a list introduction and its
         * items as separate chunks, even though they form one semantic requirement.
         */
        internal fun buildNumberedListLead(question: String, candidates: List<Citation>): SummaryLead {
            val query = normalizeForEvidenceMatch(question)
            if (!NUMBERED_LIST_QUERY_HINT.containsMatchIn(query)) return SummaryLead.EMPTY
            val queryWords = evidenceWords(query)
            val implicitSubject = IMPLICIT_ENUMERATION_SUBJECT.find(query)?.groupValues?.get(1)
                ?.let(::evidenceWords)?.singleOrNull()
            val ofSubjectWords = ENUMERATION_OF_SUBJECT.find(query)?.groupValues?.get(1)
                ?.let(::evidenceWords).orEmpty()
            val byIndex = candidates.associateBy { it.chunkIndex }
            val requestedCount = requestedListItemCount(query)
            val exactCountRequested = EXACT_COUNT_QUERY.containsMatchIn(query)
            val bestScore = candidates.maxOfOrNull { it.score } ?: 0.0
            val topRanked = candidates.sortedByDescending { it.score }.take(MAX_ADJACENT_PRIMARY_SEEDS).map { it.chunkId }.toSet()
            val retrievedIds = candidates.filter { it.retrievalProvenance == "RETRIEVED" }.map { it.chunkId }.toSet()
            val matches = candidates.mapNotNull { introduction ->
                if (introduction.contentKind == "LIST" || !LIST_INTRODUCTION.containsMatchIn(introduction.text.trim())) return@mapNotNull null
                val follower = byIndex[introduction.chunkIndex + 1]
                val scopeWords = evidenceWords(introduction.text + " " + follower?.text.orEmpty())
                if (implicitSubject != null && implicitSubject !in scopeWords) return@mapNotNull null
                // A question such as "five basic steps of HIRA" supplies a decisive scope word.
                // It may occur in the list itself rather than its generic colon introduction.
                if (ofSubjectWords.isNotEmpty() && !scopeWords.containsAll(ofSubjectWords)) return@mapNotNull null
                val sharedWords = evidenceWords(introduction.text).count(queryWords::contains)
                // A colon-terminated sentence with two shared words is weak evidence on its own; a
                // low-ranked introduction from an unrelated page must not hijack the answer.
                val linkedToProminent = introduction.sourceChunkId in retrievedIds || follower?.sourceChunkId in retrievedIds
                val lexicalScopeMatches = if (implicitSubject != null) {
                    linkedToProminent && sharedWords >= 1
                } else {
                    sharedWords >= 2 && sharedWords * 5 >= queryWords.size * 2
                }
                if (!lexicalScopeMatches) return@mapNotNull null
                val prominent = introduction.chunkId in topRanked || follower?.chunkId in topRanked ||
                    linkedToProminent ||
                    (bestScore > 0 && introduction.score >= bestScore * ADJACENT_SEED_RELATIVE_FLOOR)
                if (!prominent) return@mapNotNull null
                // The list is the run of LIST chunks physically following the introduction. A
                // list that continues onto the next page is linked by the chunker's continuation
                // edge; an unrelated list on a later page is not.
                val firstListChunk = byIndex[introduction.chunkIndex + 1]
                    ?.takeIf { it.contentKind == "LIST" }
                val listChunks = if (firstListChunk?.listId?.isNotBlank() == true) {
                    candidates.filter { it.listId == firstListChunk.listId }.sortedBy { it.listItemStart }
                } else {
                    val adjacent = ArrayList<Citation>()
                    var previous: Citation = introduction
                    var next = firstListChunk
                    while (next != null && next.contentKind == "LIST" && adjacent.size < MAX_NUMBERED_LIST_CHUNKS) {
                        val samePage = next.pageNumber == previous.pageNumber
                        val continues = previous.contentKind == "LIST" && previous.continuesToChunkIndex == next.chunkIndex
                        if (!samePage && !continues) break
                        adjacent += next
                        previous = next
                        next = byIndex[next.chunkIndex + 1]
                    }
                    adjacent
                }
                val items = listChunks.flatMap { chunk ->
                    splitListItems(chunk).map { text -> chunk to text }
                }
                val structuredComplete = firstListChunk?.listId?.isNotBlank() == true &&
                    listChunks.firstOrNull()?.listItemStart == 0 &&
                    listChunks.zipWithNext().all { (left, right) ->
                        left.listItemStart + left.listItemCount == right.listItemStart
                    } && listChunks.lastOrNull()?.listComplete == true &&
                    listChunks.sumOf { it.listItemCount } == firstListChunk.listTotalItems &&
                    items.size == firstListChunk.listTotalItems
                val previous = listChunks.lastOrNull() ?: introduction
                val missingContinuation = previous.contentKind == "LIST" && previous.continuesToChunkIndex >= 0 &&
                    listChunks.none { it.chunkIndex == previous.continuesToChunkIndex }
                val legacyComplete = firstListChunk?.listId.isNullOrBlank() && !missingContinuation
                if (items.isEmpty() || (!structuredComplete && !legacyComplete) ||
                    (requestedCount != null && items.size < requestedCount) ||
                    (requestedCount != null && exactCountRequested && items.size != requestedCount)) null
                else introduction to items.take(requestedCount ?: items.size)
            }
            val introductionCount = candidates.count {
                it.contentKind != "LIST" && LIST_INTRODUCTION.containsMatchIn(it.text.trim())
            }
            runCatching {
                Log.i(
                    TAG,
                    "list lead query='${question.take(80)}' candidates=${candidates.size} " +
                        "introductions=$introductionCount " +
                        "matches=${matches.map { it.first.chunkId + ':' + it.second.size }}",
                )
            }
            if (matches.size != 1) return SummaryLead.EMPTY
            val (_, items) = matches.single()
            return SummaryLead(
                items.joinToString("\n") { (citation, text) -> "$text [Page ${citation.pageNumber}]" },
                items.map { it.first }.distinctBy { it.chunkId },
                decisive = true,
            )
        }

        /** LIST chunks store one labelled item per line (wrapped lines were merged at extraction). */
        internal fun splitListItems(chunk: Citation): List<String> {
            if (chunk.contentKind == "LIST") {
                val lines = chunk.text.lines().map(String::trim).filter(String::isNotBlank)
                // Each stored line is one labelled item; a compact single line may still hold
                // several inline numbered items.
                return lines.flatMap { line ->
                    if (LIST_LABEL_PREFIX.containsMatchIn(line)) splitNumberedListItems(line) else listOf(line)
                }
            }
            return splitNumberedListItems(chunk.text)
        }

        private fun splitNumberedListItems(text: String): List<String> {
            val bulletStarts = BULLET_LIST_ITEM.findAll(text).mapNotNull { it.groups[1]?.range?.first }.toList()
            val labelStarts = NUMBERED_LIST_ITEM.findAll(text).mapNotNull { it.groups[1]?.range?.first }.toList()
            val starts = when {
                bulletStarts.firstOrNull() != null &&
                    (labelStarts.firstOrNull() == null || bulletStarts.first() < labelStarts.first()) -> bulletStarts
                labelStarts.isNotEmpty() -> (bulletStarts + labelStarts).sorted()
                else -> bulletStarts
            }
            if (starts.isEmpty()) return listOf(text.trim()).filter(String::isNotBlank)
            return starts.mapIndexed { index, start ->
                text.substring(start, starts.getOrElse(index + 1) { text.length }).trim()
            }.filter(String::isNotBlank)
        }

        /** Copies the bullets belonging to one numbered subprocedure and stops at the next one. */
        internal fun buildScopedNumberedProcedureLead(question: String, candidates: List<Citation>): SummaryLead {
            if (!AnswerPolicy.isProcedural(question)) return SummaryLead.EMPTY
            val subjectStems = evidenceWords(question).filter { it.length >= 5 && it !in PROCEDURE_SCOPE_STOP }
                .map { it.take(6) }.toSet()
            if (subjectStems.isEmpty()) return SummaryLead.EMPTY
            val ordered = candidates.filter { it.contentKind == "LIST" }.sortedBy { it.chunkIndex }
            val starts = ordered.flatMap { citation ->
                NUMBERED_SUBPROCEDURE.findAll(citation.text)
                    .map { match -> Triple(citation, match, match.groupValues[2]) }.toList()
            }.filter { (_, _, title) ->
                evidenceWords(title).map { it.take(6) }.any(subjectStems::contains)
            }
            if (starts.size != 1) return SummaryLead.EMPTY
            val (first, match, _) = starts.single()
            val byIndex = ordered.associateBy { it.chunkIndex }
            val items = ArrayList<Pair<Citation, String>>()
            var current: Citation? = first
            var firstChunk = true
            var closed = false
            while (current != null && items.size < MAX_SCOPED_PROCEDURE_ITEMS) {
                val bodyStart = if (firstChunk) match.range.last + 1 else 0
                val body = current.text.substring(bodyStart).trim()
                val nextHeading = NUMBERED_SUBPROCEDURE.find(body)
                val scopedBody = if (nextHeading != null) body.substring(0, nextHeading.range.first) else body
                splitBulletItems(scopedBody).forEach { item -> items += current to item }
                if (nextHeading != null) {
                    closed = true
                    break
                }
                val next = byIndex[current.chunkIndex + 1]
                if (next == null || next.sectionId != first.sectionId) {
                    closed = current.listComplete
                    break
                }
                if (NUMBERED_SUBPROCEDURE.containsMatchIn(next.text)) {
                    closed = true
                    break
                }
                current = next
                firstChunk = false
            }
            if (!closed || items.size < 2) return SummaryLead.EMPTY
            return SummaryLead(
                items.joinToString("\n") { (citation, item) -> "$item [Page ${citation.pageNumber}]" },
                items.map { it.first }.distinctBy { it.chunkId },
                decisive = true,
            )
        }

        /** Extracts the uniquely strongest I–V control hierarchy from competing nearby lists. */
        internal fun buildHierarchyLead(question: String, candidates: List<Citation>): SummaryLead {
            val query = normalizeForEvidenceMatch(question)
            if (!HIERARCHY_CONTROL_QUERY.containsMatchIn(query)) return SummaryLead.EMPTY
            data class Entry(val ordinal: Int, val title: String, val citation: Citation)
            val entries = candidates.filter { it.contentKind == "LIST" }.flatMap { citation ->
                ROMAN_HIERARCHY_ITEM.findAll(citation.text).mapNotNull { match ->
                    ROMAN_ORDINALS.indexOf(match.groupValues[1].uppercase()).takeIf { it >= 0 }
                        ?.let { Entry(it + 1, match.groupValues[2].trim(), citation) }
                }.toList()
            }.sortedWith(compareBy<Entry> { it.citation.chunkIndex }.thenBy { it.ordinal })
            val sequences = entries.indices.mapNotNull { start ->
                if (entries[start].ordinal != 1) return@mapNotNull null
                val sequence = ArrayList<Entry>()
                var expected = 1
                for (entry in entries.drop(start)) {
                    if (entry.ordinal == expected) {
                        sequence += entry
                        expected++
                        if (expected == 6) break
                    }
                }
                sequence.takeIf { it.size == 5 }
            }
            val ranked = sequences.map { sequence ->
                sequence to sequence.count { entry ->
                    CONTROL_HIERARCHY_TERMS.any { term -> term in entry.title.lowercase() }
                }
            }.sortedByDescending { it.second }
            val best = ranked.firstOrNull() ?: return SummaryLead.EMPTY
            if (best.second < 4 || ranked.getOrNull(1)?.second == best.second) return SummaryLead.EMPTY
            return SummaryLead(
                best.first.joinToString("\n") { entry ->
                    "${ROMAN_ORDINALS[entry.ordinal - 1]}. ${entry.title} [Page ${entry.citation.pageNumber}]"
                },
                best.first.map { it.citation }.distinctBy { it.chunkId },
                decisive = true,
            )
        }

        private fun splitBulletItems(text: String): List<String> {
            val starts = BULLET_LIST_ITEM.findAll(text).mapNotNull { it.groups[1]?.range?.first }.toList()
            return starts.mapIndexed { index, start ->
                text.substring(start, starts.getOrElse(index + 1) { text.length }).trim()
            }.filter(String::isNotBlank)
        }

        /** Preserves an exact unique multi-value requirement even when a small summary model omits it. */
        internal fun buildEnumeratedSummaryLead(candidates: List<Citation>): SummaryLead {
            val byIndex = candidates.associateBy { it.chunkIndex }
            val matches = candidates.mapNotNull { heading ->
                if (!AS_FOLLOWS.containsMatchIn(heading.text)) return@mapNotNull null
                val values = byIndex[heading.chunkIndex + 1] ?: return@mapNotNull null
                if (DECIMAL_VALUE.findAll(values.text).count() < 2) return@mapNotNull null
                heading to values
            }
            if (matches.size != 1) return SummaryLead.EMPTY
            val (heading, values) = matches.single()
            val nextRequirement = NEXT_NUMBERED_REQUIREMENT.find(values.text)?.range?.first ?: values.text.length
            val body = values.text.substring(0, nextRequirement).trim()
            return SummaryLead(
                "Key exact values:\n${heading.text.trim()}\n$body [Page ${values.pageNumber}]",
                listOf(heading, values),
            )
        }

        private fun normalizeForEvidenceMatch(value: String): String = value.lowercase()
            .replace(",", "")
        private val NUMBER_TOKEN = Regex("\\d+(?:\\.\\d+)?")
        private val RANGE_ENDS = Regex("^(\\d+)\\s*(?:to|and|-|–|—)\\s*(\\d+)$")
        private val TYPED_CLASS = Regex("(?i)\\b(type|class|grade|group|category)\\s+(?:is\\s+)?([A-Za-z0-9]{1,4})\\b")
        private val CLASS_WORD = Regex("\\bclass\\b")
        private val GENERIC_ROW_KEYS = setOf("class", "type", "grade", "group", "item", "label", "category")
        private val CONDITION_CUE = Regex(
            "(?i)\\b(?:if|when|unless|where|less than|greater than|up to|below|above|deep|deeper|maximum|minimum)\\b",
        )
        private val QUANTIFIED_QUERY_HINT = Regex(
            "(?i)\\b(?:minimum|maximum|limit|dimensions?|depth|length|height|width|speed|distance|clearance|duration|temperature|pressure|voltage|openings?)\\b",
        )
        private val QUANTIFIED_COMPARISON_HINT = Regex(
            "(?i)\\b(?:compare|comparison|differences?|between|each|respectively)\\b",
        )
        private val QUANTIFIED_NON_NUMERIC_ATTRIBUTE = Regex(
            "(?i)\\b(?:colou?r|finish|appearance|white|pigmented)\\b",
        )
        private val RATIO_QUERY_HINT = Regex("(?i)\\bratios?\\b")
        private val TEMPERATURE_DURATION_QUERY = Regex(
            "(?is)(?=.*\\btemperature\\b)(?=.*\\b(?:duration|hours?|time)\\b)",
        )
        private val QUANTIFIED_VALUE = Regex(
            "(?i)(?<![\\p{L}\\p{N}])\\d+(?:[.,]\\d+)?(?:\\s*(?:to|[-–—/x×])\\s*\\d+(?:[.,]\\d+)?)?\\s*" +
                "(?:square\\s+(?:inches?|feet)|sq\\.?\\s*(?:in|ft)|gallons?|percent|times?|km/h|kmph|mm|cm|m|metres?|meters?|feet|foot|ft|inches?|inch|in|hours?|hrs?|minutes?|mins?|days?|volts?|v|degrees?|°[cf])\\b",
        )
        private val COVERAGE_RATE_QUERY = Regex("(?i)\\bcoverage\\s+rate\\b")
        private val COVERAGE_RATE_VALUE = Regex("(?i)\\bgallons?\\s+per\\s+\\d+\\s+square\\s+feet\\b")
        private val LOAD_MULTIPLIER_QUERY = Regex("(?i)(?=.*\\b(?:load|weight|capacity)\\b)(?=.*\\bsupport)")
        private val MULTIPLIER_VALUE = Regex("(?i)(?<![\\p{L}\\p{N}])\\d+(?:\\.\\d+)?\\s+times?\\b")
        private val SUPPORT_OR_CAPACITY = Regex("(?i)\\b(?:support|capacity|weight)\\w*\\b")
        private val QUANTIFIED_QUERY_STOP = setOf(
            "allow", "allowed", "allowable", "document", "does", "recommended", "shall", "should", "what", "which",
        )
        private val LIST_ITEM_PREFIX = Regex("(?m)(^|\\s)([A-Z]|\\d{1,3})\\.\\s+")
        private val SENTENCE_BOUNDARY = Regex("(?:\\r?\\n+|(?<=[.!?;])\\s+)")
        private val WORD_TOKEN = Regex("[a-z]{2,}")
        private val TABLE_QUERY_HINT = Regex("\\b(?:table|row|column|matrix|rating|schedule|class|slump|cement content)\\b")
        private val EXPLICIT_TABLE_NUMBER = Regex("(?i)\\btable\\s*\\(?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*\\)?")
        private const val INLINE_TABLE_RADIUS = 3
        private val RISK_MATRIX_QUERY = Regex("(?i)\\b(?:risk score|assessed risk)\\b")
        private val RISK_LIKELIHOODS = listOf("almost certain", "likely", "possible", "unlikely", "rare")
        private val RISK_CONSEQUENCES = listOf("insignificant", "minor", "moderate", "major", "catastrophic")
        private val PARENTHESIZED_SCORE = Regex("\\((\\d{1,2})\\)")
        private val RISK_RESPONSE_QUERY = Regex("(?i)(?=.*\\b(?:assessed )?risk\\b)(?=.*\\bactions?\\b)")
        private val RISK_RESPONSE_LEVELS = listOf("extreme", "high", "moderate", "low")
        private val RISK_ACTION_EVIDENCE = Regex("(?i)\\b(?:controls?|alternatives?|action|required|undertake|monitor)\\b")
        private val FIRE_CLASS_QUERY = Regex("(?i)(?=.*\\belectrical fires?\\b)(?=.*\\bclass\\b)(?=.*\\bextinguisher)")
        private val FIRE_EXTINGUISHER_METHODS = Regex("(?i)\\b(?:ABC powder|powder(?: type)?|carbon dioxide|CO2|foam spray|wet chemical|water)\\b")
        private val MATRIX_INTERSECTION_QUERY = Regex("(?i)\\b(?:and|versus|vs\\.?|intersection|row.+column|column.+row)\\b")
        private val WHICH_OPTION_QUERY = Regex("(?i)^\\s*which\\b.*\\b(?:option|type|class|item)\\b")
        private val FEATURE_HEADER = Regex("(?i)^\\s*feature\\s+")
        private val PAIRED_DIRECTIVE = Regex("(?i)\\bNever\\s+[^.!?]{3,180}[.!?]\\s+Always\\s+[^.!?]{3,180}[.!?]")
        private val SEVERITY_LEVEL_QUERY = Regex("(?i)\\bseverity\\s+level\\s+(\\d+)\\b")
        private fun phraseInQuery(query: String, phrase: String): Boolean = Regex(
            "(?<![a-z0-9])${Regex.escape(phrase)}(?![a-z0-9])",
        ).containsMatchIn(query)

        /** Maps an unprinted `2.3.1` reference to the first complete list under printed parent 2.3. */
        internal fun implicitSubsectionListIds(
            plan: QuestionPlan,
            manifest: DocumentStructureManifest,
        ): List<String> {
            val reference = plan.explicitSectionNumber ?: return emptyList()
            if (manifest.sections.any { it.sectionNumber.equals(reference, true) }) return emptyList()
            val separator = reference.lastIndexOf('.')
            if (separator <= 0) return emptyList()
            val ordinal = reference.substring(separator + 1).toIntOrNull()?.takeIf { it > 0 } ?: return emptyList()
            val parentNumber = reference.substring(0, separator)
            val parent = manifest.sections.singleOrNull { section ->
                section.sectionNumber.equals(parentNumber, true) &&
                    (plan.explicitSpecificationNumber == null ||
                        section.specificationNumber.equals(plan.explicitSpecificationNumber, true))
            } ?: return emptyList()
            if (plan.resolvedSectionId != parent.sectionId) return emptyList()
            return manifest.lists
                .filter { it.sectionId == parent.sectionId && it.complete }
                .sortedWith(compareBy<com.example.pdfgemmarag.inference.store.ListRecord> { it.startPage }
                    .thenBy { it.orderedChunkIds.firstOrNull()?.removePrefix("c")?.toIntOrNull() ?: Int.MAX_VALUE })
                .getOrNull(ordinal - 1)
                ?.orderedChunkIds
                .orEmpty()
        }
        private val TOLERANCE_QUERY_HINT = Regex("\\b(?:tolerance|variation)\\b")
        private val TOLERANCE_PARAGRAPH = Regex(
            "(?i)((?:variation|tolerance)[\\p{L}\\s,]{0,160}?)\\s+(\\d+/\\d+[\\\"”″]?\\s+in\\s+\\d+[’′']?)",
        )
        private val FLATTENED_TABLE_ROW = Regex(
            "(?i)([\\p{L}][\\p{L}\\p{N} .,/()'’&-]{1,180}?)\\s*:\\s*" +
                "([+\\-±]?\\d+(?:\\s+\\d+/\\d+|/\\d+|\\.\\d+|\\s*(?:-|–|—|to)\\s*\\d+(?:\\.\\d+)?)?%?[\\\"”″]?)",
        )
        private val TABLE_UNIT_HEADER = Regex("(?i)\\(\\s*in\\s+([\\p{L}][\\p{L} /-]{1,24})\\s*\\)")
        private val MULTI_ROW_QUERY_HINT = Regex(
            "(?i)\\b(?:values|rows|both|each|respectively)\\b|" +
                "\\b(?:minimum|maximum)\\b.{0,30}\\band\\b.{0,30}\\b(?:minimum|maximum)\\b",
        )
        private val MIN_MAX_SLUMP_QUERY = Regex(
            "(?is)(?=.*\\bminimum\\b)(?=.*\\bmaximum\\b)(?=.*\\bslump\\b)",
        )
        private val INLINE_MIN_MAX_HEADER = Regex("(?i)\\bminimum\\s+slump\\s+maximum\\s+slump\\b")
        private val INLINE_MIN_MAX_ROW = Regex(
            "(?i)([a-z][a-z ()/:-]{2,100}?)\\s+(\\d+(?:\\.\\d+)?[\\\"”″]?)\\s+(\\d+(?:\\.\\d+)?[\\\"”″]?)",
        )
        private const val FLATTENED_SCOPE_CHARS = 240
        private const val FLATTENED_SAFE_MARGIN = 2
        private val LEADING_REPEATED_WORD = Regex("(?i)^([\\p{L}]+)\\s+\\1\\b")
        private val SECTION_LOOKUP_HINT = Regex("\\b(?:what|which) section\\b")
        private val LIST_LOOKUP_HINT = Regex("\\b(?:list|what are)\\b")
        /** A request for an enumeration: a list verb followed by a plural noun within the clause. */
        private val NUMBERED_LIST_QUERY_HINT = Regex(
            "(?i)\\b(?:what are|list|name|give|which|enumerate|state|mention|identify)\\b.{0,80}\\b[a-z]{3,}(?:s|es)\\b|" +
                "\\bwhat\\b.{0,80}\\b(?:include|includes|consider|cover|address|contain)\\b|" +
                "\\b(?:priority\\s+order|hierarchy\\s+of\\s+(?:risk\\s+)?controls?)\\b",
        )
        private val IMPLICIT_ENUMERATION_SUBJECT = Regex(
            "(?i)\\bwhat\\s+(?!are\\b)([\\p{L}]{3,}(?:s|es))\\b.{0,80}\\b(?:include|includes|consider|cover|address|contain)\\b",
        )
        private val ENUMERATION_OF_SUBJECT = Regex(
            "(?i)\\b(?:steps?|stages?|classes?|types?|categories?|items?|requirements?)\\s+of\\s+(?:the\\s+)?" +
                "([\\p{L}\\p{N}-]+(?:\\s+[\\p{L}\\p{N}-]+){0,3})\\s*\\??$",
        )
        /** Any clause that ends with a colon introduces the list that physically follows it. */
        private val LIST_INTRODUCTION = Regex("(?i)[\\p{L}][^.!?]{3,240}:$")
        private val LIST_LABEL_PREFIX = Regex("^(?:(?:\\d{1,3}|[A-Za-z]|[ivxl]{2,6}|[IVXL]{2,6})[.)]|\\([A-Za-z0-9]{1,4}\\)|[•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF-])\\s")
        private val NUMBERED_LIST_ITEM = Regex("(?m)(?:^|\\s+)((?:\\d{1,2}|[A-Za-z]|[ivxl]{2,6}|[IVXL]{2,6})[.)]\\s+)")
        private val BULLET_LIST_ITEM = Regex("(?m)(?:^|\\s+)([•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF-]\\s+)")
        private val NUMBERED_SUBPROCEDURE = Regex(
            "(?i)(?:^|\\s)(\\d{1,2})[.)]\\s+([^•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF\\d]{2,80}?)(?=\\s+[•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF]|\\s+\\d{1,2}[.)]\\s+|$)",
        )
        private val PROCEDURE_SCOPE_STOP = setOf("first", "proced", "handli", "body", "part", "steps", "instruc")
        private const val MAX_SCOPED_PROCEDURE_ITEMS = 24
        private val HIERARCHY_CONTROL_QUERY = Regex("(?i)\\bhierarchy\\b.{0,40}\\bcontrols?\\b|\\bcontrols?\\b.{0,40}\\bhierarchy\\b")
        private val ROMAN_HIERARCHY_ITEM = Regex(
            "(?i)(?:^|\\s)(I|II|III|IV|V)\\.\\s+([^:–—-]{2,60}?)(?=\\s*[:–—-])",
        )
        private val ROMAN_ORDINALS = listOf("I", "II", "III", "IV", "V")
        private val CONTROL_HIERARCHY_TERMS = listOf(
            "elimination", "substitution", "engineering", "administrative", "protective equipment",
        )
        private const val MAX_NUMBERED_LIST_CHUNKS = 4
        /** "five basic steps", "3 classes": a count that qualifies a following plural noun. */
        private val REQUESTED_COUNT = Regex(
            "\\b(one|two|three|four|five|six|seven|eight|nine|ten|[1-9]|10)\\b(?=\\s+(?:[a-z]+\\s+){0,2}[a-z]{3,}(?:s|es)\\b)",
        )
        private val EXACT_COUNT_QUERY = Regex(
            "(?i)\\bwhat\\s+are\\s+(?:the\\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|[1-9]|10)\\b",
        )
        private val COUNT_WORDS = listOf("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        private fun requestedListItemCount(question: String): Int? {
            val token = REQUESTED_COUNT.find(question)?.groupValues?.get(1) ?: return null
            return token.toIntOrNull() ?: (COUNT_WORDS.indexOf(token) + 1).takeIf { it > 0 }
        }
        private val AS_FOLLOWS = Regex("(?i)\\bas follows\\s*:?$")
        private val DECIMAL_VALUE = Regex("(?<![\\p{L}\\p{N}])\\d+\\.\\d+(?![\\p{L}\\p{N}])")
        private val NEXT_NUMBERED_REQUIREMENT = Regex("\\s+\\d+[.)]\\s+")
        private val SECTION_POINTER = Regex(
            "(?i)\\bsection\\s+(0\\d{2,})\\s*[-–—:]\\s*([\\p{L}][\\p{L} ]{2,80})",
        )
        private val TABLE_STOP_WORDS = setOf(
            "what", "which", "are", "the", "and", "for", "with", "has", "have", "from", "values", "value",
            "about", "do", "got", "how", "in", "it", "much", "of", "okay", "on", "to", "we",
        )

        private data class FlattenedRow(
            val citation: Citation,
            val label: String,
            val value: String,
            val score: Int,
        )

        /** Reads a five-by-five risk matrix whose PDF extraction flattened each likelihood row. */
        private fun buildFlatRiskMatrixLead(query: String, candidates: List<Citation>): SummaryLead {
            if (!RISK_MATRIX_QUERY.containsMatchIn(query)) return SummaryLead.EMPTY
            val likelihood = RISK_LIKELIHOODS.firstOrNull { phrase -> phraseInQuery(query, phrase) }
                ?: return SummaryLead.EMPTY
            val consequenceIndex = RISK_CONSEQUENCES.indexOfFirst { phrase -> phraseInQuery(query, phrase) }
            if (consequenceIndex < 0) return SummaryLead.EMPTY
            val words = likelihood.split(' ')
            val source = candidates.firstOrNull { citation ->
                val text = normalizeForEvidenceMatch(citation.text)
                words.all { it in text } && PARENTHESIZED_SCORE.findAll(text).count() >= 5
            } ?: return SummaryLead.EMPTY
            // This table defines both axes as ordinal 1..5 scales and states Risk = Severity ×
            // Likelihood immediately above it. Apryse can emit each visual row's descriptions
            // first and its parenthesized scores later, so positional score parsing is unsafe.
            // Derive the intersection from the two explicit scale positions instead.
            val likelihoodScore = RISK_LIKELIHOODS.size - RISK_LIKELIHOODS.indexOf(likelihood)
            val consequenceScore = consequenceIndex + 1
            val value = likelihoodScore * consequenceScore
            return SummaryLead(
                "${likelihood.replaceFirstChar(Char::uppercase)} × ${RISK_CONSEQUENCES[consequenceIndex].replaceFirstChar(Char::uppercase)}: $value [Page ${source.pageNumber}]",
                listOf(source),
                decisive = true,
            )
        }

        /** Resolves a numbered definition row flattened into a list, such as severity level 4. */
        private fun buildNumberedInlineRowLead(query: String, candidates: List<Citation>): SummaryLead {
            val level = SEVERITY_LEVEL_QUERY.find(query)?.groupValues?.get(1) ?: return SummaryLead.EMPTY
            val row = Regex("(?i)(?:^|\\s)${Regex.escape(level)}[.)]\\s+(.+?)(?=\\s+\\d+[.)]\\s+|$)")
            val matches = candidates.mapNotNull { citation ->
                row.find(citation.text)?.groupValues?.get(1)?.trim()?.let { citation to it }
            }.distinctBy { normalizeForEvidenceMatch(it.second) }
            val match = matches.singleOrNull() ?: return SummaryLead.EMPTY
            return SummaryLead(
                "Severity level $level: ${match.second} [Page ${match.first.pageNumber}]",
                listOf(match.first),
                decisive = true,
            )
        }

        /** Resolves a letter-labelled row when a named visual table was extracted as a LIST. */
        internal fun buildLabelledListTableRowLead(query: String, candidates: List<Citation>): SummaryLead {
            if (!EXPLICIT_TABLE_NUMBER.containsMatchIn(query)) return SummaryLead.EMPTY
            val typed = TYPED_CLASS.findAll(query).map { it.groupValues[1] to it.groupValues[2] }
                .distinct().singleOrNull() ?: return SummaryLead.EMPTY
            val (kind, label) = typed
            val prefix = Regex("(?i)^${Regex.escape(label)}[.)]\\s+")
            val matches = candidates.filter { it.contentKind == "LIST" }.flatMap { citation ->
                splitListItems(citation).mapNotNull { item ->
                    item.takeIf(prefix::containsMatchIn)?.let { citation to it }
                }
            }.distinctBy { normalizeForEvidenceMatch(it.second) }
            val match = matches.singleOrNull() ?: return SummaryLead.EMPTY
            return SummaryLead(
                "${kind.replaceFirstChar(Char::uppercase)} ${match.second} [Page ${match.first.pageNumber}]",
                listOf(match.first),
                decisive = true,
            )
        }

        /** Rebuilds requested rows from a flattened `label minimum maximum` table paragraph. */
        private fun buildInlineMinMaxRowsLead(query: String, candidates: List<Citation>): SummaryLead {
            if (!MIN_MAX_SLUMP_QUERY.containsMatchIn(query)) return SummaryLead.EMPTY
            data class Row(val citation: Citation, val label: String, val minimum: String, val maximum: String)
            val queryWords = evidenceWords(query)
            val rows = candidates.flatMap { citation ->
                val text = citation.text.replace(Regex("\\s+"), " ").trim()
                val header = INLINE_MIN_MAX_HEADER.find(text) ?: return@flatMap emptyList()
                INLINE_MIN_MAX_ROW.findAll(text.substring(header.range.last + 1)).mapNotNull { match ->
                    val label = match.groupValues[1].trim().trimEnd(':')
                    val overlap = evidenceWords(label).count(queryWords::contains)
                    if (overlap < 3) null else Row(citation, label, match.groupValues[2], match.groupValues[3])
                }.toList()
            }.distinctBy { normalizeForEvidenceMatch(it.label) }
            if (rows.size < 2) return SummaryLead.EMPTY
            val citations = rows.map { it.citation }.distinctBy { it.chunkId }
            return SummaryLead(
                rows.joinToString("\n") { row ->
                    "- ${row.label}: minimum ${row.minimum}; maximum ${row.maximum} [Page ${row.citation.pageNumber}]"
                },
                citations,
                decisive = true,
            )
        }

        private fun buildFlattenedTableLead(query: String, candidates: List<Citation>): SummaryLead {
            val queryWords = evidenceWords(query)
            if (queryWords.isEmpty()) return SummaryLead.EMPTY
            val matches = candidates.flatMap { citation ->
                FLATTENED_TABLE_ROW.findAll(citation.text).mapNotNull { match ->
                    val label = match.groupValues[1].replace(Regex("\\s+"), " ").trim()
                    val labelWords = evidenceWords(label)
                    val labelHits = labelWords.count(queryWords::contains)
                    // One shared noun (for example "reinforcement") is not enough to select a
                    // row from a different nearby table. Deterministic answers require a compound
                    // label match; lower-confidence lookups remain with grounded generation.
                    if (labelHits < 2) return@mapNotNull null
                    val scopeStart = (match.range.first - FLATTENED_SCOPE_CHARS).coerceAtLeast(0)
                    val scopedPath = citation.text.substring(scopeStart, match.range.last + 1)
                    val scopeHits = evidenceWords(scopedPath).count(queryWords::contains)
                    if (scopeHits < 3) return@mapNotNull null
                    // Coverage makes a concise matching row label outrank a parser fragment that
                    // accidentally swallowed several words from its parent scope. Scope matches
                    // then distinguish repeated labels under different parents.
                    val labelCoverage = labelHits * 1_000 / labelWords.size.coerceAtLeast(1)
                    FlattenedRow(citation, label, match.groupValues[2].trim(), labelCoverage * 10 + scopeHits)
                }.toList()
            }
            val groups = matches.groupBy { canonicalTableRow(listOf(it.label, it.value)) }
                .values
                .sortedByDescending { group -> group.maxOf { it.score } }
            val bestGroup = groups.firstOrNull() ?: return SummaryLead.EMPTY
            // A single-row deterministic shortcut must never collapse a request for several
            // rows/values. Preserve the whole evidence set for the grounded multi-row path.
            if (MULTI_ROW_QUERY_HINT.containsMatchIn(query) && groups.size > 1) return SummaryLead.EMPTY
            val bestScore = bestGroup.maxOf { it.score }
            val runnerUp = groups.getOrNull(1)?.maxOf { it.score }
            if (runnerUp != null && bestScore - runnerUp < FLATTENED_SAFE_MARGIN) {
                return SummaryLead.EMPTY
            }
            val best = bestGroup.maxWithOrNull(
                compareBy<FlattenedRow> { it.score }.thenBy { it.citation.score },
            ) ?: return SummaryLead.EMPTY
            val unit = candidates.mapNotNull { TABLE_UNIT_HEADER.find(it.text)?.groupValues?.get(1)?.trim() }
                .distinctBy { it.lowercase() }
                .singleOrNull()
                .orEmpty()
            val displayedValue = listOf(best.value, unit).filter(String::isNotBlank).joinToString(" ")
            return SummaryLead(
                "${best.label} — $displayedValue [Page ${best.citation.pageNumber}]",
                listOf(best.citation),
                decisive = true,
            )
        }

        /** Rows with column positions preserved (blank cells kept), separator rows removed. */
        private fun markdownGrid(citation: Citation): List<List<String>> =
            citation.text.lines().map(String::trim).filter { it.startsWith("|") }.mapNotNull { line ->
                val cells = line.trim().trim('|').split('|').map(String::trim)
                cells.takeIf { values -> values.size >= 2 && !values.all { it.isEmpty() || it.all { ch -> ch == '-' || ch == ':' } } }
            }

        private fun markdownRows(citation: Citation): List<List<String>> =
            citation.text.split(Regex("\\|\\s*\\|")).mapNotNull { row ->
                val cells = row.trim().trim('|').split('|').map(String::trim)
                    .filter { it.isNotBlank() }
                cells.takeIf { values ->
                    values.size >= 2 && values.none { value -> value.all { it == '-' || it == ':' } }
                }
            }

        private fun evidenceWords(value: String): Set<String> = WORD_TOKEN.findAll(value.lowercase())
            .map { match ->
                val word = match.value
                when {
                    word == "sectional" -> "section"
                    word.length > 4 && word.endsWith('s') -> word.dropLast(1)
                    else -> word
                }
            }
            .filterNot { it in TABLE_STOP_WORDS }
            .toSet()

        private fun canonicalTableRow(cells: List<String>): String =
            normalizeForEvidenceMatch(cells.joinToString(" "))
                .replace('±', '+')
                .replace(Regex("[\\\"”″'’′]"), "")
                .replace(Regex("\\s+"), " ")

        /** A chunk whose last sentence introduces what follows ("… shall be as follows:"). */
        internal fun isListIntroduction(citation: Citation): Boolean =
            citation.contentKind != "LIST" && citation.text.trimEnd().endsWith(":")

        internal fun looksLikeStructuredFragment(citation: Citation): Boolean {
            if (citation.contentKind == "TABLE" || '|' in citation.text) return true
            val text = citation.text.lowercase()
            return LIST_INTRODUCTION.containsMatchIn(text.trim()) ||
                "tolerance" in text || "concrete type" in text || "classification" in text ||
                ("variation" in text && ("plumb" in text || "dimension" in text || "maximum" in text)) ||
                ("cross section" in text && ("column" in text || "wall" in text || "beam" in text)) ||
                ("minimum" in text && "maximum" in text) ||
                ("strength" in text && "cement" in text && NUMBER_TOKEN.findAll(text).count() >= 2)
        }

        /** Dependency-free lexical rank for manifest-scoped compatibility retrieval. */
        internal fun rankDirectScope(question: String, candidates: List<Citation>): List<Citation> {
            val terms = HybridQuery.keywordTerms(question)
            if (terms.isEmpty()) return candidates
            return candidates.map { citation ->
                val normalized = normalizeForEvidenceMatch(
                    listOf(citation.sectionPath, citation.text).joinToString(" "),
                )
                val matches = terms.count { term -> Regex("(?<![a-z0-9])${Regex.escape(term)}(?![a-z0-9])").containsMatchIn(normalized) }
                val phrase = terms.filterNot { it.all(Char::isDigit) }.joinToString(" ")
                val score = matches.toDouble() + if (phrase.isNotBlank() && phrase in normalized) 2.0 else 0.0
                citation.copy(score = score)
            }.sortedWith(compareByDescending<Citation> { it.score }.thenBy { it.chunkIndex })
        }

        internal fun rewriteForRetrieval(question: String, history: List<QaPair>): String {
            if (history.isEmpty()) return question
            val lower = question.lowercase()
            val looksLikeFollowUp = FOLLOW_UP_WORDS.any { Regex("\\b$it\\b").containsMatchIn(lower) } ||
                FOLLOW_UP_PREFIXES.any { lower.startsWith(it) }
            if (!looksLikeFollowUp) return question
            return "Previous question: ${history.last().question.take(300)}\nCurrent question: $question"
        }

        /** Repairs only an unambiguous truncated decimal; never invents or rounds a source value. */
        internal fun correctTruncatedDecimals(answer: String, source: String): String {
            val sourceDecimals = DECIMAL.findAll(source).map { it.value }.toSet()
            if (sourceDecimals.isEmpty()) return answer
            return DECIMAL.replace(answer) { match ->
                val value = match.value
                if (value in sourceDecimals) return@replace value
                val candidates = sourceDecimals.filter { it.startsWith(value) }
                if (candidates.size == 1) candidates.single() else value
            }
        }
    }

    /**
     * A decisive definition whose last evidence chunk ends by introducing a list (`:` or `;`)
     * is completed with the physically following list chunk when retrieval did not include it.
     */
    private suspend fun completeDefinition(
        lead: SummaryLead,
        ranked: List<Citation>,
        manifest: DocumentStructureManifest,
        question: String,
    ): SummaryLead {
        val last = lead.citations.lastOrNull() ?: return lead
        val body = lead.text.substringBeforeLast(" [Page").trimEnd()
        if (!(body.endsWith(":") || body.endsWith(";"))) return lead
        val nextId = DocumentStructureManifest.chunkId(last.chunkIndex + 1)
        if (ranked.any { it.chunkId == nextId } || nextId !in manifest.chunksInOrder.toHashSet()) return lead
        val extra = runCatching { store.getChunks(manifest, listOf(nextId)) }
            .onFailure { Log.w(TAG, "definition completion fetch failed: ${it.message}") }
            .getOrDefault(emptyList())
        if (extra.isEmpty()) return lead
        return buildDefinitionLead(question, ranked + extra).takeIf { it.decisive } ?: lead
    }

    /**
     * An explicit table identity is fetched from the manifest and merged ahead of hybrid hits so
     * the row/cell lead can answer from the grid even when paraphrase search ranked something else.
     */
    private suspend fun mergeResolvedTable(
        tableId: String,
        manifest: DocumentStructureManifest,
        primary: List<Citation>,
    ): List<Citation> {
        val table = manifest.tables.firstOrNull { it.tableId == tableId } ?: return primary
        if (table.orderedRowChunkIds.isEmpty()) return primary
        val finalRowIndex = table.orderedRowChunkIds.mapNotNull { it.removePrefix("c").toIntOrNull() }.maxOrNull()
        val immediateContinuation = finalRowIndex?.plus(1)?.let(DocumentStructureManifest::chunkId)
            ?.takeIf { it in manifest.chunksInOrder }
        val requestedIds = table.orderedRowChunkIds + listOfNotNull(immediateContinuation)
        val rows = runCatching { store.getChunks(manifest, requestedIds) }
            .onFailure { Log.w(TAG, "table fetch failed: ${it.message}") }
            .getOrDefault(emptyList())
        if (rows.isEmpty()) return primary
        val byId = primary.associateBy { it.chunkId }
        val boosted = rows.map { row ->
            val hit = byId[row.chunkId]
            hit?.copy(score = hit.score + TABLE_FETCH_BOOST) ?: row.copy(score = TABLE_FETCH_BOOST)
        }
        val extras = primary.filter { candidate -> rows.none { it.chunkId == candidate.chunkId } }
        return (boosted + extras).sortedByDescending { it.score }
    }

    private suspend fun expandStructuralNeighbors(
        primary: List<Citation>,
        manifest: DocumentStructureManifest?,
        retrievalQuestion: String,
    ): List<Citation> {
        if (manifest == null || primary.isEmpty()) return primary
        val neighbors = LinkedHashMap<String, LinkedHashMap<String, NeighborKind>>()
        val availableChunkIds = manifest.chunksInOrder.toHashSet()
        val listByFirstChunk = manifest.lists.associateBy { it.orderedChunkIds.first() }
        val bestPrimaryScore = primary.first().score
        val adjacentDistance = when {
            AnswerShape.of(retrievalQuestion) == AnswerShape.PROCEDURE -> 2
            LIMIT_OR_DURATION_QUERY.containsMatchIn(retrievalQuestion) -> 3
            else -> 1
        }
        primary.forEachIndexed { rank, citation ->
            val wanted = neighbors.getOrPut(citation.chunkId) { LinkedHashMap() }
            fun request(id: String, kind: NeighborKind) {
                val current = wanted[id]
                if (current == null || kind.priority > current.priority) wanted[id] = kind
            }
            if (citation.continuesFromChunkIndex >= 0) {
                request(DocumentStructureManifest.chunkId(citation.continuesFromChunkIndex), NeighborKind.CONTINUATION)
            }
            if (citation.continuesToChunkIndex >= 0) {
                request(DocumentStructureManifest.chunkId(citation.continuesToChunkIndex), NeighborKind.CONTINUATION)
            }
            // A list item is only meaningful with the sentence that introduces the list, which
            // the extractor stores as the preceding chunk and which may sit on the previous page.
            if (citation.contentKind == "LIST" && citation.chunkIndex > 0) {
                DocumentStructureManifest.chunkId(citation.chunkIndex - 1)
                    .takeIf { it in availableChunkIds }
                    ?.let { request(it, NeighborKind.INTRODUCTION) }
            }
            val list = citation.listId.takeIf(String::isNotBlank)?.let { id -> manifest.lists.firstOrNull { it.listId == id } }
                ?: listByFirstChunk[DocumentStructureManifest.chunkId(citation.chunkIndex + 1)]
                    ?.takeIf { isListIntroduction(citation) }
            list?.orderedChunkIds?.forEach { request(it, NeighborKind.LIST_MEMBER) }
            if (HIERARCHY_CONTROL_QUERY.containsMatchIn(retrievalQuestion) &&
                "hierarchy" in citation.text.lowercase()
            ) {
                manifest.lists.filter { candidate ->
                    candidate.sectionId == citation.sectionId &&
                        candidate.startPage <= citation.pageNumber + 2 && candidate.endPage >= citation.pageNumber - 1
                }.flatMap { it.orderedChunkIds }.forEach { request(it, NeighborKind.LIST_MEMBER) }
            }
            if (looksLikeStructuredFragment(citation)) {
                // Table captions are often emitted as several uppercase headings, each with a
                // distinct section id. Use validated physical chunk ids here; after fetching we
                // retain only same-page neighbours. Cross-page structure uses explicit links.
                for (distance in 1..MAX_STRUCTURAL_NEIGHBOR_DISTANCE) {
                    DocumentStructureManifest.chunkId(citation.chunkIndex - distance)
                        .takeIf { citation.chunkIndex >= distance && it in availableChunkIds }
                        ?.let { request(it, NeighborKind.STRUCTURAL) }
                    DocumentStructureManifest.chunkId(citation.chunkIndex + distance)
                        .takeIf { it in availableChunkIds }
                        ?.let { request(it, NeighborKind.STRUCTURAL) }
                }
            } else if (shouldExpandAdjacentSeed(rank, citation.score, bestPrimaryScore)) {
                // Expand a bounded number of actual top-ranked hits. AppSearch scores are
                // query-relative and higher is better, so an absolute score threshold is invalid.
                for (distance in 1..adjacentDistance) {
                    if (citation.chunkIndex >= distance) {
                        DocumentStructureManifest.chunkId(citation.chunkIndex - distance)
                            .takeIf { it in availableChunkIds }
                            ?.let { request(it, NeighborKind.ADJACENT) }
                    }
                    DocumentStructureManifest.chunkId(citation.chunkIndex + distance)
                        .takeIf { it in availableChunkIds }
                        ?.let { request(it, NeighborKind.ADJACENT) }
                }
            }
        }
        val extras = neighbors.values.flatMap { it.keys }.distinct().filter { id -> primary.none { it.chunkId == id } }
        if (extras.isEmpty()) return primary
        val queryTerms = HybridQuery.keywordTerms(retrievalQuestion).toSet()
        return runCatching {
            val fetched = store.getChunks(manifest, extras).associateBy { it.chunkId }
            // Keep a structural neighbour beside the ranked hit that requested it. Appending all
            // neighbours at the end made small models read a competing table before its value cell.
            buildList {
                val emitted = HashSet<String>()
                primary.forEach { citation ->
                    if (emitted.add(citation.chunkId)) add(citation)
                    neighbors[citation.chunkId].orEmpty().forEach { (id, kind) ->
                        fetched[id]
                            ?.takeIf { neighbor -> isEligibleNeighbor(citation, neighbor, kind, queryTerms) }
                            ?.takeIf { emitted.add(id) }
                            ?.let { neighbor ->
                                // Direct fetches have score 0. Preserve bounded provenance from the
                                // seed so ContextSelector does not immediately discard valid context.
                                add(
                                    neighbor.copy(
                                        score = citation.score * ADJACENT_SCORE_FACTOR,
                                        retrievalProvenance = kind.name,
                                        sourceChunkId = citation.chunkId,
                                    ),
                                )
                            }
                    }
                }
            }
        }.getOrElse {
            Log.w(TAG, "structural expansion skipped: ${it.message}")
            primary
        }
    }

    private fun isEligibleNeighbor(
        seed: Citation,
        neighbor: Citation,
        kind: NeighborKind,
        queryTerms: Set<String>,
    ): Boolean = when (kind) {
        NeighborKind.CONTINUATION -> true
        NeighborKind.LIST_MEMBER -> true
        NeighborKind.INTRODUCTION -> neighbor.sectionId == seed.sectionId && isListIntroduction(neighbor)
        NeighborKind.STRUCTURAL -> neighbor.pageNumber == seed.pageNumber
        NeighborKind.ADJACENT -> isEligibleAdjacentNeighbor(seed, neighbor, queryTerms)
    }

    private enum class NeighborKind(val priority: Int) {
        ADJACENT(1),
        STRUCTURAL(2),
        INTRODUCTION(3),
        CONTINUATION(4),
        LIST_MEMBER(5),
    }

    private fun deterministic(
        generationId: Long,
        started: Long,
        text: String,
        listener: Listener,
        citations: List<Citation> = emptyList(),
        sourceSectionId: String = "",
        manifestFallback: Boolean = false,
        answeredBy: String = "DETERMINISTIC",
        intent: QuestionIntent? = null,
        trace: EvidenceTrace? = null,
    ): GemmaEngine.Generation? {
        listener.onToken(text)
        // Publish only evidence used by the completed deterministic answer.
        listener.onRetrieved(citations.map { it.copy(text = "") })
        Log.i(TAG, "deterministic answer by $answeredBy (${citations.size} citations) in ${SystemClock.elapsedRealtime() - started} ms")
        listener.onDone(
            GenerationStats(
                generationId, 0, SystemClock.elapsedRealtime() - started, text.length, 0.0,
                0, 0, engine.backendName, false,
                sourceSectionId = sourceSectionId,
                manifestFallback = manifestFallback,
                intent = intent?.name.orEmpty(),
                answeredBy = answeredBy,
                candidateChunks = trace?.expanded?.size ?: citations.size,
                candidateChunkIds = (trace?.expanded ?: citations).joinToString(",") { it.chunkId },
                candidateProvenance = (trace?.expanded ?: citations).joinToString(",") {
                    "${it.chunkId}:${it.retrievalProvenance}:${it.sourceChunkId}"
                },
                selectedChunkIds = citations.joinToString(",") { it.chunkId },
                selectedPages = citations.map { it.pageNumber }.distinct().joinToString(","),
                answerCitationChunkIds = citations.joinToString(",") { it.chunkId },
                evidenceComplete = trace?.completeness == EvidenceCompleteness.COMPLETE,
                rawCandidateChunks = trace?.raw?.size ?: 0,
                rawCandidateChunkIds = trace?.raw?.joinToString(",") { it.chunkId }.orEmpty(),
                rawCandidatePages = trace?.raw?.map { it.pageNumber }?.distinct()?.joinToString(",").orEmpty(),
                evidenceCompleteness = trace?.completeness?.name ?: EvidenceCompleteness.UNKNOWN.name,
                selectionComplete = trace?.expanded?.all { candidate -> citations.any { it.chunkId == candidate.chunkId } } ?: false,
            ),
        )
        return null
    }

    private data class EvidenceTrace(
        val raw: List<Citation>,
        val expanded: List<Citation>,
        val completeness: EvidenceCompleteness,
    )

    private enum class EvidenceCompleteness { COMPLETE, PARTIAL, UNKNOWN }

    private suspend fun ambiguityCitations(
        manifest: DocumentStructureManifest?,
        plan: QuestionPlan,
    ): List<Citation> {
        if (manifest == null) return emptyList()
        val headingIds = plan.candidateSections.mapNotNull { it.orderedChunkIds.firstOrNull() }
        if (headingIds.isEmpty()) return emptyList()
        return runCatching { store.getChunks(manifest, headingIds) }
            .onFailure { Log.w(TAG, "could not attach ambiguity citations: ${it.message}") }
            .getOrDefault(emptyList())
    }
}
