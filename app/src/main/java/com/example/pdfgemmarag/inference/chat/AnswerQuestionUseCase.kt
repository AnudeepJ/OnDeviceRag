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

        val ranked = when {
            plan.intent == QuestionIntent.DOCUMENT_OVERVIEW && manifest != null -> {
                val ids = overviewChunkIds(manifest)
                if (ids.isEmpty()) return deterministic(generationId, t0, REPAIR_MESSAGE, listener)
                try {
                    store.getChunks(manifest, ids)
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
                    manifest.descendantsOf(section).flatMap { it.orderedChunkIds }.distinct().take(MAX_SUBTREE_CHUNKS)
                } else section.orderedChunkIds
                Log.i(TAG, "section summary ${section.kind} '${section.title.take(40)}' subtree=${plan.resolvedSubtree} chunks=${ids.size}")
                try {
                    store.getChunks(manifest, ids)
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
        val answerEvidence = plan.resolvedTableId?.let { tableId -> ranked.filter { it.tableId == tableId } } ?: ranked
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
                )
            }
        }
        if (plan.intent == QuestionIntent.FACT) {
            val standard = buildStandardReferenceLead(question, answerEvidence)
            if (standard.decisive) {
                return deterministic(
                    generationId, t0, standard.text, listener, standard.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "STANDARD_REFERENCE_LEAD", plan.intent,
                )
            }
            val numberedList = buildNumberedListLead(question, answerEvidence)
            if (numberedList.decisive) {
                return deterministic(
                    generationId, t0, numberedList.text, listener, numberedList.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "LIST_LEAD", plan.intent,
                )
            }
            val enumerated = buildEnumeratedValueAnswer(question, answerEvidence)
            if (enumerated.text.isNotEmpty()) {
                return deterministic(
                    generationId, t0, enumerated.text, listener, enumerated.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "ENUMERATED_VALUES_LEAD", plan.intent,
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
                    )
                }
            }
            val exactTableRow = buildTableLead(question, answerEvidence, plan.resolvedTableId)
            if (exactTableRow.decisive) {
                return deterministic(
                    generationId, t0, exactTableRow.text, listener, exactTableRow.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "TABLE_ROW_LEAD", plan.intent,
                )
            }
            val conditional = buildConditionalValueLead(question, answerEvidence)
            if (conditional.decisive) {
                return deterministic(
                    generationId, t0, conditional.text, listener, conditional.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback, "CONDITIONAL_VALUE_LEAD", plan.intent,
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
            if (sections.isEmpty()) return emptyList()
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
            val pool = if (health.degraded && representatives.size < maxSections) sections else representatives
            val sampled = quartileSample(pool.sortedBy { it.startPage }, manifest.lastPage, maxSections)
            return sampled.flatMap { it.orderedChunkIds.take(chunksPerSection) }.distinct()
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
            return SummaryLead(text, roots)
        }

        /** Exact table rows plus their structural neighbours, ranked by query words and numbers. */
        internal fun buildTableLead(
            question: String,
            candidates: List<Citation>,
            resolvedTableId: String? = null,
        ): SummaryLead {
            val scoped = if (resolvedTableId.isNullOrBlank()) {
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
            val matches = candidates.flatMap { citation ->
                citation.text.lines().mapNotNull { line ->
                    val trimmed = line.trim()
                    if (!STANDARD_ID.containsMatchIn(trimmed)) return@mapNotNull null
                    val hits = evidenceWords(trimmed).count(queryWords::contains)
                    if (hits < 2) null else Triple(citation, trimmed, hits)
                }
            }.groupBy { (_, line, _) -> normalizeForEvidenceMatch(line).replace(Regex("\\s+"), " ") }
                .values.map { group -> group.maxByOrNull { it.first.score }!! }
                .sortedByDescending { it.third }
            val best = matches.firstOrNull() ?: return SummaryLead.EMPTY
            if (matches.drop(1).any { it.third == best.third }) return SummaryLead.EMPTY
            return SummaryLead("${best.second} [Page ${best.first.pageNumber}]", listOf(best.first), decisive = true)
        }

        private val STANDARD_QUERY_HINT = Regex("\\b(?:standards?|codes?|norms?)\\b")
        private val STANDARD_QUERY_STOP = setOf("standard", "code", "norm", "which", "cover", "covers", "applicable", "apply", "applies")
        /** Letter-prefixed standard identifiers with optional year or part suffix; grammar only, no issuer list. */
        private val STANDARD_ID = Regex("\\b[A-Z]{2,6}(?:\\s?[A-Z]{1,3})?[\\s-]?\\d{2,6}(?:[-–/.]\\d{1,4})*(?:\\s?[:(]\\s?\\d{4}\\)?)?\\b")

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

        private fun evidenceSentences(text: String): List<String> =
            text.split(SENTENCE_BOUNDARY).map(String::trim).filter { it.length >= 12 }

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
            val byIndex = candidates.associateBy { it.chunkIndex }
            val requestedCount = requestedListItemCount(query)
            val bestScore = candidates.maxOfOrNull { it.score } ?: 0.0
            val topRanked = candidates.sortedByDescending { it.score }.take(MAX_ADJACENT_PRIMARY_SEEDS).map { it.chunkId }.toSet()
            val matches = candidates.mapNotNull { introduction ->
                if (introduction.contentKind == "LIST" || !LIST_INTRODUCTION.containsMatchIn(introduction.text.trim())) return@mapNotNull null
                if (evidenceWords(introduction.text).count(queryWords::contains) < 2) return@mapNotNull null
                // A colon-terminated sentence with two shared words is weak evidence on its own; a
                // low-ranked introduction from an unrelated page must not hijack the answer.
                val follower = byIndex[introduction.chunkIndex + 1]
                val prominent = introduction.chunkId in topRanked || follower?.chunkId in topRanked ||
                    (bestScore > 0 && introduction.score >= bestScore * ADJACENT_SEED_RELATIVE_FLOOR)
                if (!prominent) return@mapNotNull null
                // The list is the run of LIST chunks physically following the introduction. A
                // list that continues onto the next page is linked by the chunker's continuation
                // edge; an unrelated list on a later page is not.
                val listChunks = ArrayList<Citation>()
                var previous: Citation = introduction
                var next = byIndex[introduction.chunkIndex + 1]
                while (next != null && next.contentKind == "LIST" && listChunks.size < MAX_NUMBERED_LIST_CHUNKS) {
                    val samePage = next.pageNumber == previous.pageNumber
                    val continues = previous.contentKind == "LIST" && previous.continuesToChunkIndex == next.chunkIndex
                    if (!samePage && !continues) break
                    listChunks += next
                    previous = next
                    next = byIndex[next.chunkIndex + 1]
                }
                val items = listChunks.flatMap { chunk ->
                    splitListItems(chunk).map { text -> chunk to text }
                }
                val missingContinuation = previous.contentKind == "LIST" &&
                    previous.continuesToChunkIndex >= 0 &&
                    listChunks.none { it.chunkIndex == previous.continuesToChunkIndex }
                if (items.isEmpty() || missingContinuation || (requestedCount != null && items.size < requestedCount)) null
                else introduction to items.take(requestedCount ?: items.size)
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
            val starts = NUMBERED_LIST_ITEM.findAll(text).mapNotNull { it.groups[1]?.range?.first }.toList()
            if (starts.isEmpty()) return listOf(text.trim()).filter(String::isNotBlank)
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
        private val TYPED_CLASS = Regex("(?i)\\b(type|class|grade|group)\\s+([A-Za-z0-9]{1,4})\\b")
        private val CLASS_WORD = Regex("\\bclass\\b")
        private val GENERIC_ROW_KEYS = setOf("class", "type", "grade", "group", "item", "label", "category")
        private val CONDITION_CUE = Regex(
            "(?i)\\b(?:if|when|unless|where|less than|greater than|up to|below|above|deep|deeper|maximum|minimum)\\b",
        )
        private val SENTENCE_BOUNDARY = Regex("(?<=[.!?;\\n])\\s+")
        private val WORD_TOKEN = Regex("[a-z]{2,}")
        private val TABLE_QUERY_HINT = Regex("\\b(?:table|row|column|matrix|rating|schedule|class|slump|cement content)\\b")
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
        private const val FLATTENED_SCOPE_CHARS = 240
        private const val FLATTENED_SAFE_MARGIN = 2
        private val LEADING_REPEATED_WORD = Regex("(?i)^([\\p{L}]+)\\s+\\1\\b")
        private val SECTION_LOOKUP_HINT = Regex("\\b(?:what|which) section\\b")
        private val LIST_LOOKUP_HINT = Regex("\\b(?:list|what are)\\b")
        /** A request for an enumeration: a list verb followed by a plural noun within the clause. */
        private val NUMBERED_LIST_QUERY_HINT = Regex(
            "(?i)\\b(?:what are|list|name|give|which|enumerate|state|mention|identify)\\b.{0,80}\\b[a-z]{3,}(?:s|es)\\b",
        )
        /** Any clause that ends with a colon introduces the list that physically follows it. */
        private val LIST_INTRODUCTION = Regex("(?i)[\\p{L}][^.!?]{3,240}:$")
        private val LIST_LABEL_PREFIX = Regex("^(?:(?:\\d{1,3}|[A-Za-z]|[ivxl]{2,6}|[IVXL]{2,6})[.)]|\\([A-Za-z0-9]{1,4}\\)|[•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF-])\\s")
        private val NUMBERED_LIST_ITEM = Regex("(?m)(?:^|\\s+)((?:\\d{1,2}|[a-z]|[ivxl]{2,6}|[IVXL]{2,6})[.)]\\s+)")
        private const val MAX_NUMBERED_LIST_CHUNKS = 4
        /** "five basic steps", "3 classes": a count that qualifies a following plural noun. */
        private val REQUESTED_COUNT = Regex(
            "\\b(one|two|three|four|five|six|seven|eight|nine|ten|[1-9]|10)\\b(?=\\s+(?:[a-z]+\\s+){0,2}[a-z]{3,}(?:s|es)\\b)",
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
        val rows = runCatching { store.getChunks(manifest, table.orderedRowChunkIds) }
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
        val bestPrimaryScore = primary.first().score
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
                if (citation.chunkIndex > 0) {
                    DocumentStructureManifest.chunkId(citation.chunkIndex - 1)
                        .takeIf { it in availableChunkIds }
                        ?.let { request(it, NeighborKind.ADJACENT) }
                }
                DocumentStructureManifest.chunkId(citation.chunkIndex + 1)
                    .takeIf { it in availableChunkIds }
                    ?.let { request(it, NeighborKind.ADJACENT) }
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
                                add(neighbor.copy(score = citation.score * ADJACENT_SCORE_FACTOR))
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
        NeighborKind.INTRODUCTION -> neighbor.sectionId == seed.sectionId && isListIntroduction(neighbor)
        NeighborKind.STRUCTURAL -> neighbor.pageNumber == seed.pageNumber
        NeighborKind.ADJACENT -> isEligibleAdjacentNeighbor(seed, neighbor, queryTerms)
    }

    private enum class NeighborKind(val priority: Int) {
        ADJACENT(1),
        STRUCTURAL(2),
        INTRODUCTION(3),
        CONTINUATION(4),
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
            ),
        )
        return null
    }

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
