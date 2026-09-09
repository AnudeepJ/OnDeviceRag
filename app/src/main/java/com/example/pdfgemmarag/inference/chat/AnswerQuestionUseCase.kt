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
                try {
                    store.getChunks(manifest, section.orderedChunkIds)
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
                expandStructuralNeighbors(primary, manifest)
            }
        }
        if (plan.intent == QuestionIntent.FACT) {
            val numberedList = buildNumberedListLead(question, ranked)
            if (numberedList.decisive) {
                return deterministic(
                    generationId, t0, numberedList.text, listener, numberedList.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback,
                )
            }
            val enumerated = buildEnumeratedValueAnswer(question, ranked)
            if (enumerated.text.isNotEmpty()) {
                return deterministic(
                    generationId, t0, enumerated.text, listener, enumerated.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback,
                )
            }
            val pointer = buildSectionPointerAnswer(question, ranked)
            if (pointer.text.isNotEmpty()) {
                return deterministic(
                    generationId, t0, pointer.text, listener, pointer.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback,
                )
            }
            val exactTableRow = buildTableLead(question, ranked)
            if (exactTableRow.decisive) {
                return deterministic(
                    generationId, t0, exactTableRow.text, listener, exactTableRow.citations,
                    plan.resolvedSectionId.orEmpty(), manifestFallback,
                )
            }
        }
        val selected = selector.select(question, ranked, plan.intent)
        Log.i(TAG, "planned ${plan.intent} retrieved ${ranked.size} -> ${selected.excerpts.size} chunks (~${selected.approxTokens} tokens) in ${SystemClock.elapsedRealtime() - t0} ms")
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
            QuestionIntent.FACT -> buildTableLead(question, ranked)
            QuestionIntent.SECTION_SUMMARY -> buildEnumeratedSummaryLead(ranked)
            QuestionIntent.DOCUMENT_OVERVIEW -> buildOverviewLead(ranked)
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
            maxOutputTokens = when (plan.intent) {
                QuestionIntent.SECTION_SUMMARY -> 288
                QuestionIntent.DOCUMENT_OVERVIEW -> 224
                else -> {
                    val lower = question.lowercase()
                    if (lower.contains("steps") || lower.contains("procedure") || lower.contains("first aid") ||
                        lower.contains("list") || lower.contains("how to") || lower.contains("explain") ||
                        lower.contains("technique") || lower.contains("precautions")
                    ) 288 else 160
                }
            },
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

                override fun onDone(cancelled: Boolean) {
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
                    if (filter.hadGroundingFailure) Log.w(TAG, "answer contained unsupported values or citation ids")
                    val genMs = if (firstToken > 0) (end - firstToken).coerceAtLeast(1) else 1
                    listener.onDone(
                        GenerationStats(
                            generationId = generationId,
                            timeToFirstTokenMs = if (firstToken > 0) firstToken - t0 else -1,
                            totalMs = end - t0,
                            outputChars = chars,
                            approxTokensPerSecond = ContextAssembler.estimateTokens("x".repeat(chars)) * 1000.0 / genMs,
                            retrievedChunks = selected.excerpts.size,
                            contextTokensApprox = selected.approxTokens,
                            backend = engine.backendName,
                            cancelled = cancelled,
                            visibleTimeToFirstTokenMs = if (firstVisibleToken > 0) firstVisibleToken - t0 else -1,
                            groundingFailure = filter.hadGroundingFailure,
                            sourceSectionId = plan.resolvedSectionId.orEmpty(),
                            manifestFallback = manifestFallback,
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
        private const val MAX_NUMBERED_LIST_ITEMS = 8

        internal fun canFallbackWithoutManifest(
            intent: QuestionIntent,
            docHash: String,
            activeIndexNamespace: String,
        ): Boolean = intent == QuestionIntent.FACT && activeIndexNamespace.isNotBlank() &&
            (activeIndexNamespace == docHash || activeIndexNamespace.startsWith("$docHash:"))

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
            val positiveLevels = sections.map { it.level }.filter { it > 0 }
            val topLevel = positiveLevels.minOrNull()
            var representatives = if (topLevel == null) sections else sections.filter { it.level == topLevel }
            // A document with one root heading still needs breadth: include its immediate children.
            if (representatives.size < 3 && topLevel != null) {
                val childLevel = sections.map { it.level }.filter { it > topLevel }.minOrNull()
                if (childLevel != null) representatives = (representatives + sections.filter { it.level == childLevel }).distinct()
            }
            val sampled = evenlySpaced(representatives, maxSections)
            return sampled.flatMap { it.orderedChunkIds.take(chunksPerSection) }.distinct()
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
                    val identity = citation.specificationNumber
                        .takeIf(String::isNotBlank)
                        ?.let { "Specification $it — " }
                        .orEmpty()
                    val title = citation.sectionTitle.ifBlank {
                        citation.text.lineSequence().firstOrNull().orEmpty().trim()
                    }
                    append("- ").append(identity).append(title.ifBlank { "Untitled section" })
                        .append(" [Page ").append(citation.pageNumber).append("]\n")
                }
            }.trimEnd()
            return SummaryLead(text, roots)
        }

        /** Exact table rows plus their structural neighbours, ranked by query words and numbers. */
        internal fun buildTableLead(question: String, candidates: List<Citation>): SummaryLead {
            val query = normalizeForEvidenceMatch(question)
            val toleranceAsk = TOLERANCE_QUERY_HINT.containsMatchIn(query)
            // Lists of ratios and numbered requirements are structured too; that alone must not
            // cause an unrelated nearby table to be streamed as a supposedly relevant answer.
            if (!TABLE_QUERY_HINT.containsMatchIn(query) && !toleranceAsk) return SummaryLead.EMPTY
            if (toleranceAsk) {
                val queryEvidence = evidenceWords(query)
                val matches = candidates.flatMap { citation ->
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
            val flattened = buildFlattenedTableLead(query, candidates)
            if (flattened.decisive) return flattened
            // Some extractors preserve a table as pipe-delimited text even when geometry was too
            // weak for the conservative TABLE label. The delimiters are still explicit evidence.
            val tables = candidates.filter { it.contentKind == "TABLE" || '|' in it.text }
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
            val matches = candidates.mapNotNull { introduction ->
                if (!LIST_INTRODUCTION.containsMatchIn(introduction.text.trim())) return@mapNotNull null
                if (evidenceWords(introduction.text).count(queryWords::contains) < 2) return@mapNotNull null
                val listChunks = generateSequence(introduction.chunkIndex + 1) { it + 1 }
                    .mapNotNull(byIndex::get)
                    .takeWhile { it.pageNumber == introduction.pageNumber && it.contentKind == "LIST" }
                    .take(MAX_NUMBERED_LIST_ITEMS)
                    .toList()
                val items = listChunks.flatMap { chunk ->
                    splitNumberedListItems(chunk.text).map { text -> chunk to text }
                }
                if (items.isEmpty() || (requestedCount != null && items.size < requestedCount)) null
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
        private val WORD_TOKEN = Regex("[a-z]{2,}")
        private val TABLE_QUERY_HINT = Regex("\\b(?:table|row|class|slump|cement content)\\b")
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
        private val NUMBERED_LIST_QUERY_HINT = Regex(
            "(?i)\\b(?:what are|list|name|give|which)\\b.{0,80}\\b(?:rules|steps|precautions|guidelines|requirements|responsibilities|principles|measures)\\b",
        )
        private val LIST_INTRODUCTION = Regex(
            "(?i)\\b(?:rules|steps|precautions|guidelines|requirements|responsibilities|principles|measures)\\b[^.!?]{0,180}:$",
        )
        private val NUMBERED_LIST_ITEM = Regex("(?m)(?:^|\\s+)((?:\\d{1,2}|[a-z])[.)]\\s+)")
        private fun requestedListItemCount(question: String): Int? = when {
            Regex("\\b(?:one|1)\\b").containsMatchIn(question) -> 1
            Regex("\\b(?:two|2)\\b").containsMatchIn(question) -> 2
            Regex("\\b(?:three|3)\\b").containsMatchIn(question) -> 3
            Regex("\\b(?:four|4)\\b").containsMatchIn(question) -> 4
            Regex("\\b(?:five|5)\\b").containsMatchIn(question) -> 5
            else -> null
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

    private suspend fun expandStructuralNeighbors(
        primary: List<Citation>,
        manifest: DocumentStructureManifest?,
    ): List<Citation> {
        if (manifest == null || primary.isEmpty()) return primary
        val neighbors = LinkedHashMap<String, LinkedHashSet<String>>()
        val availableChunkIds = manifest.chunksInOrder.toHashSet()
        primary.forEach { citation ->
            val wanted = neighbors.getOrPut(citation.chunkId) { LinkedHashSet() }
            if (citation.continuesFromChunkIndex >= 0) wanted += DocumentStructureManifest.chunkId(citation.continuesFromChunkIndex)
            if (citation.continuesToChunkIndex >= 0) wanted += DocumentStructureManifest.chunkId(citation.continuesToChunkIndex)
            if (looksLikeStructuredFragment(citation)) {
                // Table captions are often emitted as several uppercase headings, each with a
                // distinct section id. Use validated physical chunk ids here; after fetching we
                // retain only same-page neighbours. Cross-page structure uses explicit links.
                for (distance in 1..MAX_STRUCTURAL_NEIGHBOR_DISTANCE) {
                    DocumentStructureManifest.chunkId(citation.chunkIndex - distance)
                        .takeIf { citation.chunkIndex >= distance && it in availableChunkIds }
                        ?.let(wanted::add)
                    DocumentStructureManifest.chunkId(citation.chunkIndex + distance)
                        .takeIf { it in availableChunkIds }
                        ?.let(wanted::add)
                }
            } else if (citation == primary.firstOrNull() || citation.score <= 1.0) {
                // For high-ranking paragraphs and lists, include 1-hop adjacent same-page chunks
                // to prevent heading/paragraph splits from dropping rule names or numerical criteria.
                if (citation.chunkIndex > 0) {
                    DocumentStructureManifest.chunkId(citation.chunkIndex - 1)
                        .takeIf { it in availableChunkIds }
                        ?.let(wanted::add)
                }
                DocumentStructureManifest.chunkId(citation.chunkIndex + 1)
                    .takeIf { it in availableChunkIds }
                    ?.let(wanted::add)
            }
        }
        val extras = neighbors.values.flatten().distinct().filter { id -> primary.none { it.chunkId == id } }
        if (extras.isEmpty()) return primary
        return runCatching {
            val fetched = store.getChunks(manifest, extras).associateBy { it.chunkId }
            // Keep a structural neighbour beside the ranked hit that requested it. Appending all
            // neighbours at the end made small models read a competing table before its value cell.
            buildList {
                val emitted = HashSet<String>()
                primary.forEach { citation ->
                    if (emitted.add(citation.chunkId)) add(citation)
                    neighbors[citation.chunkId].orEmpty().forEach { id ->
                        fetched[id]
                            ?.takeIf { it.pageNumber == citation.pageNumber && emitted.add(id) }
                            ?.let(::add)
                    }
                }
            }
        }.getOrElse {
            Log.w(TAG, "structural expansion skipped: ${it.message}")
            primary
        }
    }

    private fun deterministic(
        generationId: Long,
        started: Long,
        text: String,
        listener: Listener,
        citations: List<Citation> = emptyList(),
        sourceSectionId: String = "",
        manifestFallback: Boolean = false,
    ): GemmaEngine.Generation? {
        listener.onToken(text)
        // Publish only evidence used by the completed deterministic answer.
        listener.onRetrieved(citations.map { it.copy(text = "") })
        listener.onDone(
            GenerationStats(
                generationId, 0, SystemClock.elapsedRealtime() - started, text.length, 0.0,
                0, 0, engine.backendName, false,
                sourceSectionId = sourceSectionId,
                manifestFallback = manifestFallback,
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
