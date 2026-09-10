package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.inference.pdf.StructureAnalyzer
import com.example.pdfgemmarag.inference.pdf.TableIdentity
import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.SectionRecord
import com.example.pdfgemmarag.inference.store.TableRecord
import java.text.Normalizer
import java.util.Locale

enum class QuestionIntent { FACT, SECTION_SUMMARY, DOCUMENT_OVERVIEW, AMBIGUOUS_SECTION }

/** An explicit reference to a structural node: `chapter 13`, `part iv`, `appendix a`. */
data class StructuralPointer(val kind: String, val printedNumber: String)

data class QuestionPlan(
    val intent: QuestionIntent,
    val subjectText: String,
    val explicitSpecificationNumber: String? = null,
    val explicitSectionNumber: String? = null,
    val resolvedSectionId: String? = null,
    val candidateSections: List<SectionRecord> = emptyList(),
    val sectionConfidence: Double = 0.0,
    val inheritedSectionId: String? = null,
    val shape: AnswerShape = AnswerShape.DEFAULT,
    val structuralPointer: StructuralPointer? = null,
    /**
     * True when [resolvedSectionId] names a node whose children hold the content (a chapter or
     * part). Summaries then read the whole subtree; fact retrieval stays unscoped because a
     * single sectionId filter would exclude the children.
     */
    val resolvedSubtree: Boolean = false,
    /** Unique table from an explicit `table 5.1` or a titled request (`the likelihood table`). */
    val resolvedTableId: String? = null,
    val candidateTables: List<TableRecord> = emptyList(),
) {
    fun clarification(): String {
        require(intent == QuestionIntent.AMBIGUOUS_SECTION)
        return buildString {
            append("I found more than one matching section. Which one do you mean?\n")
            candidateSections.forEach { section ->
                append("• ")
                if (section.specificationNumber.isNotBlank()) append("Specification ${section.specificationNumber}, ")
                if (section.sectionNumber.isNotBlank() && section.specificationNumber.isBlank() &&
                    section.kind != "SECTION"
                ) {
                    append(section.sectionNumber.lowercase(Locale.ROOT).replaceFirstChar { it.titlecase(Locale.ROOT) }).append(", ")
                } else if (section.sectionNumber.isNotBlank()) {
                    append("section ${section.sectionNumber}, ")
                }
                append(section.title.ifBlank { section.path })
                append(" (page ${section.startPage})\n")
            }
        }.trimEnd()
    }
}

/** Deterministic, dependency-free intent and section resolver. */
class QueryPlanner {
    fun plan(
        question: String,
        manifest: DocumentStructureManifest?,
        inheritedSectionId: String? = null,
        queryEmbedding: FloatArray? = null,
    ): QuestionPlan {
        val normalized = normalize(question)
        val shape = AnswerShape.of(question)
        val summary = SUMMARY.containsMatchIn(normalized)
        val explicitSpec = SPEC_REFERENCE.find(normalized)?.groupValues?.get(1)
        val inferredSpecCandidate = if (explicitSpec == null) LEADING_ZERO_ID.find(normalized)?.value else null
        val explicitSection = SECTION_REFERENCE.find(normalized)?.groupValues?.get(1)
        val explicitTable = TABLE_REFERENCE.find(normalized)?.groupValues?.get(1)
        val titledTable = if (explicitTable == null) TABLE_TITLE_REFERENCE.find(normalized)?.groupValues?.get(1) else null
        val inferredSectionCandidate = if (explicitSection == null) {
            DOTTED_ID.find(normalized)?.value?.takeIf { candidate ->
                explicitTable == null || !candidate.equals(explicitTable, true)
            }
        } else null
        val pointer = STRUCTURAL_REFERENCE.find(normalized)?.let { match ->
            StructuralPointer(match.groupValues[1].uppercase(Locale.ROOT).let { if (it.startsWith("ANNEX")) "APPENDIX" else it }, StructureAnalyzer.arabic(match.groupValues[2]))
        }
        // A titled structural request: "summarize the chapter on excavation", "the excavation chapter".
        val titledKind = if (pointer == null) STRUCTURAL_TITLE_REFERENCE.find(normalized)?.groupValues?.get(1)
            ?.uppercase(Locale.ROOT)?.let { if (it.startsWith("ANNEX")) "APPENDIX" else it } else null
        // "Specification 03300" identifies one specification; it is not a request to summarize
        // the entire document. Explicit identities always win over document-level vocabulary.
        val overview = summary && explicitSpec == null && inferredSpecCandidate == null &&
            explicitSection == null && inferredSectionCandidate == null && pointer == null && titledKind == null &&
            DOCUMENT_WORDS.containsMatchIn(normalized)
        var subject = normalized
            .replace(SUMMARY, " ")
            .replace(SPEC_WORD, " ")
            .replace(SECTION_WORD, " ")
            .replace(STRUCTURAL_REFERENCE, " ")
            .replace(STRUCTURAL_WORD, " ")
            .replace(FILLER_WORDS, " ")
        // Explicitly-labelled identities are filters, not semantic query terms. A bare
        // leading-zero token is only removed after the manifest proves it is a specification;
        // otherwise it may be a table number, detail ID, drawing reference, or product code.
        if (explicitSpec != null) subject = subject.replace(explicitSpec, " ")
        if (explicitSection != null) subject = subject.replace(explicitSection, " ")
        subject = subject.replace(SPACES, " ").trim()

        if (manifest == null) {
            return QuestionPlan(
                intent = if (overview) QuestionIntent.DOCUMENT_OVERVIEW else if (summary) QuestionIntent.SECTION_SUMMARY else QuestionIntent.FACT,
                subjectText = subject,
                explicitSpecificationNumber = explicitSpec,
                explicitSectionNumber = explicitSection,
                inheritedSectionId = inheritedSectionId,
                shape = shape,
                structuralPointer = pointer,
            )
        }

        val usable = manifest.sections.filter { it.title.isNotBlank() || it.sectionNumber.isNotBlank() }
        val resolvedTable = resolveTable(manifest, explicitTable, titledTable)

        // Explicit structural pointers ("chapter 13", "appendix a") resolve against kind and
        // printed number. Titled structural requests ("the chapter on excavation") resolve against
        // titles of that kind only, so a body heading with the same word cannot hijack them.
        if (pointer != null || titledKind != null) {
            val ofKind = usable.filter { it.kind == (pointer?.kind ?: titledKind) }
            val matches = if (pointer != null) {
                ofKind.filter { it.printedNumber.equals(pointer.printedNumber, true) }
            } else {
                val scored = ofKind.map { it to lexicalScore(subject, normalizeTitle(it.title)) }
                    .filter { it.second >= MIN_SCORE }
                    .sortedByDescending { it.second }
                val best = scored.firstOrNull()?.second
                if (best == null) emptyList() else scored.filter { best - it.second < SAFE_MARGIN }.map { it.first }
            }
            if (matches.size > 1 && summary) return ambiguous(subject, explicitSpec, explicitSection, matches, inheritedSectionId, shape, pointer)
            if (matches.size == 1) {
                val node = matches.single()
                return QuestionPlan(
                    intent = if (summary) QuestionIntent.SECTION_SUMMARY else QuestionIntent.FACT,
                    subjectText = subject,
                    explicitSpecificationNumber = explicitSpec,
                    explicitSectionNumber = explicitSection,
                    // A fact inside a chapter is found by hybrid search over the whole document; a
                    // sectionId filter would restrict it to the chapter's own introductory chunks.
                    resolvedSectionId = if (summary) node.sectionId else null,
                    candidateSections = matches,
                    sectionConfidence = 1.0,
                    inheritedSectionId = inheritedSectionId,
                    shape = shape,
                    structuralPointer = pointer ?: StructuralPointer(node.kind, node.printedNumber),
                    resolvedSubtree = summary,
                    resolvedTableId = resolvedTable?.tableId,
                    candidateTables = listOfNotNull(resolvedTable),
                )
            }
            // A pointer that this outline does not contain falls through to plain title matching
            // and hybrid retrieval rather than refusing outright.
        }
        // Table numbers and cross-references often look like CSI ids (03210B, 01450).
        // Only treat a leading-zero token as a specification filter when this document
        // actually contains that specification.
        val inferredSpec = inferredSpecCandidate?.takeIf { candidate ->
            usable.any { it.specificationNumber.equals(candidate, true) }
        }
        val inferredSection = inferredSectionCandidate?.takeIf { candidate ->
            usable.any { it.sectionNumber.equals(candidate, true) }
        }
        val spec = explicitSpec ?: inferredSpec
        val section = explicitSection ?: inferredSection
        if (inferredSpec != null) subject = subject.replace(inferredSpec, " ").replace(SPACES, " ").trim()
        if (inferredSection != null) subject = subject.replace(inferredSection, " ").replace(SPACES, " ").trim()
        val exactIdentity = usable.filter { candidate ->
            (spec == null || candidate.specificationNumber.equals(spec, true)) &&
                (section == null || candidate.sectionNumber.equals(section, true)) &&
                (spec != null || section != null)
        }
        if (exactIdentity.size == 1) {
            return QuestionPlan(
                intent = if (summary) QuestionIntent.SECTION_SUMMARY else QuestionIntent.FACT,
                subjectText = subject,
                explicitSpecificationNumber = spec,
                explicitSectionNumber = section,
                resolvedSectionId = exactIdentity.single().sectionId,
                candidateSections = exactIdentity,
                sectionConfidence = 1.0,
                inheritedSectionId = inheritedSectionId,
                shape = shape,
                resolvedTableId = resolvedTable?.tableId,
                candidateTables = listOfNotNull(resolvedTable),
            )
        }
        if (exactIdentity.size > 1 && summary) return ambiguous(subject, spec, section, exactIdentity, inheritedSectionId, shape, pointer)

        val exactTitle = usable.filter { normalizeTitle(it.title) == subject && subject.isNotBlank() }
        if (exactTitle.size > 1 && summary) return ambiguous(subject, spec, section, exactTitle, inheritedSectionId, shape, pointer)
        if (exactTitle.size == 1 && summary) {
            val node = exactTitle.single()
            return QuestionPlan(
                QuestionIntent.SECTION_SUMMARY,
                subject, spec, section, node.sectionId, exactTitle, 1.0, inheritedSectionId,
                shape = shape, resolvedSubtree = node.isTopLevelKind,
            )
        }

        // Lexical title matching is for summaries. A FACT question that happens to contain
        // a short heading word must not pin retrieval to that heading's chunks — sibling
        // subsections hold the requirements. Keep an explicit specification as a search
        // filter instead.
        if (!summary) {
            return QuestionPlan(
                intent = if (overview) QuestionIntent.DOCUMENT_OVERVIEW else QuestionIntent.FACT,
                subjectText = subject,
                explicitSpecificationNumber = spec,
                explicitSectionNumber = section,
                resolvedSectionId = inheritedSectionId.takeIf { looksLikeFollowUp(normalized) },
                inheritedSectionId = inheritedSectionId,
                shape = shape,
                structuralPointer = pointer,
                resolvedTableId = resolvedTable?.tableId,
                candidateTables = listOfNotNull(resolvedTable),
            )
        }

        val scored = usable.map { candidate ->
            val lexical = lexicalScore(subject, normalizeTitle(candidate.title))
            val centroid = if (lexical < MIN_SCORE && queryEmbedding != null) {
                candidate.centroid?.takeIf { it.size == queryEmbedding.size }?.let { cosine(queryEmbedding, it) } ?: 0.0
            } else 0.0
            candidate to maxOf(lexical, centroid)
        }
            .filter { it.second >= MIN_SCORE }
            .sortedByDescending { it.second }
        val best = scored.firstOrNull()
        val tied = if (best == null) emptyList() else scored.filter { best.second - it.second < SAFE_MARGIN }.map { it.first }
        if (summary && tied.size > 1) return ambiguous(subject, spec, section, tied, inheritedSectionId, shape, pointer)
        val selected = best?.takeIf { scored.getOrNull(1)?.let { second -> it.second - second.second >= SAFE_MARGIN } != false }
        val resolved = selected?.first?.sectionId ?: inheritedSectionId.takeIf { looksLikeFollowUp(normalized) }
        return QuestionPlan(
            intent = if (overview) QuestionIntent.DOCUMENT_OVERVIEW else if (summary) QuestionIntent.SECTION_SUMMARY else QuestionIntent.FACT,
            subjectText = subject,
            explicitSpecificationNumber = spec,
            explicitSectionNumber = section,
            resolvedSectionId = resolved,
            candidateSections = selected?.let { listOf(it.first) }.orEmpty(),
            sectionConfidence = selected?.second ?: 0.0,
            inheritedSectionId = inheritedSectionId,
            shape = shape,
            structuralPointer = pointer,
            resolvedSubtree = selected?.first?.isTopLevelKind == true,
            resolvedTableId = resolvedTable?.tableId,
            candidateTables = listOfNotNull(resolvedTable),
        )
    }

    private fun ambiguous(
        subject: String,
        spec: String?,
        section: String?,
        candidates: List<SectionRecord>,
        inherited: String?,
        shape: AnswerShape = AnswerShape.DEFAULT,
        pointer: StructuralPointer? = null,
    ) = QuestionPlan(
        QuestionIntent.AMBIGUOUS_SECTION,
        subject,
        spec,
        section,
        candidateSections = candidates.sortedWith(compareBy<SectionRecord> { it.startPage }.thenBy { it.sectionNumber }),
        sectionConfidence = 1.0,
        inheritedSectionId = inherited,
        shape = shape,
        structuralPointer = pointer,
    )

    private fun resolveTable(
        manifest: DocumentStructureManifest?,
        number: String?,
        titled: String?,
    ): TableRecord? {
        if (manifest == null || manifest.tables.isEmpty()) return null
        if (number != null) {
            return manifest.tables.filter { tableHasNumber(it, number) }.singleOrNull()
        }
        val phrase = titled?.trim().orEmpty()
        if (phrase.isBlank()) return null
        val scored = manifest.tables.map { it to tableTitleScore(phrase, it) }
            .filter { it.second >= MIN_TABLE_SCORE }
            .sortedByDescending { it.second }
        val best = scored.firstOrNull() ?: return null
        if (scored.drop(1).any { best.second - it.second < SAFE_MARGIN }) return null
        return best.first
    }

    private fun tableHasNumber(table: TableRecord, number: String): Boolean {
        if (table.tableNumber.equals(number, true)) return true
        if (TableIdentity.numberFromCaption(table.caption).equals(number, true)) return true
        return Regex("(?i)\\btable\\s*\\(?\\s*" + Regex.escape(number) + "\\s*\\)?").containsMatchIn(table.caption)
    }

    private fun tableTitleScore(phrase: String, table: TableRecord): Double {
        val caption = normalize(table.caption).replace(Regex("^table\\s+[0-9][0-9a-z.\\-]*\\s*"), "").trim()
        val q = phrase.split(' ').filter(String::isNotBlank).toSet()
        val t = (caption.split(' ').filter(String::isNotBlank) + table.aliases).toSet()
        val queryCoverage = q.intersect(t).size.toDouble() / q.size.coerceAtLeast(1)
        return maxOf(lexicalScore(phrase, caption), queryCoverage)
    }

    private fun lexicalScore(query: String, title: String): Double {
        if (query.isBlank() || title.isBlank()) return 0.0
        val q = query.split(' ').filter(String::isNotBlank).toSet()
        val t = title.split(' ').filter(String::isNotBlank).toSet()
        val jaccard = q.intersect(t).size.toDouble() / q.union(t).size.coerceAtLeast(1)
        val coverage = q.intersect(t).size.toDouble() / t.size.coerceAtLeast(1)
        val maxLen = maxOf(query.length, title.length).coerceAtLeast(1)
        val edit = if (maxLen <= MAX_EDIT_INPUT) {
            1.0 - boundedLevenshtein(query, title, MAX_EDIT_DISTANCE).coerceAtMost(maxLen).toDouble() / maxLen
        } else 0.0
        return maxOf(jaccard, coverage * 0.9, edit * 0.75)
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }

    companion object {
        private const val MIN_SCORE = 0.62
        private const val MIN_TABLE_SCORE = 0.66
        private const val SAFE_MARGIN = 0.12
        private const val MAX_EDIT_INPUT = 96
        private const val MAX_EDIT_DISTANCE = 24
        private val SPACES = Regex("\\s+")
        private val SUMMARY = Regex("\\b(?:summarize|summarise|summary|overview|key points?|explain)\\b")
        private val DOCUMENT_WORDS = Regex("\\b(?:document|pdf|specifications?|manual)\\b")
        private val SPEC_WORD = Regex("\\b(?:specification|spec)\\b")
        private val SECTION_WORD = Regex("\\b(?:section|clause)\\b")
        private val SPEC_REFERENCE = Regex("\\b(?:specification|spec)\\s+([0-9]{3,8}(?:[-.]?[a-z0-9]+)?)\\b")
        private val SECTION_REFERENCE = Regex("\\b(?:section|clause)\\s+((?:[0-9]+\\.)+[0-9a-z]+)\\b")
        private val LEADING_ZERO_ID = Regex("(?<!\\d)0\\d{2,}(?!\\d)")
        private val DOTTED_ID = Regex("(?<!\\d)(?:[0-9]+\\.)+[0-9a-z]+(?![\\w.])")
        private val FOLLOW_UP = Regex("^(?:and|what about|how about)\\b|\\b(?:it|that|this|those|these|same)\\b")
        /** `chapter 13`, `chapter xiii`, `part 4`, `appendix a`, `annexure ii`. */
        private val STRUCTURAL_REFERENCE = Regex(
            "\\b(chapter|part|appendix|annexure|annex)\\s+([0-9]{1,3}|[ivxlc]{1,7}|[a-z])\\b(?![.-]\\d)",
        )
        /** `the chapter on excavation`, `the excavation chapter`, `part about ppe`. */
        private val STRUCTURAL_TITLE_REFERENCE = Regex("\\b(chapter|part|appendix|annexure|annex)\\b")
        private val STRUCTURAL_WORD = Regex("\\b(?:chapter|part|appendix|annexure|annex)\\b")
        private val FILLER_WORDS = Regex("\\b(?:the|on|about|for|covering|regarding|of|this|a|an)\\b")
        /** `table 5.1`, `table 03210b`. Distinct from section/specification ids. */
        private val TABLE_REFERENCE = Regex("\\btable\\s+([0-9][0-9a-z.\\-]*)\\b")
        /** `in the likelihood table`, `in the risk assessment matrix`. */
        private val TABLE_TITLE_REFERENCE = Regex(
            "\\b(?:in|from|using)\\s+(?:the\\s+)?((?:[a-z0-9]+\\s+){0,6}[a-z0-9]+)\\s+(?:table|matrix)\\b",
        )

        fun normalize(text: String): String = Normalizer.normalize(text, Normalizer.Form.NFKC)
            .lowercase(Locale.ROOT)
            .replace(Regex("[^\\p{L}\\p{N}._-]+"), " ")
            .trim()
            .replace(SPACES, " ")

        fun normalizeTitle(text: String): String = StructureAnalyzer.normalizeHeading(text)

        fun looksLikeFollowUp(normalized: String): Boolean = FOLLOW_UP.containsMatchIn(normalized)

        /** Returns [limit] + 1 when the true distance exceeds the bound. */
        internal fun boundedLevenshtein(a: String, b: String, limit: Int): Int {
            if (kotlin.math.abs(a.length - b.length) > limit) return limit + 1
            var previous = IntArray(b.length + 1) { it }
            for (i in a.indices) {
                val current = IntArray(b.length + 1)
                current[0] = i + 1
                var rowMin = current[0]
                for (j in b.indices) {
                    current[j + 1] = minOf(
                        current[j] + 1,
                        previous[j + 1] + 1,
                        previous[j] + if (a[i] == b[j]) 0 else 1,
                    )
                    rowMin = minOf(rowMin, current[j + 1])
                }
                if (rowMin > limit) return limit + 1
                previous = current
            }
            return previous[b.length]
        }
    }
}
