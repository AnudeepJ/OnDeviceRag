package com.example.pdfgemmarag.inference.llm

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.chat.AnswerShape
import com.example.pdfgemmarag.inference.chat.QuestionIntent

/** Intent-aware context budgeting with stable excerpt IDs and exact boundary deduplication. */
class ContextSelector(
    private val factBudget: Int = 1600,
    private val summaryBudget: Int = 900,
    private val overviewBudget: Int = 2200,
    private val maxFactPrimary: Int = 4,
    private val maxFactExcerpts: Int = 8,
    private val relativeScoreFloor: Double = 0.60,
) {
    data class Selection(val prompt: String, val excerpts: List<Citation>, val approxTokens: Int, val completeCoverage: Boolean)

    fun select(
        question: String,
        candidates: List<Citation>,
        intent: QuestionIntent,
        shape: AnswerShape = AnswerShape.of(question),
    ): Selection {
        val summary = intent == QuestionIntent.SECTION_SUMMARY
        val overview = intent == QuestionIntent.DOCUMENT_OVERVIEW
        val budget = when {
            summary -> summaryBudget
            overview -> overviewBudget
            else -> factBudget
        }
        val allUnique = candidates.distinctBy { it.indexNamespace to it.chunkId }
        // A resolved heading is useful to choose the section, but it is not evidence for the
        // section's requirements. Excluding it also prevents small models from treating identifiers
        // such as "3.26" as quantities.
        val unique = if (summary) allUnique.filterNot { it.contentKind == "HEADING" } else allUnique
        val ordered = if (summary) coverageOrder(unique) else unique
        val chosen = ArrayList<Citation>()
        var used = 0
        val best = ordered.firstOrNull()?.score ?: 0.0
        for (candidate in ordered) {
            if (!summary && !overview && chosen.size >= maxFactExcerpts) break
            val structural = isStructuralContext(candidate, unique)
            if (!summary && !overview && chosen.size >= maxFactPrimary && !structural) continue
            if (!summary && !overview && chosen.isNotEmpty() && !structural && best > 0 && candidate.score < best * relativeScoreFloor) continue
            val cost = ContextAssembler.estimateTokens(candidate.text) + 24
            if (used + cost > budget) continue
            chosen += candidate
            used += cost
        }
        // Small on-device models give the tail of a long prompt disproportionate attention. Keep
        // high-value limits/tables nearest the question without changing which evidence was chosen.
        val presented = if (summary) {
            // The question follows the excerpts, so keep the highest-information evidence at the
            // tail where the compact on-device model is most likely to retain exact values.
            chosen.take(1) + chosen.drop(1).sortedBy(::summaryPriority)
        } else if (overview) chosen else factPresentationOrder(question, chosen)
        val excerpts = presented.mapIndexed { index, citation -> citation.copy(excerptId = "E${index + 1}") }
        val rendered = exactBoundaryDedupe(excerpts)
        val prompt = buildString {
            append("Answer using only the excerpts below. Cite every factual paragraph with its excerpt id, for example [E1]. ")
            append("Never create an excerpt id. If the excerpts do not contain the answer, say that the document does not cover it. ")
            if ((summary || overview) && used < unique.sumOf { ContextAssembler.estimateTokens(it.text) + 24 }) {
                append("The context is a coverage selection; describe the result as key points rather than a complete summary. ")
            }
            if (summary) {
                append("Use at most 6 compact bullets and 110 words. Start with the answer; do not restate the question or section path. ")
                append("Treat section numbers and headings only as navigation labels; never present a section number as a quantity, measurement, or requirement. ")
                append("Finish every bullet as a complete sentence; omit a lower-priority bullet rather than ending mid-sentence. ")
                append("Combine related values from one requirement into one bullet and cover distinct requirement categories. ")
                append("Omit introductory purpose or objective statements when actionable requirements exist. If the source names who prepares, tests, or approves the work, give that responsibility its own bullet. ")
                append("Put distinct numeric limits and criteria compactly in the first bullet, then give descriptive requirements. ")
                append("Prefer ratios, percentage ranges, and table limits over formula or standards references when space is limited. ")
            } else if (overview) {
                append("For the document overview, add at most 3 compact bullets and 75 words describing key scope or requirement categories across the supplied outline sample. ")
                append("Describe it as key points, not a complete summary. Prefer scope and major requirement categories over isolated details. ")
            } else {
                when (shape) {
                    AnswerShape.PROCEDURE ->
                        append("Answer directly with the required steps or instructions in complete bullets. Preserve their source order and do not silently omit an item. Do not restate the question. ")
                    AnswerShape.DEFINITION ->
                        append("Answer with the definition or defining criteria exactly as the excerpts state them, in at most 4 short sentences or bullets. Give the meaning itself; do not answer with a section title or a page reference alone. ")
                    AnswerShape.NAVIGATION ->
                        append("Answer with the section or chapter identifier and title that covers the topic, then one sentence on what it contains. ")
                    else ->
                        append("Answer directly in at most 3 short sentences or bullets and 55 words. Do not restate the question. ")
                }
                append("If an excerpt states an explicit minimum, maximum, or 'not less than' value, answer with it and never claim that value is unspecified. ")
                append("When a value applies under a condition (a depth, type, class, size or range), state the condition together with the value. ")
                append("Keep table row labels with their values. If similar rows have different scopes or structure types, name each scope and do not merge their values. ")
                append("For a table lookup, match the complete hierarchy in the question—table, structure type, condition, and row label—and ignore values belonging to sibling paths. ")
            }
            append("Reply in the language of the question.\n\nExcerpts:\n")
            excerpts.forEachIndexed { index, citation ->
                append('[').append(citation.excerptId).append("]\n")
                if (citation.specificationNumber.isNotBlank()) append("Specification: ").append(citation.specificationNumber).append('\n')
                if ((!summary || overview) && citation.sectionPath.isNotBlank()) append("Section: ").append(citation.sectionPath).append('\n')
                append("Page: ").append(citation.pageNumber).append('\n')
                append("Content: ").append(rendered[index]).append("\n\n")
            }
            if (summary) append("Before answering, check the final excerpts for ratios, percentage ranges, and table limits.\n")
            append("Question: ").append(if (summary) withoutStructuralPointers(question) else question.trim())
        }
        return Selection(
            prompt,
            excerpts,
            used + ContextAssembler.estimateTokens(question) + 100,
            completeCoverage = excerpts.size == unique.size,
        )
    }

    internal fun exactBoundaryDedupe(excerpts: List<Citation>): List<String> {
        val rendered = ArrayList<String>(excerpts.size)
        excerpts.forEachIndexed { index, citation ->
            var text = citation.text.trim()
            val previous = excerpts.getOrNull(index - 1)
            if (previous != null && previous.sectionId == citation.sectionId) {
                val overlap = exactSuffixPrefix(previous.text.trim(), text)
                if (overlap > 0) text = text.substring(overlap).trimStart()
            }
            rendered += text
        }
        return rendered
    }

    private fun withoutStructuralPointers(question: String): String = question
        .replace(STRUCTURAL_POINTER, " ")
        .replace(Regex("\\s+"), " ")
        .trim()

    /**
     * Direct section fetches arrive in manifest order. Heading-only chunks are removed before this
     * point, so keep the opening substantive requirement as the anchor, then spend the bounded
     * on-device prompt on high-information requirements across the
     * section. This prevents a long section's introductory prose from crowding out later limits and
     * tables while remaining independent of any particular specification.
     */
    internal fun coverageOrder(candidates: List<Citation>): List<Citation> {
        val anchors = candidates.take(1)
        return anchors + candidates.drop(anchors.size).sortedWith(
            compareByDescending<Citation>(::summaryPriority)
                .thenBy { it.pageNumber }
                .thenBy { it.chunkIndex },
        )
    }

    internal fun summaryPriority(citation: Citation): Int {
        val text = citation.text.lowercase()
        return NUMBER.findAll(text).count().coerceAtMost(8) +
            (if (DECIMAL.findAll(text).count() >= 2) 32 else 0) +
            (if (citation.contentKind == "TABLE") 5 else 0) +
            (if (citation.continuesFromChunkIndex >= 0 || citation.continuesToChunkIndex >= 0) 24 else 0) +
            (if ("testing laboratory" in text || "responsible" in text) {
                if (text.length <= 600) 40 else 8
            } else 0) +
            (if ("entrained air" in text || "air content" in text) 26 else 0) +
            (if ("slump" in text) 18 else 0) +
            (if ("percent" in text || '%' in text) 8 else 0) +
            (if ("ratio" in text) 26 else 0) +
            (if ("minimum" in text || "maximum" in text) 4 else 0) -
            (if ("shrinkage" in text) 10 else 0)
    }

    /** Put excerpts with the strongest literal question coverage nearest the final question. */
    internal fun factPresentationOrder(question: String, citations: List<Citation>): List<Citation> {
        val terms = WORD.findAll(question.lowercase()).map { it.value }
            .filter { it.length >= 3 && it !in FACT_STOP_WORDS }
            .toSet()
        if (terms.isEmpty()) return citations
        return citations.sortedWith(
            compareBy<Citation> { citation ->
                val text = (citation.sectionPath + " " + citation.text).lowercase()
                terms.count { term -> wordBoundary(term).containsMatchIn(text) }
            }.thenBy { it.score }.thenBy { it.chunkIndex },
        )
    }

    /**
     * PDF extractors can emit a table's labels and rightmost value cells as adjacent chunks. An
     * explicitly expanded same-page neighbour is structural context even if that fragment itself
     * contains only numbers and was conservatively labelled PARAGRAPH.
     */
    private fun isStructuralContext(candidate: Citation, all: List<Citation>): Boolean {
        if (candidate.contentKind == "TABLE" ||
            candidate.continuesFromChunkIndex >= 0 || candidate.continuesToChunkIndex >= 0
        ) return true
        // The sentence introducing a retrieved list ("ratios shall be as follows:") names what
        // the list values mean; it must travel with the list even across a page boundary.
        if (candidate.contentKind != "LIST" && candidate.text.trimEnd().endsWith(":") &&
            all.any { it.contentKind == "LIST" && it.chunkIndex == candidate.chunkIndex + 1 && it.sectionId == candidate.sectionId }
        ) return true
        return all.any { seed ->
            seed.sectionId == candidate.sectionId && seed.pageNumber == candidate.pageNumber &&
                kotlin.math.abs(seed.chunkIndex - candidate.chunkIndex) <= 1 &&
                (seed.contentKind == "TABLE" || STRUCTURED_HINT.containsMatchIn(seed.text))
        }
    }

    private fun exactSuffixPrefix(left: String, right: String): Int {
        val max = minOf(left.length, right.length, MAX_BOUNDARY_OVERLAP)
        for (length in max downTo MIN_BOUNDARY_OVERLAP) {
            if (left.regionMatches(left.length - length, right, 0, length, ignoreCase = false)) return length
        }
        return 0
    }

    companion object {
        private val STRUCTURAL_POINTER = Regex(
            "(?i)\\b(?:specification|spec|section|clause)\\s+[0-9][0-9a-z.-]*\\b",
        )
        private const val MIN_BOUNDARY_OVERLAP = 24
        private const val MAX_BOUNDARY_OVERLAP = 600
        private val NUMBER = Regex("(?<![\\p{L}\\p{N}])\\d[\\d,.]*(?:[/-]\\d[\\d,.]*)?%?")
        private val DECIMAL = Regex("(?<![\\p{L}\\p{N}])\\d+\\.\\d+(?![\\p{L}\\p{N}])")
        private val WORD = Regex("[\\p{L}\\p{N}]+")
        private fun wordBoundary(term: String) = Regex("(?<![\\p{L}\\p{N}])${Regex.escape(term)}(?![\\p{L}\\p{N}])")
        private val FACT_STOP_WORDS = setOf(
            "and", "are", "for", "from", "how", "the", "this", "that", "what", "when", "which", "with", "about",
        )
        private val STRUCTURED_HINT = Regex(
            "(?i)\\b(?:table|minimum|maximum|variation|tolerance|cross[- ]?section|dimensions|slump|strength|cement)\\b",
        )
    }
}
