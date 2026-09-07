package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.inference.pdf.StructureAnalyzer
import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.SectionRecord
import java.text.Normalizer
import java.util.Locale

enum class QuestionIntent { FACT, SECTION_SUMMARY, DOCUMENT_OVERVIEW, AMBIGUOUS_SECTION }

data class QuestionPlan(
    val intent: QuestionIntent,
    val subjectText: String,
    val explicitSpecificationNumber: String? = null,
    val explicitSectionNumber: String? = null,
    val resolvedSectionId: String? = null,
    val candidateSections: List<SectionRecord> = emptyList(),
    val sectionConfidence: Double = 0.0,
    val inheritedSectionId: String? = null,
) {
    fun clarification(): String {
        require(intent == QuestionIntent.AMBIGUOUS_SECTION)
        return buildString {
            append("I found more than one matching section. Which one do you mean?\n")
            candidateSections.forEach { section ->
                append("• ")
                if (section.specificationNumber.isNotBlank()) append("Specification ${section.specificationNumber}, ")
                if (section.sectionNumber.isNotBlank()) append("section ${section.sectionNumber}, ")
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
        val summary = SUMMARY.containsMatchIn(normalized)
        val spec = SPEC_REFERENCE.find(normalized)?.groupValues?.get(1)
            ?: LEADING_ZERO_ID.find(normalized)?.value
        val section = SECTION_REFERENCE.find(normalized)?.groupValues?.get(1)
            ?: DOTTED_ID.find(normalized)?.value
        // "Specification 03300" identifies one specification; it is not a request to summarize
        // the entire document. Explicit identities always win over document-level vocabulary.
        val overview = summary && spec == null && section == null && DOCUMENT_WORDS.containsMatchIn(normalized)
        var subject = normalized
            .replace(SUMMARY, " ")
            .replace(SPEC_WORD, " ")
            .replace(SECTION_WORD, " ")
        if (spec != null) subject = subject.replace(spec, " ")
        if (section != null) subject = subject.replace(section, " ")
        subject = subject.replace(SPACES, " ").trim()

        if (manifest == null) {
            return QuestionPlan(if (overview) QuestionIntent.DOCUMENT_OVERVIEW else if (summary) QuestionIntent.SECTION_SUMMARY else QuestionIntent.FACT, subject)
        }

        val usable = manifest.sections.filter { it.title.isNotBlank() || it.sectionNumber.isNotBlank() }
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
            )
        }
        if (exactIdentity.size > 1 && summary) return ambiguous(subject, spec, section, exactIdentity, inheritedSectionId)

        val exactTitle = usable.filter { normalizeTitle(it.title) == subject && subject.isNotBlank() }
        if (exactTitle.size > 1) return ambiguous(subject, spec, section, exactTitle, inheritedSectionId)
        if (exactTitle.size == 1) {
            return QuestionPlan(
                if (summary) QuestionIntent.SECTION_SUMMARY else QuestionIntent.FACT,
                subject, spec, section, exactTitle.single().sectionId, exactTitle, 1.0, inheritedSectionId,
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
        if (summary && tied.size > 1) return ambiguous(subject, spec, section, tied, inheritedSectionId)
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
        )
    }

    private fun ambiguous(
        subject: String,
        spec: String?,
        section: String?,
        candidates: List<SectionRecord>,
        inherited: String?,
    ) = QuestionPlan(
        QuestionIntent.AMBIGUOUS_SECTION,
        subject,
        spec,
        section,
        candidateSections = candidates.sortedWith(compareBy<SectionRecord> { it.startPage }.thenBy { it.sectionNumber }),
        sectionConfidence = 1.0,
        inheritedSectionId = inherited,
    )

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
