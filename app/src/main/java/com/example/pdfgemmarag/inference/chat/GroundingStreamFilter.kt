package com.example.pdfgemmarag.inference.chat

import com.example.pdfgemmarag.core.model.Citation
import java.text.Normalizer
import java.util.Locale

/**
 * Bounded incremental filter for generated numeric/identifier values and excerpt citations.
 * Construct and call it in the inference process; it has no Android/UI dependency.
 */
class GroundingStreamFilter(
    question: String,
    private val excerpts: List<Citation>,
    structuralValues: List<String> = emptyList(),
) {
    private val pending = StringBuilder()
    // The question is intentionally not authoritative evidence: repeating a value or identifier
    // suggested by the user must not make it grounded. QueryPlanner passes only manifest-validated
    // structural pointers through [structuralValues].
    @Suppress("UNUSED_VARIABLE")
    private val originalQuestion = question
    private val evidenceText =
        excerpts.joinToString("\n") {
            listOf(
                it.text,
                it.specificationNumber,
                it.sectionNumber,
                it.sectionPath,
                it.pageNumber.toString(),
            ).joinToString("\n")
        } + "\n" + structuralValues.joinToString("\n")
    private val lexicon = EvidenceValueLexer.lex(evidenceText)
    private val allowedValues = lexicon.values
    private val allowedSpecifications = excerpts.map { normalizeValue(it.specificationNumber) }.filter(String::isNotBlank).toSet()
    private val allowedSections = excerpts.map { normalizeValue(it.sectionNumber) }.filter(String::isNotBlank).toSet()
    private val allowedStructuralValues = (
        structuralValues + excerpts.flatMap { listOf(it.specificationNumber, it.sectionNumber) }
        ).map(::normalizeValue).filter(String::isNotBlank).toSet()
    private val allowedValueUnits = lexicon.valueUnits
    private val allowedIdentifiers = lexicon.identifiers
    private val excerptById = excerpts.associateBy { it.excerptId }
    private val usedIds = LinkedHashSet<String>()
    private val emittedTail = StringBuilder()
    private val rejections = ArrayList<String>()
    private var atLineStart = true
    var hadGroundingFailure: Boolean = false
        private set

    val usedCitations: List<Citation> get() = usedIds.mapNotNull(excerptById::get)

    /** Stage-attribution reasons (`VALUE_ABSENT`, `UNIT_MISMATCH`, `IDENTIFIER_MISMATCH`, `CITATION`) for every rejection. */
    val groundingReasons: List<String> get() = rejections.toList()

    fun accept(fragment: String): String {
        if (fragment.isEmpty()) return ""
        pending.append(fragment)
        require(pending.length <= MAX_PENDING + fragment.length) { "Grounding stream buffer exceeded its bound" }
        return drain(final = false)
    }

    fun finish(): String = drain(final = true)

    private fun drain(final: Boolean): String {
        val output = StringBuilder()
        var cursor = 0
        while (cursor < pending.length) {
            val ch = pending[cursor]
            if (ch == '[') {
                val close = pending.indexOf("]", cursor + 1)
                if (close < 0) {
                    if (!final && pending.length - cursor <= MAX_CITATION_LENGTH) break
                    // Never expose a half-written internal excerpt marker such as "[E". It is
                    // neither a valid user citation nor safe text to feed into the numeric scanner.
                    reject("CITATION:" + pending.substring(cursor).take(16))
                    cursor = pending.length
                    continue
                }
                val marker = pending.substring(cursor + 1, close).trim()
                // Small models sometimes suffix a list-item letter to an excerpt id ("E5a"); the
                // excerpt is still the evidence, the suffix is dropped.
                val markerIds = marker.split(',').map { id ->
                    val trimmed = id.trim()
                    val suffixed = SUFFIXED_EXCERPT.matchEntire(trimmed)
                    val base = suffixed?.groupValues?.get(1)
                    if (trimmed !in excerptById && base != null && base in excerptById) base else trimmed
                }.filter(String::isNotEmpty)
                val valid = markerIds.filter { it in excerptById }
                val citations = valid.mapNotNull(excerptById::get)
                if (valid.isNotEmpty()) {
                    // A long combined marker may include one invented id; the valid excerpts are
                    // still real evidence and are shown, the invented one produces no page.
                    usedIds += valid
                    citations.map { it.pageNumber }.distinct().forEach { page ->
                        output.append("[Page ").append(page).append(']')
                    }
                    if (valid.size != markerIds.size) reject("CITATION_PARTIAL:" + (markerIds - valid.toSet()).joinToString(","))
                } else {
                    reject("CITATION:$marker")
                }
                cursor = close + 1
                continue
            }
            if (ch.isDigit() || ((ch == '-' || ch == '+') && cursor + 1 < pending.length && pending[cursor + 1].isDigit())) {
                var end = cursor + 1
                while (end < pending.length && isValueChar(pending[end])) end++
                if (end == pending.length && !final && end - cursor <= MAX_VALUE_LENGTH) break
                val raw = pending.substring(cursor, end)
                if (isListMarker(raw, output)) {
                    output.append(raw)
                } else {
                    val normalized = normalizeValue(raw)
                    val identifierPrefix = precedingLetters(output)
                    val identifierAllowed = identifierPrefix.isNotEmpty() &&
                        normalizeIdentifier(identifierPrefix + normalized) in allowedIdentifiers
                    val unit = followingUnit(end, final)
                    if (unit == WAIT_FOR_UNIT) break
                    val structuralCorrection = structuralCorrection(normalized, output)
                    val replacement = when {
                        identifierAllowed -> raw
                        structuralCorrection != null -> preserveWrapping(raw, structuralCorrection)
                        // QueryPlanner only supplies pointers that were validated against the
                        // manifest. Treat them as authoritative even when nearby generated prose
                        // makes the numeric scanner mistake the following word for a unit.
                        normalized in allowedStructuralValues -> raw
                        // Extracted tables often split a numeric cell from its column-unit header.
                        // A value that is present verbatim but has no explicit adjacent source unit
                        // remains grounded. Values with an explicit source unit still receive the
                        // stricter unit check below (for example, 40 degrees must not become 40 psi).
                        normalized in allowedValues && allowedValueUnits.none { it.first == normalized } -> raw
                        unit != null && allowedValueUnits.asSequence()
                            .filter { it.second == unit }
                            .map { it.first }
                            .distinct()
                            .take(2)
                            .count() >= 2 -> {
                            if ((normalized to unit) in allowedValueUnits) raw
                            else allowedValueUnits
                                .asSequence()
                                .filter { it.second == unit && it.first.startsWith(normalized) }
                                .map { it.first }
                                .distinct()
                                .singleOrNull()
                                ?.let { preserveWrapping(raw, it) }
                        }
                        normalized.isBlank() || normalized in allowedValues -> raw
                        else -> uniquePrefix(normalized)?.let { source -> preserveWrapping(raw, source) }
                    }
                    if (replacement == null) {
                        val reason = when {
                            unit != null && allowedValueUnits.any { it.second == unit } -> "UNIT_MISMATCH:$raw $unit"
                            identifierPrefix.isNotEmpty() -> "IDENTIFIER_MISMATCH:$identifierPrefix$raw"
                            else -> "VALUE_ABSENT:$raw"
                        }
                        reject(reason)
                        output.append("[unverified value]")
                    } else output.append(replacement)
                }
                cursor = end
                continue
            }
            output.append(ch)
            cursor++
        }
        pending.delete(0, cursor)
        if (final && pending.isNotEmpty()) {
            output.append(pending)
            pending.setLength(0)
        }
        if (output.isNotEmpty()) atLineStart = output.last() == '\n'
        if (output.isNotEmpty()) {
            emittedTail.append(output)
            if (emittedTail.length > EMITTED_TAIL_LENGTH) {
                emittedTail.delete(0, emittedTail.length - EMITTED_TAIL_LENGTH)
            }
        }
        return output.toString()
    }

    private fun reject(reason: String) {
        hadGroundingFailure = true
        if (rejections.size < MAX_REASONS) rejections += reason
    }

    /**
     * Waits briefly for a unit so a valid source value cannot be attached to the wrong unit. The
     * unit may be attached (`8m`) or separated by spaces (`8 m`); an attached run that does not
     * look like a unit (the `B` of `03210B`) is an identifier suffix and yields no unit.
     */
    private fun followingUnit(valueEnd: Int, final: Boolean): String? {
        val rawValue = pending.substring(0, valueEnd).takeLastWhile(::isValueChar)
        if (rawValue.endsWith('.') || rawValue.endsWith(',') || rawValue.endsWith('%')) return null
        val suffix = pending.substring(valueEnd)
        if (!final && suffix.length <= UNIT_LOOKAHEAD && suffix.none { it == '\n' || it in ".;:,[]()" }) {
            val match = FOLLOWING_UNIT.find(suffix)
            if (match == null || match.range.last == suffix.lastIndex) return WAIT_FOR_UNIT
        }
        val match = FOLLOWING_UNIT.find(suffix) ?: return null
        val unit = match.groupValues[1]
        val attached = match.range.first == 0 && !suffix.first().isWhitespace()
        if (!EvidenceValueLexer.isUnitShape(unit, attached)) return null
        return unit.lowercase(Locale.ROOT)
    }

    /** Letters immediately before the value, tolerating one space or hyphen (`IS 3764`, `Rule- 210`). */
    private fun precedingLetters(output: StringBuilder): String {
        val available = if (output.isNotEmpty()) output.toString() else emittedTail.toString()
        val trimmed = if (available.isNotEmpty() && (available.last() == ' ' || available.last() == '-')) {
            available.dropLast(1)
        } else available
        return trimmed.takeLastWhile { it.isLetter() }.takeLast(MAX_IDENTIFIER_PREFIX)
    }

    private fun structuralCorrection(value: String, output: StringBuilder): String? {
        val available = (if (output.isNotEmpty()) output.toString() else emittedTail.toString()).trimEnd()
        val word = available.takeLastWhile { it.isLetter() }.lowercase(Locale.ROOT)
        val candidates = when (word) {
            "specification", "spec" -> allowedSpecifications
            "section", "clause" -> allowedSections
            else -> return null
        }
        if (value in candidates) return null
        return candidates.singleOrNull()
    }

    private fun uniquePrefix(value: String): String? {
        if (value.length < 2) return null
        val candidates = allowedValues.filter { it.startsWith(value) && sameForm(value, it) }
        return candidates.singleOrNull()
    }

    private fun sameForm(a: String, b: String): Boolean =
        a.firstOrNull() == b.firstOrNull() && a.count { it == '.' } == b.count { it == '.' }

    private fun preserveWrapping(raw: String, corrected: String): String {
        val prefix = raw.takeWhile { it == '+' || it == '-' }
        val suffix = raw.takeLastWhile { it == '%' || it == ',' || it == '.' }
        return prefix + corrected + suffix.takeUnless { corrected.endsWith(it) }.orEmpty()
    }

    private fun isListMarker(raw: String, emitted: StringBuilder): Boolean {
        val lineStart = if (emitted.isEmpty()) atLineStart else emitted.last() == '\n'
        return lineStart && raw.matches(Regex("\\d+[.)]"))
    }

    private fun isValueChar(ch: Char): Boolean = ch.isDigit() || ch in ".,%/+-–—"

    companion object {
        private const val MAX_PENDING = 128
        private const val MAX_VALUE_LENGTH = 64
        private const val MAX_CITATION_LENGTH = 128
        private const val UNIT_LOOKAHEAD = 16
        private const val EMITTED_TAIL_LENGTH = 24
        private const val MAX_IDENTIFIER_PREFIX = 8
        private const val MAX_REASONS = 16
        private const val WAIT_FOR_UNIT = "\u0000"
        private val SUFFIXED_EXCERPT = Regex("^(E\\d+)[A-Za-z]$")
        private val FOLLOWING_UNIT = Regex("^[ \\t]*([\\p{L}°µ][\\p{L}°µ/]{0,11})")

        internal fun valueTokens(text: String): Set<String> = EvidenceValueLexer.lex(text).values

        internal fun valueUnitTokens(text: String): Set<Pair<String, String>> = EvidenceValueLexer.lex(text).valueUnits

        internal fun identifierTokens(text: String): Set<String> = EvidenceValueLexer.lex(text).identifiers

        /** Stream-side comparison form; a leading `+`/`±` is presentation, so `+1/2` matches source `1/2`. */
        internal fun normalizeValue(value: String): String =
            EvidenceValueLexer.normalizeNumber(value).trimStart('+', '±')

        private fun normalizeIdentifier(value: String): String = EvidenceValueLexer.normalizeIdentifier(value)
    }
}
