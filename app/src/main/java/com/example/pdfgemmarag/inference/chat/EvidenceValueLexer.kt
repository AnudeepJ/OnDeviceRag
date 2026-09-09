package com.example.pdfgemmarag.inference.chat

import java.text.Normalizer
import java.util.Locale

enum class EvidenceValueKind { NUMBER, PERCENT, FRACTION, RANGE, RATIO, MEASUREMENT, STANDARD_ID, COMPOUND_ID }

/** One typed value observed in evidence. [normalized] is the comparison form; [raw] keeps the source spelling. */
data class EvidenceValue(
    val kind: EvidenceValueKind,
    val raw: String,
    val normalized: String,
    /** Lower-cased unit for [EvidenceValueKind.MEASUREMENT]; null otherwise. */
    val unit: String? = null,
)

/**
 * Typed value grammar shared by grounding and the deterministic leads.
 *
 * The lexer is pure and document-agnostic: it recognises the *shape* of numbers, ranges, ratios,
 * fractions, measurements and identifiers, never a whitelist of units or values. A unit is any short
 * letter token that appears attached (`8m`) or space-separated (`8 m`) after a number in evidence;
 * the generated answer can only reuse pairs that were observed this way.
 */
object EvidenceValueLexer {

    /** Sets derived from one evidence text, in the exact forms [GroundingStreamFilter] compares against. */
    data class Lexicon(
        val tokens: List<EvidenceValue>,
        /** Normalized numeric forms: whole tokens plus every component (range ends, ratio parts, mixed fractions). */
        val values: Set<String>,
        /** (normalized value, lower-case unit) pairs observed adjacent in evidence. */
        val valueUnits: Set<Pair<String, String>>,
        /** Letter-prefixed identifiers such as `astmc150`, `is3764:1992`, `rule210`. */
        val identifiers: Set<String>,
    )

    fun lex(text: String): Lexicon {
        val normalizedText = normalizeText(text)
        val tokens = ArrayList<EvidenceValue>()
        val values = LinkedHashSet<String>()
        val valueUnits = LinkedHashSet<Pair<String, String>>()
        val identifiers = LinkedHashSet<String>()

        for (match in IDENTIFIER.findAll(normalizedText)) {
            val raw = match.value
            val identifier = normalizeIdentifier(raw)
            identifiers += identifier
            // "ASTM C150" is also referred to as "C150"; the stream sees only the letters adjacent to the digits.
            IDENTIFIER_TAIL.find(raw)?.let { identifiers += normalizeIdentifier(it.value) }
            // "IS 3764:1992" is also valid as "IS 3764"; the year is a component in its own right.
            val colon = identifier.indexOf(':')
            if (colon > 0) identifiers += identifier.substring(0, colon)
            tokens += EvidenceValue(EvidenceValueKind.STANDARD_ID, raw, identifier)
            NUMBER_CORE.findAll(raw).drop(1).forEach { values += normalizeNumber(it.value) }
            // Year or revision suffixes are plain numbers for the stream: "3764:1992" -> 1992.
            raw.split(':', '-', '/').drop(1).forEach { part ->
                NUMBER_CORE.find(part)?.let { values += normalizeNumber(it.value) }
            }
        }

        for (match in VALUE.findAll(normalizedText)) {
            val raw = match.value.trim()
            val unitRaw = match.groups[UNIT_GROUP]?.value
            val core = match.groups[CORE_GROUP]?.value ?: raw
            val normalizedCore = normalizeNumber(core)
            if (normalizedCore.isBlank()) continue
            val attached = unitRaw != null &&
                match.groups[UNIT_GROUP]!!.range.first == match.groups[CORE_GROUP]!!.range.last + 1
            val unit = unitRaw?.takeIf { isUnitShape(it, attached) }?.lowercase(Locale.ROOT)
            val kind = when {
                ':' in core -> EvidenceValueKind.RATIO
                RANGE_SEPARATOR.containsMatchIn(core) -> EvidenceValueKind.RANGE
                '/' in core -> EvidenceValueKind.FRACTION
                core.endsWith('%') -> EvidenceValueKind.PERCENT
                LEADING_ZERO_ID.matches(core) -> EvidenceValueKind.COMPOUND_ID
                unit != null -> EvidenceValueKind.MEASUREMENT
                else -> EvidenceValueKind.NUMBER
            }
            tokens += EvidenceValue(kind, raw, normalizedCore, unit)
            values += normalizedCore
            // A sign is presentation: source "±1/2" supports a generated "1/2" and "+1/2".
            values += normalizedCore.trimStart('-', '+', '±')
            for (component in components(core)) values += component
            if (unit != null) {
                valueUnits += normalizedCore to unit
                for (component in components(core)) valueUnits += component to unit
            }
        }
        return Lexicon(tokens, values, valueUnits, identifiers)
    }

    /** Component values a model may legitimately restate on their own: range ends, ratio parts, mixed-fraction parts. */
    internal fun components(core: String): List<String> {
        val normalized = normalizeNumber(core)
        if (normalized.isBlank()) return emptyList()
        val out = LinkedHashSet<String>()
        // Normalized form already uses '-' for ranges and ':' for ratios; mixed fractions keep one space.
        val parts = normalized.split('-', ':').map(String::trim).filter(String::isNotBlank)
        if (parts.size > 1) parts.forEach { part -> out += part.removeSuffix("%") }
        // "2 1/2" -> "2", "1/2"; "1 1/2:1" -> "1 1/2", "1"
        for (part in parts) {
            if (part.contains(' ')) part.split(' ').filter(String::isNotBlank).forEach { out += it }
        }
        out.remove(normalized)
        return out.toList()
    }

    /** NFKC, vulgar fractions to ASCII, dash variants to '-', collapsed whitespace. */
    fun normalizeText(text: String): String {
        val sb = StringBuilder(text.length + 8)
        for (ch in text) {
            val fraction = VULGAR_FRACTIONS[ch]
            if (fraction != null) {
                if (sb.isNotEmpty() && sb.last().isDigit()) sb.append(' ')
                sb.append(fraction)
            } else sb.append(ch)
        }
        return Normalizer.normalize(sb, Normalizer.Form.NFKC)
            .replace('\u2044', '/')
            .replace(DASHES, "-")
            .replace('\u00a0', ' ')
    }

    /** Comparison form: lower-case, no thousands separators, trailing punctuation removed, dashes unified. */
    fun normalizeNumber(value: String): String {
        var out = normalizeText(value).trim().lowercase(Locale.ROOT).replace(",", "")
        out = out.replace(Regex("\\s*-\\s*"), "-").replace(Regex("\\s*:\\s*"), ":").replace(Regex("\\s+to\\s+"), "-")
        out = out.removeSuffix("%").trimEnd('.', ',', '-', '+')
        return out.replace(Regex("\\s+"), " ")
    }

    fun normalizeIdentifier(value: String): String = normalizeText(value)
        .lowercase(Locale.ROOT)
        .replace(Regex("[\\s-]+"), "")
        .trimEnd('.', ',', ':')

    /**
     * A compact unit is letters attached to the number. Mixed-case forms such as `Mt` are units;
     * a single trailing capital such as the `B` in `03210B` is an identifier suffix. Spaced units
     * keep the historical loose definition (any short letter word) because they are only used to
     * detect unit mismatches, never to admit a value.
     */
    fun isUnitShape(unit: String, attached: Boolean): Boolean {
        if (unit.isEmpty() || unit.length > MAX_UNIT_LENGTH) return false
        if (!attached) return true
        return unit.any { it.isLowerCase() || it in UNIT_SYMBOLS }
    }

    private const val CORE_GROUP = 1
    private const val UNIT_GROUP = 2
    private const val MAX_UNIT_LENGTH = 12
    private const val UNIT_SYMBOLS = "°µ²³%"
    private val DASHES = Regex("[\\u2010\\u2011\\u2012\\u2013\\u2014\\u2212]")
    private val RANGE_SEPARATOR = Regex("-|\\s+to\\s+")
    private val LEADING_ZERO_ID = Regex("0\\d{2,}[a-z]?")
    private val NUMBER_CORE = Regex("\\d[\\d,]*(?:\\.\\d+)?")
    private val VULGAR_FRACTIONS = mapOf(
        '½' to "1/2", '⅓' to "1/3", '⅔' to "2/3", '¼' to "1/4", '¾' to "3/4",
        '⅕' to "1/5", '⅖' to "2/5", '⅗' to "3/5", '⅘' to "4/5", '⅙' to "1/6", '⅚' to "5/6",
        '⅛' to "1/8", '⅜' to "3/8", '⅝' to "5/8", '⅞' to "7/8",
    )

    /**
     * Group 1: numeric core — signed number, optional decimals, optional mixed fraction (`2 1/2`),
     * optional range (`21-25`, `21 to 25`) or ratio (`1 1/2:1`), optional percent.
     * Group 2: optional unit — attached or after whitespace; letters, degree, micro, superscripts, slash.
     */
    private val VALUE = Regex(
        "(?<![\\p{L}\\p{N}/.])" +
            "([-+±]?\\d[\\d,]*(?:\\.\\d+)?(?:\\s\\d+/\\d+|/\\d+)?" +
            "(?:\\s*(?:-|to)\\s*\\d[\\d,]*(?:\\.\\d+)?(?:\\s\\d+/\\d+|/\\d+)?)?" +
            "(?:\\s*:\\s*\\d[\\d,]*(?:\\.\\d+)?(?:\\s\\d+/\\d+|/\\d+)?)?%?)" +
            "(?:[ \\t]*([\\p{L}°µ][\\p{L}°µ²³/]{0,11}))?" +
            "(?![\\p{N}])",
    )
    /** `IS 3764:1992`, `ASTM C150`, `Rule- 210`, `Class A`-style prefixes with an optional second letter group. */
    private val IDENTIFIER = Regex("(?i)\\b[\\p{L}]{1,8}(?:[-\\s]{1,2}[\\p{L}]{1,3})?[-\\s]{0,2}\\d[\\p{L}\\p{N}./:-]*\\b")
    private val IDENTIFIER_TAIL = Regex("(?i)[\\p{L}]{1,8}[-\\s]{0,2}\\d[\\p{L}\\p{N}./:-]*$")
}
