package com.example.pdfgemmarag.inference.pdf

import java.text.Normalizer
import java.util.Locale
import kotlin.math.abs

/**
 * Converts page geometry into headings, logical lists, paragraphs and conservative tables.
 * Detection is deliberately deterministic and document-agnostic: no test-document words appear
 * here. Heading candidates are removed before table clustering so numbered clauses are not
 * mistaken for two-column data tables.
 */
class StructureAnalyzer(
    private val tables: TableClusterer = TableClusterer(),
) {
    fun analyse(page: PageLayout): PageContent {
        if (page.lines.isEmpty()) return PageContent(page.pageNumber, emptyList())
        val medianHeight = median(page.lines.map { it.lineHeight }).coerceAtLeast(1f)
        val segments = ArrayList<Segment>()
        var plainStart = 0
        var i = 0

        fun flushPlain(endExclusive: Int) {
            if (endExclusive <= plainStart) return
            val sub = page.copy(lines = page.lines.subList(plainStart, endExclusive))
            segments += tables.analyse(sub).segments
        }

        while (i < page.lines.size) {
            val heading = headingAt(page, i, medianHeight)
            if (heading != null) {
                flushPlain(i)
                segments += heading.segment
                i += heading.consumed
                plainStart = i
                continue
            }

            val firstItem = listItem(page.lines[i].text)
            if (firstItem != null) {
                flushPlain(i)
                val items = ArrayList<Segment.ListItem>()
                var j = i
                while (j < page.lines.size) {
                    val line = page.lines[j]
                    val item = listItem(line.text)
                    if (item != null) {
                        items += Segment.ListItem(item.first, item.second)
                        j++
                        continue
                    }
                    val previousLine = page.lines[j - 1]
                    val wrapsPrevious = items.isNotEmpty() && !isExplicitHeading(line.text.trim()) &&
                        line.indent > page.lines[i].indent + page.width * 0.015f &&
                        line.box.top - previousLine.box.bottom <= medianHeight * 1.2f
                    if (!wrapsPrevious) break
                    val previousItem = items.removeAt(items.lastIndex)
                    items += previousItem.copy(text = previousItem.text + " " + line.text.trim())
                    j++
                }
                // One enumerated clause is still a logical list item; retaining the label prevents
                // the chunker from separating it from its requirement.
                segments += Segment.ListBlock(page.pageNumber, items)
                i = j
                plainStart = i
                continue
            }
            i++
        }
        flushPlain(page.lines.size)
        return PageContent(page.pageNumber, segments)
    }

    private data class HeadingMatch(val segment: Segment.Heading, val consumed: Int)

    private fun headingAt(page: PageLayout, index: Int, medianHeight: Float): HeadingMatch? {
        val first = page.lines[index]
        val parsed = parseHeading(first.text) ?: return null
        val parts = ArrayList<String>()
        if (parsed.title.isNotBlank()) parts += parsed.title
        var consumed = 1
        var previous = first
        while (index + consumed < page.lines.size && consumed < MAX_HEADING_LINES) {
            val next = page.lines[index + consumed]
            if (!isHeadingContinuation(previous, next, page.width, medianHeight)) break
            parts += next.text.trim()
            previous = next
            consumed++
        }
        // Some text PDFs preserve a visibly separated, centered specification title as ordinary
        // text (no bold/font metadata). Associate that one title line with a bare SECTION number.
        if (parsed.specificationNumber != null && parts.isEmpty() && index + consumed < page.lines.size) {
            val candidate = page.lines[index + consumed]
            val gap = candidate.box.top - previous.box.bottom
            val title = candidate.text.trim()
            if (gap in -medianHeight * 0.4f..medianHeight * SPEC_TITLE_MAX_GAP_RATIO &&
                title.length in 3..MAX_TITLE_CHARS &&
                !BODY_END.containsMatchIn(title) &&
                !isExplicitHeading(title) &&
                looksLikeTitle(title)
            ) {
                parts += title
                consumed++
            }
        }
        val combinedTitle = parts.joinToString(" ").replace(SPACES, " ").trim()
        if (parsed.specificationNumber != null) {
            val headingLines = page.lines.subList(index, index + consumed)
            val hasBoundaryStyle = headingLines.any { it.isBold } ||
                headingLines.any { it.lineHeight >= medianHeight * SPEC_HEADING_HEIGHT_RATIO } ||
                isUppercaseTitle(combinedTitle)
            val hasSectionStartSequence = page.lines
                .drop(index + consumed)
                .take(MAX_SECTION_START_LOOKAHEAD)
                .any { isPartOneRoot(it.text) }
            val isNearPageStart = first.box.top <= page.height * SPEC_HEADING_TOP_FRACTION
            val hasSeparateLeadingTitle = parsed.title.isBlank() && combinedTitle.isNotBlank()
            // Construction specifications commonly list other "SECTION 01234 ..." entries as
            // cross-references. A real boundary is normally styled, starts near the top of its
            // page with a separate title, or is followed immediately by the PART 1 / 1.0 root.
            // These layout/sequence signals work when extraction exposes no font metadata.
            if (!hasBoundaryStyle && !hasSectionStartSequence &&
                !(isNearPageStart && hasSeparateLeadingTitle)
            ) return null
        }
        return HeadingMatch(
            Segment.Heading(
                pageNumber = page.pageNumber,
                number = parsed.number,
                title = combinedTitle.ifBlank { "SECTION ${parsed.specificationNumber}" },
                level = parsed.level,
                specificationNumber = parsed.specificationNumber,
            ),
            consumed,
        )
    }

    private data class ParsedHeading(
        val number: String?,
        val title: String,
        val level: Int,
        val specificationNumber: String? = null,
    )

    private fun parseHeading(raw: String): ParsedHeading? {
        val text = raw.trim().replace(SPACES, " ")
        SPEC_HEADING.matchEntire(text)?.let { m ->
            val id = m.groupValues[1]
            val title = m.groupValues[2].trim()
            return ParsedHeading(id, title, 1, id)
        }
        PART_HEADING.matchEntire(text)?.let { m ->
            return ParsedHeading("PART ${m.groupValues[1]}", m.groupValues[2].trim(), 2)
        }
        NUMBERED_HEADING.matchEntire(text)?.let { m ->
            val number = m.groupValues[1]
            val title = m.groupValues[2].trim()
            if (title.length in 2..MAX_TITLE_CHARS && looksLikeTitle(title)) {
                return ParsedHeading(number, title, 3 + number.count { it == '.' }.coerceAtMost(3))
            }
        }
        if (text.length in 3..MAX_TITLE_CHARS &&
            !BODY_END.containsMatchIn(text) &&
            isUppercaseTitle(text)
        ) {
            return ParsedHeading(null, text, 3)
        }
        return null
    }

    private fun isHeadingContinuation(previous: LineBox, next: LineBox, pageWidth: Float, medianHeight: Float): Boolean {
        val verticalGap = next.box.top - previous.box.bottom
        if (verticalGap < -medianHeight * 0.4f || verticalGap > medianHeight * 1.2f) return false
        if (abs(next.lineHeight - previous.lineHeight) > medianHeight * 0.35f) return false
        if (abs(next.indent - previous.indent) > pageWidth * 0.08f) return false
        val text = next.text.trim()
        if (text.length !in 2..MAX_TITLE_CHARS || BODY_END.containsMatchIn(text)) return false
        if (listItem(text) != null || isExplicitHeading(text)) return false
        val compactGeometry = next.box.width < pageWidth * 0.72f && verticalGap <= medianHeight * 0.55f
        return isUppercaseTitle(text) || next.isBold || previous.isBold || compactGeometry
    }

    private fun isExplicitHeading(text: String): Boolean =
        SPEC_HEADING.matches(text) || PART_HEADING.matches(text) || NUMBERED_HEADING.matches(text)

    private fun isPartOneRoot(raw: String): Boolean {
        val text = raw.trim().replace(SPACES, " ")
        return PART_ONE_ROOT.matches(text) || NUMBERED_PART_ONE_ROOT.matches(text)
    }

    private fun looksLikeTitle(text: String): Boolean {
        if (BODY_END.containsMatchIn(text)) return false
        val letters = text.count { it.isLetter() }
        if (letters < 2) return false
        return isUppercaseTitle(text) || text.split(' ').count { it.isNotBlank() } <= 12
    }

    private fun isUppercaseTitle(text: String): Boolean {
        val letters = text.filter { it.isLetter() }
        return letters.length >= 2 && letters.count { it.isUpperCase() } * 10 >= letters.length * 8
    }

    private fun listItem(text: String): Pair<String, String>? {
        val match = LIST_ITEM.matchEntire(text.trim()) ?: return null
        return match.groupValues[1] to match.groupValues[2].trim()
    }

    private fun median(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sorted()
        return sorted[sorted.size / 2]
    }

    companion object {
        private const val MAX_HEADING_LINES = 3
        private const val MAX_SECTION_START_LOOKAHEAD = 6
        private const val MAX_TITLE_CHARS = 140
        private const val SPEC_HEADING_HEIGHT_RATIO = 1.15f
        private const val SPEC_HEADING_TOP_FRACTION = 0.18f
        private const val SPEC_TITLE_MAX_GAP_RATIO = 2.0f
        private val SPACES = Regex("[\\s\\u00a0]+")
        private val BODY_END = Regex("[.;:]\\s*$")
        private val SPEC_HEADING = Regex("(?i)^SECTION\\s+([0-9]{3,8}(?:[-.]?[0-9A-Z]+)?)\\s*[-–—:]?\\s*(.*)$")
        private val PART_HEADING = Regex("(?i)^PART\\s+([0-9IVX]+)\\s*[-–—:]?\\s*(.*)$")
        private val PART_ONE_ROOT = Regex("(?i)^PART\\s+(?:1|I)(?:\\s*[-–—:]\\s*|\\s+).+$")
        private val NUMBERED_PART_ONE_ROOT = Regex("(?i)^1\\.0+\\s+.+$")
        private val NUMBERED_HEADING = Regex("^((?:[0-9]+\\.)+[0-9A-Z]+)\\s+(.+)$")
        private val LIST_ITEM = Regex("^((?:[A-Za-z]|[0-9]+)[.)]|\\([A-Za-z0-9]+\\))\\s+(.+)$")

        fun normalizeHeading(text: String): String = Normalizer.normalize(text, Normalizer.Form.NFKC)
            .lowercase(Locale.ROOT)
            .replace(Regex("[^\\p{L}\\p{N}]+"), " ")
            .trim()
            .replace(Regex("\\s+"), " ")
    }
}
