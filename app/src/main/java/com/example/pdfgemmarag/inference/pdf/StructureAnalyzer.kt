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
        // Contents rows are navigational pointers, not section boundaries. Suppressing their
        // headings avoids matching a contents entry and the actual section as two candidates.
        val isContentsPage = isContentsPage(page)
        val segments = ArrayList<Segment>()
        var plainStart = 0
        var i = 0

        fun flushPlain(endExclusive: Int) {
            if (endExclusive <= plainStart) return
            val sub = page.copy(lines = page.lines.subList(plainStart, endExclusive))
            segments += tables.analyse(sub).segments
        }

        while (i < page.lines.size) {
            val heading = if (isContentsPage) null else headingAt(page, i, medianHeight)
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
                    val continuation = line.text.trim()
                    // Some PDF producers place a wrapped list line at the same left edge as the
                    // enumerator instead of indenting it beneath the item text. Accept that only
                    // when it continues in lower case and is geometrically adjacent; this keeps a
                    // following ordinary, capitalised paragraph out of the list.
                    val alignedWithList = line.indent >= page.lines[i].indent - page.width * 0.01f
                    val continuationStyle = continuation.firstOrNull()?.isLowerCase() == true ||
                        line.indent > page.lines[i].indent + page.width * 0.015f
                    val wrapsPrevious = items.isNotEmpty() && !isExplicitHeading(line.text.trim()) &&
                        alignedWithList && continuationStyle &&
                        line.box.top - previousLine.box.bottom <= medianHeight * 1.2f
                    if (!wrapsPrevious) break
                    val previousItem = items.removeAt(items.lastIndex)
                    items += previousItem.copy(text = previousItem.text + " " + continuation)
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
        // Some text PDFs preserve a visibly separated, centered title as ordinary text (no
        // bold/font metadata). Associate that one title line with a bare structural label such as
        // "SECTION 03300", "CHAPTER 13" or "APPENDIX A". The rule depends on position, gap and
        // title shape only, never on title words.
        if (parsed.isStructuralLabel && parts.isEmpty() && index + consumed < page.lines.size) {
            val candidate = page.lines[index + consumed]
            val gap = candidate.box.top - previous.box.bottom
            val title = candidate.text.trim()
            // Chapter/appendix labels are often separated from their title by a blank line.
            val maxGap = if (parsed.kind == Segment.Heading.KIND_SECTION) SPEC_TITLE_MAX_GAP_RATIO else LABEL_TITLE_MAX_GAP_RATIO
            if (gap in -medianHeight * 0.4f..medianHeight * maxGap &&
                title.length in 3..MAX_TITLE_CHARS &&
                !BODY_END.containsMatchIn(title) &&
                !isExplicitHeading(title) &&
                listItem(title) == null &&
                looksLikeTitle(title)
            ) {
                parts += title
                consumed++
            }
        }
        val combinedTitle = parts.joinToString(" ").replace(SPACES, " ").trim()
        if (parsed.kind == Segment.Heading.KIND_CHAPTER || parsed.kind == Segment.Heading.KIND_APPENDIX) {
            // A running "Chapter 13" cross-reference or a bare label with no title is not a
            // boundary. Real chapter starts carry a title (same line or adjacent) or a styled line.
            val headingLines = page.lines.subList(index, index + consumed)
            val styled = headingLines.any { it.isBold } ||
                headingLines.any { it.lineHeight >= medianHeight * SPEC_HEADING_HEIGHT_RATIO }
            if (combinedTitle.isBlank() && !styled) return null
            if (combinedTitle.isNotBlank() && !looksLikeTitle(combinedTitle)) return null
        }
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
        // Absent printed identifiers stay absent: the fallback title names the label itself.
        val fallbackTitle = when {
            parsed.specificationNumber != null -> "SECTION ${parsed.specificationNumber}"
            parsed.number != null -> parsed.number
            else -> ""
        }
        return HeadingMatch(
            Segment.Heading(
                pageNumber = page.pageNumber,
                number = parsed.number,
                title = combinedTitle.ifBlank { fallbackTitle },
                level = parsed.level,
                specificationNumber = parsed.specificationNumber,
                kind = parsed.kind,
                printedNumber = parsed.printedNumber,
            ),
            consumed,
        )
    }

    private data class ParsedHeading(
        val number: String?,
        val title: String,
        val level: Int,
        val specificationNumber: String? = null,
        val kind: String = Segment.Heading.KIND_HEADING,
        val printedNumber: String = number.orEmpty(),
    ) {
        /** Labels that commonly stand alone on a line with their title beneath them. */
        val isStructuralLabel: Boolean
            get() = kind == Segment.Heading.KIND_SECTION || kind == Segment.Heading.KIND_CHAPTER ||
                kind == Segment.Heading.KIND_PART || kind == Segment.Heading.KIND_APPENDIX
    }

    private fun parseHeading(raw: String): ParsedHeading? {
        val text = raw.trim().replace(SPACES, " ")
        SPEC_HEADING.matchEntire(text)?.let { m ->
            val id = m.groupValues[1]
            val title = m.groupValues[2].trim()
            return ParsedHeading(id, title, 1, id, Segment.Heading.KIND_SECTION, id)
        }
        // A sentence that merely begins with "Chapter 13 describes ..." is prose, not a boundary:
        // labels are short lines whose remainder has title shape.
        if (text.length <= MAX_LABEL_LINE_CHARS) {
            CHAPTER_HEADING.matchEntire(text)?.let { m ->
                val printed = arabic(m.groupValues[1])
                val title = m.groupValues[2].trim()
                if (title.isNotBlank() && !looksLikeTitle(title)) return null
                return ParsedHeading("CHAPTER ${m.groupValues[1]}", title, 1, null, Segment.Heading.KIND_CHAPTER, printed)
            }
            APPENDIX_HEADING.matchEntire(text)?.let { m ->
                val label = m.groupValues[1].uppercase(Locale.ROOT)
                val printed = arabic(m.groupValues[2])
                val title = m.groupValues[3].trim()
                if (title.isNotBlank() && !looksLikeTitle(title)) return null
                return ParsedHeading("$label ${m.groupValues[2]}", title, 1, null, Segment.Heading.KIND_APPENDIX, printed)
            }
        }
        PART_HEADING.matchEntire(text)?.let { m ->
            return ParsedHeading(
                "PART ${m.groupValues[1]}", m.groupValues[2].trim(), 2, null,
                Segment.Heading.KIND_PART, arabic(m.groupValues[1]),
            )
        }
        NUMBERED_HEADING.matchEntire(text)?.let { m ->
            val number = m.groupValues[1]
            val title = m.groupValues[2].trim()
            if (title.length in 2..MAX_TITLE_CHARS && looksLikeTitle(title)) {
                return ParsedHeading(number, title, 3 + number.count { it == '.' }.coerceAtMost(3), null, Segment.Heading.KIND_CLAUSE, number)
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
        SPEC_HEADING.matches(text) || PART_HEADING.matches(text) || NUMBERED_HEADING.matches(text) ||
            CHAPTER_HEADING.matches(text) || APPENDIX_HEADING.matches(text)

    private fun isContentsPage(page: PageLayout): Boolean {
        val lines = page.lines.map { it.text.trim().replace(SPACES, " ") }
        val entryCount = lines.count { CONTENTS_ENTRY.matches(it) }
        val hasContentsTitle = lines.take(CONTENTS_TITLE_LOOKAHEAD)
            .any { CONTENTS_TITLE.matches(it.removeSuffix(":")) }
        return (hasContentsTitle && entryCount >= MIN_CONTENTS_ENTRIES_WITH_TITLE) ||
            entryCount >= MIN_CONTENTS_ENTRIES
    }

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
        private const val LABEL_TITLE_MAX_GAP_RATIO = 3.5f
        private const val CONTENTS_TITLE_LOOKAHEAD = 12
        private const val MIN_CONTENTS_ENTRIES_WITH_TITLE = 3
        private const val MIN_CONTENTS_ENTRIES = 6
        private val SPACES = Regex("[\\s\\u00a0]+")
        private val BODY_END = Regex("[.;:]\\s*$")
        private val SPEC_HEADING = Regex("(?i)^SECTION\\s+([0-9]{3,8}(?:[-.]?[0-9A-Z]+)?)\\s*[-–—:]?\\s*(.*)$")
        private val PART_HEADING = Regex("(?i)^PART\\s+([0-9IVX]+)\\s*[-–—:]?\\s*(.*)$")
        /** `CHAPTER 13`, `Chapter XIII – Excavation`, `CHAPTER 13: EXCAVATION`. */
        private val CHAPTER_HEADING = Regex("(?i)^CHAPTER\\s+([0-9]{1,3}|[IVXLC]{1,7})\\s*[-–—:.]?\\s*(.*)$")
        /** `APPENDIX A`, `Annexure II - Standards`, `ANNEX 3`. */
        private val APPENDIX_HEADING = Regex("(?i)^(APPENDIX|ANNEXURE|ANNEX)\\s+([0-9]{1,3}|[IVXLC]{1,7}|[A-Z])\\s*[-–—:.]?\\s*(.*)$")
        private val ROMAN = Regex("^[IVXLCDM]{1,9}$")
        private val ROMAN_VALUES = mapOf('I' to 1, 'V' to 5, 'X' to 10, 'L' to 50, 'C' to 100, 'D' to 500, 'M' to 1000)
        private val PART_ONE_ROOT = Regex("(?i)^PART\\s+(?:1|I)(?:\\s*[-–—:]\\s*|\\s+).+$")
        private val NUMBERED_PART_ONE_ROOT = Regex("(?i)^1\\.0+\\s+.+$")
        private val NUMBERED_HEADING = Regex("^((?:[0-9]+\\.)+[0-9A-Z]+)\\s+(.+)$")
        private val CONTENTS_TITLE = Regex("(?i)^(?:table\\s+of\\s+contents|contents|index)$")
        private val CONTENTS_ENTRY = Regex("^\\d+(?:\\.\\d+){0,5}\\s+.+?\\s+\\d{1,4}$")
        /** Arabic, alphabetic, Roman (`ii.`, `IV)`), parenthesised and bullet/check labels. */
        private val LIST_ITEM = Regex(
            "^((?:[A-Za-z]|[0-9]{1,3}|[ivxl]{2,6}|[IVXL]{2,6})[.)]|\\([A-Za-z0-9]{1,4}\\)|[•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF]|[-–—](?=\\s))\\s+(.+)$",
        )
        private const val MAX_LABEL_LINE_CHARS = 90

    /** `13` for `13`, `XIII`, `xiii`; letters pass through upper-cased so `Appendix a` and `APPENDIX A` agree. */
        fun arabic(printed: String): String {
            val upper = printed.trim().uppercase(Locale.ROOT)
            if (upper.all(Char::isDigit)) return upper.trimStart('0').ifEmpty { "0" }
            if (!ROMAN.matches(upper)) return upper
            var total = 0
            var previous = 0
            for (ch in upper.reversed()) {
                val value = ROMAN_VALUES.getValue(ch)
                if (value < previous) total -= value else { total += value; previous = value }
            }
            return total.toString()
        }

        fun normalizeHeading(text: String): String = Normalizer.normalize(text, Normalizer.Form.NFKC)
            .lowercase(Locale.ROOT)
            .replace(Regex("[^\\p{L}\\p{N}]+"), " ")
            .trim()
            .replace(Regex("\\s+"), " ")
    }
}
