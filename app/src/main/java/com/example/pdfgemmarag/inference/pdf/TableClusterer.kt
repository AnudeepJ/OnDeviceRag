package com.example.pdfgemmarag.inference.pdf

import kotlin.math.abs

/**
 * Turns positioned lines into paragraphs and tables.
 *
 * PDFs have no table semantics; a table is just rows of words whose horizontal gaps line up. The
 * heuristic: split each line into cells at gaps wider than [gapFactor] average character widths,
 * then group consecutive multi-cell lines whose cell starts align into a table block. Column
 * boundaries are clustered from the block's cell starts and every row is re-emitted as a
 * pipe-delimited Markdown row, so header text stays attached to values in every chunk.
 */
class TableClusterer(
    private val gapFactor: Float = 2.5f,
    private val minRows: Int = 2,
    private val minCells: Int = 2,
    private val paragraphGapFactor: Float = 1.6f,
) {

    fun analyse(page: PageLayout): PageContent {
        if (page.lines.isEmpty()) return PageContent(page.pageNumber, emptyList())
        val cellsPerLine = page.lines.map { splitCells(it) }
        val segments = ArrayList<Segment>()
        val paragraph = StringBuilder()
        val medianLineHeight = median(page.lines.map { it.box.height }).coerceAtLeast(1f)

        fun flushParagraph() {
            if (paragraph.isNotBlank()) segments += Segment.Paragraph(page.pageNumber, paragraph.toString().trim())
            paragraph.setLength(0)
        }

        var i = 0
        while (i < page.lines.size) {
            val run = tableRunFrom(page, cellsPerLine, i)
            if (run != null) {
                flushParagraph()
                segments += buildTable(page, cellsPerLine, i, run)
                i += run
                continue
            }
            val line = page.lines[i]
            if (i > 0 && paragraph.isNotEmpty()) {
                val prev = page.lines[i - 1]
                val gap = abs(line.box.centerY - prev.box.centerY)
                if (gap > paragraphGapFactor * medianLineHeight) flushParagraph()
            }
            if (paragraph.isNotEmpty()) paragraph.append(joiner(paragraph))
            paragraph.append(line.text)
            i++
        }
        flushParagraph()
        return PageContent(page.pageNumber, segments)
    }

    /** CJK text has no inter-word spaces; joining lines with a space would corrupt it. */
    private fun joiner(sb: StringBuilder): String {
        val last = sb[sb.length - 1]
        val script = Character.UnicodeScript.of(last.code)
        return if (script == Character.UnicodeScript.HAN || script == Character.UnicodeScript.HIRAGANA ||
            script == Character.UnicodeScript.KATAKANA || script == Character.UnicodeScript.HANGUL
        ) "" else " "
    }

    private data class Cell(val text: String, val left: Float, val right: Float)

    private fun splitCells(line: LineBox): List<Cell> {
        if (line.words.isEmpty()) return emptyList()
        val threshold = gapFactor * line.avgCharWidth.coerceAtLeast(0.5f)
        val cells = ArrayList<Cell>()
        var cur = StringBuilder(line.words[0].text)
        var left = line.words[0].box.left
        var right = line.words[0].box.right
        for (k in 1 until line.words.size) {
            val w = line.words[k]
            if (w.box.left - right > threshold) {
                cells += Cell(cur.toString(), left, right)
                cur = StringBuilder(w.text); left = w.box.left; right = w.box.right
            } else {
                cur.append(' ').append(w.text); right = maxOf(right, w.box.right)
            }
        }
        cells += Cell(cur.toString(), left, right)
        return cells
    }

    /** Returns the number of consecutive lines starting at [start] that form a table, or null. */
    private fun tableRunFrom(page: PageLayout, cells: List<List<Cell>>, start: Int): Int? {
        if (cells[start].size < minCells) return null
        val tol = alignmentTolerance(page)
        var columns = cells[start].map { it.left }.toMutableList()
        var count = 1
        var j = start + 1
        while (j < cells.size) {
            val row = cells[j]
            if (row.size < minCells || abs(row.size - cells[j - 1].size) > 1) break
            val aligned = row.count { c -> columns.any { abs(it - c.left) <= tol } }
            if (aligned < minOf(row.size, columns.size) - 1 || aligned < 2) break
            for (c in row) if (columns.none { abs(it - c.left) <= tol }) columns += c.left
            count++
            j++
        }
        if (count < minRows) return null
        val runRows = cells.subList(start, start + count)
        // Specification clauses frequently align the enumerator and body into two columns. They
        // are lists, not data tables, and must stay as label+text logical units.
        val enumerated = runRows.count { row -> row.firstOrNull()?.text?.trim()?.matches(ENUMERATOR) == true }
        if (enumerated * 2 >= runRows.size) return null
        // Two-column layouts need repeated evidence in the value column. Three or more aligned
        // columns remain a strong table signal.
        if (columns.size <= 2) {
            val populated = runRows.count { it.size >= 2 && it[1].text.isNotBlank() }
            if (populated < 3) return null
        }
        return count
    }

    private fun buildTable(page: PageLayout, cells: List<List<Cell>>, start: Int, run: Int): Segment.Table {
        val tol = alignmentTolerance(page)
        val starts = (start until start + run).flatMap { cells[it].map { c -> c.left } }.sorted()
        val columns = ArrayList<Float>()
        for (x in starts) if (columns.isEmpty() || x - columns.last() > tol) columns += x

        fun rowToMarkdown(row: List<Cell>): String {
            val slots = Array(columns.size) { "" }
            for (c in row) {
                val idx = columns.indices.minByOrNull { abs(columns[it] - c.left) } ?: 0
                slots[idx] = if (slots[idx].isEmpty()) c.text else slots[idx] + " " + c.text
            }
            return "| " + slots.joinToString(" | ") { it.replace("|", "/") } + " |"
        }

        val header = rowToMarkdown(cells[start])
        val rows = (start + 1 until start + run).map { rowToMarkdown(cells[it]) }
        return Segment.Table(page.pageNumber, header, rows)
    }

    private fun alignmentTolerance(page: PageLayout) = (page.width * 0.015f).coerceAtLeast(2f)

    private fun median(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val s = values.sorted()
        return s[s.size / 2]
    }

    companion object {
        private val ENUMERATOR = Regex(
            "^(?:[A-Za-z]|[0-9]+|[ivxl]{2,6}|[IVXL]{2,6})[.)]$|^\\([A-Za-z0-9]{1,4}\\)$|^[•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF]$",
        )
    }
}
