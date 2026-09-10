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
            val run = tableRunFrom(page, cellsPerLine, i, medianLineHeight)
            if (run != null) {
                // "Table 3 — Cover" style captions sit on the line before or after the grid.
                val before = page.lines.getOrNull(i - 1)?.text?.trim()
                val after = page.lines.getOrNull(i + run.lineCount)?.text?.trim()
                val caption = listOfNotNull(before, after).firstOrNull { TableIdentity.looksLikeCaption(it) && it.length <= MAX_CAPTION_CHARS }.orEmpty()
                if (caption.isNotEmpty() && before == caption) {
                    // Remove the caption from the pending paragraph so it is not emitted twice.
                    val idx = paragraph.lastIndexOf(caption)
                    if (idx >= 0) paragraph.setLength(idx)
                }
                flushParagraph()
                segments += buildTable(page, run, caption, TableIdentity.numberFromCaption(caption))
                i += run.lineCount
                if (caption.isNotEmpty() && after == caption) i++
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

    internal data class Cell(val text: String, val left: Float, val right: Float)

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

    /** A detected grid: consumed physical lines and logical rows (wrapped cells already merged). */
    internal class TableRun(val lineCount: Int, val rows: List<List<Cell>>, val columns: List<Float>)

    /**
     * Returns the grid starting at [start], or null. A physical line joins the grid either as a
     * new row (>= [minCells] cells aligned with known columns) or as a *wrapped cell*: a line with
     * fewer cells whose every cell sits inside an existing column band, tightly below the previous
     * line. Wrapped cells are appended to the cell above, so a description that spans three lines
     * next to a one-line value stays one row.
     */
    private fun tableRunFrom(page: PageLayout, cells: List<List<Cell>>, start: Int, medianLineHeight: Float): TableRun? {
        if (cells[start].size < minCells) return null
        val tol = alignmentTolerance(page)
        val columns = cells[start].map { it.left }.toMutableList()
        val rows = ArrayList<MutableList<Cell>>()
        rows += cells[start].toMutableList()
        var consumed = 1
        var j = start + 1
        var previousLine = page.lines[start]
        // Wrapped lines are committed only when a further real row follows (or a short tail
        // ends the grid); a long run of single-cell lines after the last row is prose.
        var pendingWraps = 0
        var committedConsumed = 1
        while (j < cells.size) {
            val row = cells[j]
            val line = page.lines[j]
            if (row.size >= minCells) {
                if (abs(row.size - rows.last().size.coerceAtMost(columns.size)) > 1 && row.size > columns.size + 1) break
                val aligned = row.count { c -> columns.any { abs(it - c.left) <= tol } }
                if (aligned < minOf(row.size, columns.size) - 1 || aligned < 2) break
                for (c in row) if (columns.none { abs(it - c.left) <= tol }) columns += c.left
                rows += row.toMutableList()
                pendingWraps = 0
                committedConsumed = consumed + 1
            } else {
                if (pendingWraps >= MAX_WRAPPED_LINES) break
                val gap = line.box.top - previousLine.box.bottom
                if (row.isEmpty() || gap > WRAP_MAX_GAP_LINES * medianLineHeight) break
                val sortedColumns = columns.sorted()
                val tableRight = rows.flatten().maxOf { it.right }
                val fits = row.all { c ->
                    val k = sortedColumns.indexOfFirst { abs(it - c.left) <= tol }
                    if (k < 0) return@all false
                    val bandRight = sortedColumns.getOrNull(k + 1)?.minus(tol) ?: (tableRight + tol)
                    // A wrapped label in the first column continues in lower case; a capitalised
                    // first-column line is the next row label or prose after the grid.
                    val continuesLabel = k > 0 || c.text.trim().firstOrNull()?.isLowerCase() == true
                    c.right <= bandRight && continuesLabel
                }
                // A wrapped cell continues text; a new enumerated clause or heading does not.
                if (!fits || row.first().text.trim().matches(ENUMERATOR) || isSentenceStartAtMargin(row, sortedColumns.first(), tol)) break
                rows.last().addAll(row)
                pendingWraps++
            }
            previousLine = line
            consumed++
            j++
        }
        if (pendingWraps > MAX_TRAILING_WRAPS) {
            // Roll back uncommitted trailing wraps: they belong to whatever follows the grid.
            val last = rows.last()
            val keep = last.size - pendingWraps.coerceAtMost(last.size - 1)
            while (last.size > keep) last.removeAt(last.lastIndex)
            consumed = committedConsumed
        }
        if (rows.size < minRows) return null
        // Specification clauses frequently align the enumerator and body into two columns. They
        // are lists, not data tables, and must stay as label+text logical units.
        val enumerated = rows.count { row -> row.firstOrNull()?.text?.trim()?.matches(ENUMERATOR) == true }
        if (enumerated * 2 >= rows.size) return null
        // Two-column layouts need repeated evidence in the value column. Three or more aligned
        // columns remain a strong table signal.
        if (columns.size <= 2) {
            val populated = rows.count { it.size >= 2 && it[1].text.isNotBlank() }
            if (populated < 3) return null
        }
        return TableRun(consumed, rows, columns)
    }

    /** A lone first-column line starting a capitalised sentence is prose that happens to share the margin. */
    private fun isSentenceStartAtMargin(row: List<Cell>, firstColumn: Float, tol: Float): Boolean {
        if (row.size != 1) return false
        val c = row.single()
        if (abs(c.left - firstColumn) > tol) return false
        val first = c.text.trim().firstOrNull() ?: return false
        return first.isUpperCase() && c.text.trim().split(' ').size >= 6
    }

    private fun buildTable(page: PageLayout, run: TableRun, caption: String, tableNumber: String): Segment.Table {
        val tol = alignmentTolerance(page)
        val starts = run.rows.flatten().map { it.left }.sorted()
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

        val header = rowToMarkdown(run.rows.first())
        val rows = run.rows.drop(1).map(::rowToMarkdown)
        return Segment.Table(page.pageNumber, header, rows, caption, tableNumber)
    }

    private fun alignmentTolerance(page: PageLayout) = (page.width * 0.015f).coerceAtLeast(2f)

    private fun median(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val s = values.sorted()
        return s[s.size / 2]
    }

    companion object {
        private const val WRAP_MAX_GAP_LINES = 1.3f
        private const val MAX_WRAPPED_LINES = 6
        private const val MAX_TRAILING_WRAPS = 2
        private const val MAX_CAPTION_CHARS = 120
        private val ENUMERATOR = Regex(
            "^(?:[A-Za-z]|[0-9]+|[ivxl]{2,6}|[IVXL]{2,6})[.)]$|^\\([A-Za-z0-9]{1,4}\\)$|^[•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF]$",
        )
    }
}
