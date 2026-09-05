package com.example.pdfgemmarag.inference.pdf

import kotlin.math.abs
import kotlin.math.ceil

/**
 * Repairs TextExtractor's reading-order output on tabular regions.
 *
 * Apryse groups text into flow "lines" by proximity. Table cells are closer to the cell above than
 * to the cell on the right, so Spike 5 (emulator, PDFNet 12.1) returned every cell as its own
 * one-word line, in column order (`Year`, `Revenue`, `Profit`, `2021`, `2022`, `2023`, `3.1M`, ...);
 * other generators produce a whole column as one vertical "line". Either way the table clusterer
 * would see one-cell rows and emit paragraph soup.
 *
 * Fix, geometry only:
 * 1. explode vertical stacks (words of one line on different baselines) into one line per word;
 * 2. group lines by baseline; a group of >= 2 lines that are each *narrow* (< [narrowFraction] of
 *    the page width) or already split by wide gaps is a candidate row, which excludes two-column
 *    body text whose lines are ~45% wide with only word-sized gaps;
 * 3. keep candidate rows only when >= [minRows] vertically adjacent candidates share >= 2 column
 *    x-positions, i.e. they look like a table, and merge each into a single left-to-right row.
 * Everything else is returned untouched, in the original order.
 */
class RowReflow(
    private val narrowFraction: Float = 0.30f,
    private val minRows: Int = 2,
) {

    fun reflow(lines: List<LineBox>, pageWidth: Float): List<LineBox> {
        if (lines.size < 2) return lines
        val exploded = lines.flatMap { if (isVerticalStack(it)) it.words.map { w -> LineBox(listOf(w)) } else listOf(it) }
        val medianHeight = exploded.map { it.box.height }.sorted().let { it[it.size / 2] }.coerceAtLeast(1f)
        val baselineTol = medianHeight * 0.5f
        val maxNarrow = pageWidth * narrowFraction
        val colTol = (pageWidth * 0.015f).coerceAtLeast(2f)

        // Baseline groups, as index lists into `exploded`.
        val byY = exploded.indices.sortedBy { exploded[it].box.centerY }
        val groups = ArrayList<ArrayList<Int>>()
        for (idx in byY) {
            val cy = exploded[idx].box.centerY
            val last = groups.lastOrNull()
            if (last != null && abs(exploded[last.last()].box.centerY - cy) <= baselineTol) last += idx else groups += arrayListOf(idx)
        }
        // A line qualifies when it is narrow (a lone cell) or already cellular (a partial header row with
        // wide internal gaps); a body-text line is wide and has no gap wider than a couple of characters.
        fun cellular(line: LineBox): Boolean {
            if (line.box.width <= maxNarrow) return true
            val gap = 2.5f * line.avgCharWidth.coerceAtLeast(0.5f)
            return (1 until line.words.size).any { line.words[it].box.left - line.words[it - 1].box.right > gap }
        }
        val candidates = groups.filter { g -> g.size >= 2 && g.all { cellular(exploded[it]) } }
        if (candidates.isEmpty()) return if (exploded.size == lines.size) lines else exploded

        // Runs of vertically adjacent candidate rows with aligned column starts.
        fun lefts(g: List<Int>) = g.map { exploded[it].box.left }
        fun aligned(a: List<Int>, b: List<Int>): Boolean {
            val la = lefts(a); val lb = lefts(b)
            return lb.count { x -> la.any { abs(it - x) <= colTol } } >= 2
        }
        fun cy(g: List<Int>) = exploded[g.first()].box.centerY
        val merged = HashSet<List<Int>>()
        var run = arrayListOf(candidates[0])
        fun closeRun() { if (run.size >= minRows) merged += run; }
        for (k in 1 until candidates.size) {
            val prev = run.last(); val cur = candidates[k]
            if (cy(cur) - cy(prev) <= 2.5f * medianHeight && aligned(prev, cur)) run += cur else { closeRun(); run = arrayListOf(cur) }
        }
        closeRun()
        if (merged.isEmpty()) return if (exploded.size == lines.size) lines else exploded

        // Emit: each merged row replaces its earliest member; other members are dropped.
        val rowOfIndex = HashMap<Int, List<Int>>()
        for (g in merged) for (i in g) rowOfIndex[i] = g
        val emitted = HashSet<List<Int>>()
        val out = ArrayList<LineBox>(exploded.size)
        for ((i, line) in exploded.withIndex()) {
            val g = rowOfIndex[i]
            if (g == null) { out += line; continue }
            if (!emitted.add(g)) continue
            out += LineBox(g.flatMap { exploded[it].words }.sortedBy { it.box.left })
        }
        return out
    }

    /** True when the words of a TextExtractor line sit on clearly different baselines. */
    internal fun isVerticalStack(line: LineBox): Boolean {
        if (line.words.size < 2) return false
        val heights = line.words.map { it.box.height }
        val tol = (heights.sorted()[heights.size / 2]).coerceAtLeast(1f) * 0.5f
        var breaks = 0
        for (k in 1 until line.words.size) {
            if (abs(line.words[k].box.centerY - line.words[k - 1].box.centerY) > tol) breaks++
        }
        // A majority of adjacent pairs must change baseline; a lone superscript in a sentence must not qualify.
        return breaks >= ceil((line.words.size - 1) * 0.75).toInt().coerceAtLeast(1)
    }
}
