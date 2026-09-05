package com.example.pdfgemmarag.inference.pdf

/**
 * Removes running headers/footers and bare page numbers. Long documents repeat the same 1-2 lines at
 * the top and bottom of every page; left in, they pollute every chunk and dominate keyword search.
 */
class HeaderFooterStripper(
    private val edgeLines: Int = 2,
    private val minPageFraction: Double = 0.30,
    private val minOccurrences: Int = 3,
) {
    private val pageNumberPattern = Regex(
        """^\s*(page\s*)?[\divxlc]{1,5}(\s*(/|of)\s*\d{1,5})?\s*$""",
        RegexOption.IGNORE_CASE,
    )

    fun strip(pages: List<PageLayout>): List<PageLayout> {
        if (pages.size < minOccurrences) return pages.map { dropPageNumbers(it) }

        val counts = HashMap<String, Int>()
        for (page in pages) {
            edgeCandidates(page).map { normalize(it.text) }.toSet().forEach { key ->
                counts[key] = (counts[key] ?: 0) + 1
            }
        }
        val threshold = maxOf(minOccurrences, (pages.size * minPageFraction).toInt())
        val repeated = counts.filterValues { it >= threshold }.keys

        return pages.map { page ->
            val edges = edgeCandidates(page).toSet()
            val kept = page.lines.filter { line ->
                val isEdge = line in edges
                !(isEdge && (normalize(line.text) in repeated || pageNumberPattern.matches(line.text)))
            }
            page.copy(lines = kept)
        }
    }

    private fun dropPageNumbers(page: PageLayout): PageLayout {
        val edges = edgeCandidates(page).toSet()
        return page.copy(lines = page.lines.filter { !(it in edges && pageNumberPattern.matches(it.text)) })
    }

    private fun edgeCandidates(page: PageLayout): List<LineBox> {
        val n = page.lines.size
        if (n == 0) return emptyList()
        val top = page.lines.take(edgeLines)
        val bottom = page.lines.takeLast(edgeLines)
        return (top + bottom).distinct()
    }

    /** Digits collapse so "Chapter 3 - 41" and "Chapter 3 - 42" match. */
    internal fun normalize(text: String): String =
        text.lowercase().replace(Regex("\\d+"), "#").replace(Regex("\\s+"), " ").trim()
}
