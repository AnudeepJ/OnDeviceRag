package com.example.pdfgemmarag.inference.pdf

/** Axis-aligned box in PDF user space (origin bottom-left) or bitmap space; only relative geometry is used. */
data class Box(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val centerY: Float get() = (top + bottom) / 2f

    fun union(other: Box) = Box(
        minOf(left, other.left), minOf(top, other.top), maxOf(right, other.right), maxOf(bottom, other.bottom),
    )
}

data class WordBox(val text: String, val box: Box)

data class LineBox(val words: List<WordBox>) {
    val text: String get() = words.joinToString(" ") { it.text }
    val box: Box by lazy { words.map { it.box }.reduce { a, b -> a.union(b) } }
    val avgCharWidth: Float by lazy {
        val chars = words.sumOf { it.text.length }.coerceAtLeast(1)
        words.sumOf { it.box.width.toDouble() }.toFloat() / chars
    }
}

/** Geometry-first representation of one page, produced by Apryse (or OCR) before layout analysis. */
data class PageLayout(
    val pageNumber: Int,
    val width: Float,
    val height: Float,
    val lines: List<LineBox>,
    val hasImages: Boolean,
    val source: Source,
) {
    enum class Source { TEXT, OCR }

    val charCount: Int get() = lines.sumOf { l -> l.words.sumOf { it.text.length } }
    val plainText: String get() = lines.joinToString("\n") { it.text }
}

/** Layout-analysed content of a page: paragraphs and detected tables, in reading order. */
sealed class Segment {
    abstract val pageNumber: Int

    data class Paragraph(override val pageNumber: Int, val text: String) : Segment()

    /** [header] and [rows] are already pipe-delimited Markdown table lines. */
    data class Table(override val pageNumber: Int, val header: String, val rows: List<String>) : Segment()
}

data class PageContent(val pageNumber: Int, val segments: List<Segment>) {
    val charCount: Int
        get() = segments.sumOf {
            when (it) {
                is Segment.Paragraph -> it.text.length
                is Segment.Table -> it.header.length + it.rows.sumOf { r -> r.length }
            }
        }
}
