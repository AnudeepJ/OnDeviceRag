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

data class WordBox(
    val text: String,
    val box: Box,
    val fontName: String? = null,
    val fontSize: Float? = null,
    val bold: Boolean = false,
)

data class LineBox(val words: List<WordBox>) {
    val text: String get() = words.joinToString(" ") { it.text }
    val box: Box by lazy { words.map { it.box }.reduce { a, b -> a.union(b) } }
    val avgCharWidth: Float by lazy {
        val chars = words.sumOf { it.text.length }.coerceAtLeast(1)
        words.sumOf { it.box.width.toDouble() }.toFloat() / chars
    }
    val indent: Float get() = box.left
    val lineHeight: Float get() = box.height
    val baseline: Float get() = box.bottom
    val isBold: Boolean get() = words.isNotEmpty() && words.count { it.bold } * 2 >= words.size
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

/** Layout-analysed content of a page, in physical reading order. */
sealed class Segment {
    abstract val pageNumber: Int

    data class Heading(
        override val pageNumber: Int,
        val number: String?,
        val title: String,
        val level: Int,
        val specificationNumber: String? = null,
        /** Structural kind: SECTION (specification), PART, CHAPTER, APPENDIX, CLAUSE (dotted), HEADING (styled). */
        val kind: String = if (specificationNumber != null) KIND_SECTION else if (number == null) KIND_HEADING else KIND_CLAUSE,
        /** Printed identifier normalised for lookup: `13` for `CHAPTER XIII`, `A` for `APPENDIX A`, `1.01` for a clause. */
        val printedNumber: String = number.orEmpty(),
    ) : Segment() {
        val text: String get() = listOfNotNull(number, title).joinToString(" ").trim()

        companion object {
            const val KIND_SECTION = "SECTION"
            const val KIND_PART = "PART"
            const val KIND_CHAPTER = "CHAPTER"
            const val KIND_APPENDIX = "APPENDIX"
            const val KIND_CLAUSE = "CLAUSE"
            const val KIND_HEADING = "HEADING"
        }
    }

    data class Paragraph(override val pageNumber: Int, val text: String) : Segment()

    data class ListItem(val label: String, val text: String)

    data class ListBlock(override val pageNumber: Int, val items: List<ListItem>) : Segment()

    /** [header] and [rows] are already pipe-delimited Markdown table lines. */
    data class Table(
        override val pageNumber: Int,
        val header: String,
        val rows: List<String>,
        val caption: String = "",
    ) : Segment()
}

data class PageContent(val pageNumber: Int, val segments: List<Segment>) {
    val charCount: Int
        get() = segments.sumOf {
            when (it) {
                is Segment.Heading -> it.text.length
                is Segment.Paragraph -> it.text.length
                is Segment.ListBlock -> it.items.sumOf { item -> item.label.length + item.text.length }
                is Segment.Table -> it.header.length + it.rows.sumOf { r -> r.length }
            }
        }
}
