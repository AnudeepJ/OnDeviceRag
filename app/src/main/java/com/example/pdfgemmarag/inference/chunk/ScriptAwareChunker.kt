package com.example.pdfgemmarag.inference.chunk

import com.example.pdfgemmarag.inference.ocr.Script
import com.example.pdfgemmarag.inference.ocr.ScriptDetector
import com.example.pdfgemmarag.inference.pdf.PageContent
import com.example.pdfgemmarag.inference.pdf.Segment
import java.text.BreakIterator
import java.util.Locale

data class Chunk(
    val chunkIndex: Int,
    val pageNumber: Int,
    val text: String,
    val isTable: Boolean,
    val script: Script,
)

/**
 * Sentence-aware, script-adaptive chunking.
 *
 * Gemma tokenises CJK at roughly 0.7-1.5 characters per token and European text at ~4, so a fixed
 * character window would produce wildly different token counts. CJK windows are ~330-400 chars,
 * European ~1000-1200; both land near 300-400 tokens, under EmbeddingGemma's 512-token input with
 * the task prefix. Chunks never cross pages so every chunk carries an exact page citation.
 */
class ScriptAwareChunker(
    private val cjkTarget: Int = 360,
    private val cjkMax: Int = 400,
    private val latinTarget: Int = 1100,
    private val latinMax: Int = 1200,
    private val overlapFraction: Double = 0.12,
    /** Table rows are packed until the chunk reaches this many chars; headers repeat per chunk. */
    private val tableTarget: Int = 1000,
    /** Adjacent same-page fragments shorter than this are concatenated (spec PDFs emit one-line segments). */
    private val minMergeChars: Int = 160,
    /** After merging, drop leftovers this short (running labels like "FORMWORK" / "03600-6"). */
    private val minKeepChars: Int = 40,
) {

    fun chunk(pages: List<PageContent>): List<Chunk> {
        val out = ArrayList<Chunk>()
        var index = 0
        for (page in pages) {
            for (segment in page.segments) {
                when (segment) {
                    is Segment.Paragraph -> {
                        for (text in chunkParagraph(segment.text)) {
                            out += Chunk(index++, page.pageNumber, text, isTable = false, script = ScriptDetector.detect(text))
                        }
                    }
                    is Segment.Table -> {
                        for (text in chunkTable(segment)) {
                            out += Chunk(index++, page.pageNumber, text, isTable = true, script = ScriptDetector.detect(text))
                        }
                    }
                }
            }
        }
        return mergeTiny(out)
    }

    internal fun mergeTiny(chunks: List<Chunk>): List<Chunk> {
        if (chunks.isEmpty()) return chunks
        val merged = ArrayList<Chunk>()
        var acc: Chunk? = null
        fun flush() { acc?.let { merged += it }; acc = null }
        for (c in chunks) {
            if (c.isTable) {
                flush()
                merged += c
                continue
            }
            val cur = acc
            if (cur == null) {
                acc = c
            } else if (cur.pageNumber == c.pageNumber && cur.text.length < minMergeChars) {
                acc = cur.copy(text = cur.text.trimEnd() + "\n" + c.text.trim())
            } else {
                merged += cur
                acc = c
            }
        }
        flush()
        return merged
            .filter { it.isTable || it.text.length >= minKeepChars }
            .mapIndexed { i, c -> c.copy(chunkIndex = i) }
    }

    internal fun chunkParagraph(text: String): List<String> {
        val cleaned = text.replace(Regex("[ \\t\\u00A0]+"), " ").trim()
        if (cleaned.isEmpty()) return emptyList()
        val script = ScriptDetector.detect(cleaned)
        val (target, max) = if (script.isCjk) cjkTarget to cjkMax else latinTarget to latinMax
        if (cleaned.length <= max) return listOf(cleaned)

        val sentences = splitSentences(cleaned, script).flatMap { hardSplit(it, max) }
        val chunks = ArrayList<String>()
        val current = StringBuilder()
        val overlapChars = (target * overlapFraction).toInt()

        fun flush() {
            if (current.isNotBlank()) chunks += current.toString().trim()
        }

        for (sentence in sentences) {
            if (current.isNotEmpty() && current.length + sentence.length > target) {
                flush()
                // Carry the tail sentences (~overlapFraction of the window) into the next chunk.
                val carry = tail(current.toString(), overlapChars, script)
                current.setLength(0)
                current.append(carry)
            }
            if (current.isNotEmpty() && !script.isCjk && !current.endsWith(" ")) current.append(' ')
            current.append(sentence)
        }
        flush()
        return chunks
    }

    internal fun chunkTable(table: Segment.Table): List<String> {
        val separator = "|" + " --- |".repeat(table.header.count { it == '|' } - 1)
        val head = table.header + "\n" + separator + "\n"
        if (table.rows.isEmpty()) return listOf(table.header)
        val chunks = ArrayList<String>()
        val current = StringBuilder(head)
        for (row in table.rows) {
            if (current.length > head.length && current.length + row.length + 1 > tableTarget) {
                chunks += current.toString().trimEnd()
                current.setLength(0)
                current.append(head) // header injection keeps columns meaningful in every chunk
            }
            current.append(row).append('\n')
        }
        chunks += current.toString().trimEnd()
        return chunks
    }

    internal fun splitSentences(text: String, script: Script): List<String> {
        val locale = when (script) {
            Script.JAPANESE -> Locale.JAPANESE
            Script.CHINESE -> Locale.CHINESE
            Script.KOREAN -> Locale.KOREAN
            Script.LATIN -> Locale.ROOT
        }
        val it = BreakIterator.getSentenceInstance(locale)
        it.setText(text)
        val out = ArrayList<String>()
        var start = it.first()
        var end = it.next()
        while (end != BreakIterator.DONE) {
            val s = text.substring(start, end).trim()
            if (s.isNotEmpty()) out += s
            start = end
            end = it.next()
        }
        // BreakIterator can miss full-width terminators without trailing space; split those too.
        return if (script.isCjk) out.flatMap { splitCjkTerminators(it) } else out
    }

    private fun splitCjkTerminators(s: String): List<String> {
        val parts = ArrayList<String>()
        var startIdx = 0
        for (i in s.indices) {
            val c = s[i]
            if (c == '。' || c == '！' || c == '？' || c == '\n') {
                parts += s.substring(startIdx, i + 1).trim()
                startIdx = i + 1
            }
        }
        if (startIdx < s.length) parts += s.substring(startIdx).trim()
        return parts.filter { it.isNotEmpty() }
    }

    /** A single "sentence" longer than the max (no punctuation) is split on whitespace or hard length. */
    private fun hardSplit(sentence: String, max: Int): List<String> {
        if (sentence.length <= max) return listOf(sentence)
        val out = ArrayList<String>()
        var i = 0
        while (i < sentence.length) {
            var end = minOf(i + max, sentence.length)
            if (end < sentence.length) {
                val space = sentence.lastIndexOf(' ', end)
                if (space > i + max / 2) end = space
            }
            out += sentence.substring(i, end).trim()
            i = end
        }
        return out.filter { it.isNotEmpty() }
    }

    private fun tail(text: String, chars: Int, script: Script): String {
        if (chars <= 0 || text.length <= chars) return if (text.length <= chars) text else ""
        val cut = text.length - chars
        val boundary = if (script.isCjk) cut else text.indexOf(' ', cut).let { if (it == -1) cut else it + 1 }
        return text.substring(boundary).trim().let { if (it.isEmpty()) "" else "$it " }
    }
}
