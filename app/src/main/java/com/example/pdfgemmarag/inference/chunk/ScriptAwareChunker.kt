package com.example.pdfgemmarag.inference.chunk

import com.example.pdfgemmarag.inference.ocr.Script
import com.example.pdfgemmarag.inference.ocr.ScriptDetector
import com.example.pdfgemmarag.inference.pdf.PageContent
import com.example.pdfgemmarag.inference.pdf.Segment
import java.text.BreakIterator
import java.security.MessageDigest
import java.util.Locale

data class Chunk(
    val chunkIndex: Int,
    val pageNumber: Int,
    val text: String,
    val isTable: Boolean,
    val script: Script,
    val bodyText: String = text,
    val retrievalText: String = text,
    val sectionId: String = "",
    val specificationNumber: String = "",
    val sectionNumber: String = "",
    val sectionTitle: String = "",
    val sectionPath: String = "",
    val sectionLevel: Int = 0,
    /** Structural kind of the owning heading (SECTION, PART, CHAPTER, APPENDIX, CLAUSE, HEADING); blank at root. */
    val sectionKind: String = "",
    /** Lookup form of the owning heading's printed identifier (`13`, `A`, `2.05`). */
    val sectionPrintedNumber: String = "",
    val positionInSection: Int = 0,
    val contentKind: String = if (isTable) "TABLE" else "PARAGRAPH",
    val identifierAtoms: List<String> = emptyList(),
    val continuesFromChunkIndex: Int? = null,
    val continuesToChunkIndex: Int? = null,
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

    fun chunk(pages: List<PageContent>): List<Chunk> = chunk("document", pages)

    fun chunk(docHash: String, pages: List<PageContent>): List<Chunk> {
        val out = ArrayList<Chunk>()
        var index = 0
        var ordinal = 0
        val sectionStack = ArrayList<Segment.Heading>()
        var section = SectionState.root(docHash)
        var position = 0

        fun add(page: Int, body: String, table: Boolean, kind: String) {
            val clean = body.trim()
            if (clean.isEmpty()) return
            val retrieval = listOf(section.path, clean).filter { it.isNotBlank() }.joinToString("\n")
            out += Chunk(
                chunkIndex = index++,
                pageNumber = page,
                text = clean,
                isTable = table,
                script = ScriptDetector.detect(clean),
                bodyText = clean,
                retrievalText = retrieval,
                sectionId = section.id,
                specificationNumber = section.specificationNumber,
                sectionNumber = section.number,
                sectionTitle = section.title,
                sectionPath = section.path,
                sectionLevel = section.level,
                sectionKind = section.kind,
                sectionPrintedNumber = section.printedNumber,
                positionInSection = position++,
                contentKind = kind,
                identifierAtoms = IdentifierAtoms.extract(retrieval),
            )
        }

        for (page in pages) {
            for (segment in page.segments) {
                when (segment) {
                    is Segment.Heading -> {
                        while (sectionStack.isNotEmpty() && sectionStack.last().level >= segment.level) {
                            sectionStack.removeAt(sectionStack.lastIndex)
                        }
                        sectionStack += segment
                        ordinal++
                        section = SectionState.from(docHash, sectionStack, ordinal)
                        position = 0
                        add(page.pageNumber, segment.text, table = false, kind = "HEADING")
                    }
                    is Segment.Paragraph -> {
                        for (text in chunkParagraph(segment.text)) {
                            add(page.pageNumber, text, table = false, kind = "PARAGRAPH")
                        }
                    }
                    is Segment.ListBlock -> {
                        for (text in chunkList(segment)) {
                            add(page.pageNumber, text, table = false, kind = "LIST")
                        }
                    }
                    is Segment.Table -> {
                        for (text in chunkTable(segment)) {
                            add(page.pageNumber, text, table = true, kind = "TABLE")
                        }
                    }
                }
            }
        }
        return linkContinuations(mergeTiny(out))
    }

    /** Re-establishes IDs, section positions and continuation edges after token-window splitting. */
    fun reindex(chunks: List<Chunk>): List<Chunk> {
        val positions = HashMap<String, Int>()
        val indexed = chunks.mapIndexed { index, chunk ->
            val position = positions[chunk.sectionId] ?: 0
            positions[chunk.sectionId] = position + 1
            chunk.copy(
                chunkIndex = index,
                positionInSection = position,
                continuesFromChunkIndex = null,
                continuesToChunkIndex = null,
            )
        }
        return linkContinuations(indexed)
    }

    /**
     * Enforces the embedder's real tokenizer window. Character targets are only an estimate, and
     * tables/mixed scripts can otherwise be silently truncated by the model input encoder.
     */
    fun fitToTokenWindow(chunk: Chunk, maxTokens: Int, tokenCount: (String) -> Int): List<Chunk> {
        if (tokenCount(chunk.retrievalText) <= maxTokens) return listOf(chunk)
        val lines = chunk.bodyText.lines()
        val header = if (chunk.isTable && lines.size > 2) lines.take(2).joinToString("\n") + "\n" else ""
        val body = if (header.isEmpty()) chunk.text else lines.drop(2).joinToString("\n")
        val out = ArrayList<Chunk>()
        var offset = 0
        while (offset < body.length) {
            var low = offset + 1
            var high = body.length
            var best = -1
            while (low <= high) {
                val mid = (low + high) ushr 1
                val candidate = retrievalFor(chunk, header + body.substring(offset, mid))
                if (tokenCount(candidate) <= maxTokens) {
                    best = mid
                    low = mid + 1
                } else {
                    high = mid - 1
                }
            }
            check(best > offset) { "A table header exceeds the embedding token window" }
            var end = best
            if (best < body.length) {
                val newline = body.lastIndexOf('\n', best - 1)
                val space = body.lastIndexOf(' ', best - 1)
                val boundary = maxOf(newline, space)
                if (boundary > offset + (best - offset) / 2) end = boundary + 1
            }
            val text = (header + body.substring(offset, end).trim()).trim()
            check(tokenCount(retrievalFor(chunk, text)) <= maxTokens) { "Chunk split still exceeds embedding token window" }
            if (text.isNotEmpty()) out += chunk.copy(
                text = text,
                bodyText = text,
                retrievalText = retrievalFor(chunk, text),
                script = ScriptDetector.detect(text),
                identifierAtoms = IdentifierAtoms.extract(retrievalFor(chunk, text)),
            )
            offset = end
            while (offset < body.length && body[offset].isWhitespace()) offset++
        }
        return out
    }

    internal fun mergeTiny(chunks: List<Chunk>): List<Chunk> {
        if (chunks.isEmpty()) return chunks
        val merged = ArrayList<Chunk>()
        var acc: Chunk? = null
        fun flush() { acc?.let { merged += it }; acc = null }
        for (c in chunks) {
            if (c.isTable || c.contentKind == "HEADING" || c.contentKind == "LIST") {
                flush()
                merged += c
                continue
            }
            val cur = acc
            if (cur == null) {
                acc = c
            } else if (cur.pageNumber == c.pageNumber && cur.sectionId == c.sectionId && cur.text.length < minMergeChars) {
                val body = cur.bodyText.trimEnd() + "\n" + c.bodyText.trim()
                acc = cur.copy(
                    text = body,
                    bodyText = body,
                    retrievalText = retrievalFor(cur, body),
                    identifierAtoms = IdentifierAtoms.extract(retrievalFor(cur, body)),
                )
            } else {
                merged += cur
                acc = c
            }
        }
        flush()
        return merged
            .filter { it.isTable || it.contentKind == "HEADING" || it.contentKind == "LIST" || it.text.length >= minKeepChars }
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

    internal fun chunkList(list: Segment.ListBlock): List<String> {
        val chunks = ArrayList<String>()
        val current = StringBuilder()
        for (item in list.items) {
            val row = "${item.label} ${item.text}".trim()
            if (current.isNotEmpty() && current.length + row.length + 1 > latinTarget) {
                chunks += current.toString().trimEnd()
                current.setLength(0)
            }
            current.append(row).append('\n')
        }
        if (current.isNotBlank()) chunks += current.toString().trimEnd()
        return chunks
    }

    private fun linkContinuations(chunks: List<Chunk>): List<Chunk> = chunks.mapIndexed { i, chunk ->
        val previous = chunks.getOrNull(i - 1)?.takeIf {
            it.sectionId == chunk.sectionId && it.pageNumber + 1 == chunk.pageNumber &&
                it.contentKind == chunk.contentKind
        }
        val next = chunks.getOrNull(i + 1)?.takeIf {
            it.sectionId == chunk.sectionId && it.pageNumber == chunk.pageNumber + 1 &&
                it.contentKind == chunk.contentKind
        }
        chunk.copy(
            continuesFromChunkIndex = previous?.chunkIndex,
            continuesToChunkIndex = next?.chunkIndex,
        )
    }

    private fun retrievalFor(chunk: Chunk, body: String): String =
        listOf(chunk.sectionPath, body.trim()).filter { it.isNotBlank() }.joinToString("\n")

    private data class SectionState(
        val id: String,
        val specificationNumber: String,
        val number: String,
        val title: String,
        val path: String,
        val level: Int,
        val kind: String = "",
        val printedNumber: String = "",
    ) {
        companion object {
            fun root(docHash: String) = SectionState(stableId("$docHash|root"), "", "", "", "", 0)

            fun from(docHash: String, stack: List<Segment.Heading>, ordinal: Int): SectionState {
                val leaf = stack.last()
                val specification = stack.asReversed().firstNotNullOfOrNull { it.specificationNumber } ?: ""
                val path = stack.joinToString(" > ") { it.text }
                val key = "$docHash|$specification|${leaf.number.orEmpty()}|${leaf.title}|$ordinal"
                return SectionState(
                    stableId(key), specification, leaf.number.orEmpty(), leaf.title, path, leaf.level,
                    leaf.kind, leaf.printedNumber,
                )
            }

            private fun stableId(value: String): String = MessageDigest.getInstance("SHA-256")
                .digest(value.toByteArray())
                .take(12)
                .joinToString("") { "%02x".format(it) }
        }
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
