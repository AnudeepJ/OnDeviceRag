package com.example.pdfgemmarag.inference.chunk

import com.example.pdfgemmarag.inference.ocr.Script
import com.example.pdfgemmarag.inference.ocr.ScriptDetector
import com.example.pdfgemmarag.inference.pdf.PageContent
import com.example.pdfgemmarag.inference.pdf.Segment
import com.example.pdfgemmarag.inference.pdf.TableIdentity
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
    /** Stable parent node; empty only for the document root. */
    val parentSectionId: String = "",
    val positionInSection: Int = 0,
    val contentKind: String = if (isTable) "TABLE" else "PARAGRAPH",
    val identifierAtoms: List<String> = emptyList(),
    val continuesFromChunkIndex: Int? = null,
    val continuesToChunkIndex: Int? = null,
    /** Caption of the table this chunk belongs to (`Table 03210B - Minimum cover`); blank otherwise. */
    val tableCaption: String = "",
    /** Stable table identity shared by every chunk of one grid. */
    val tableId: String = "",
    /** Printed table number (`5.1`, `03210B`); blank when the grid has no caption number. */
    val tableNumber: String = "",
    /** Stable identity and completeness metadata for one logical extracted list. */
    val listId: String = "",
    val listItemStart: Int = 0,
    val listItemCount: Int = 0,
    val listComplete: Boolean = false,
    /** Canonical cells are captured before token-window splitting. */
    val tableCells: List<CanonicalTableCell> = emptyList(),
    /** All embedding fragments derived from one original table chunk share this identity. */
    val tableFragmentGroupId: String = "",
)

data class CanonicalTableCell(
    val rowIndex: Int,
    val columnIndex: Int,
    val rowHeader: String,
    val columnHeaderPath: List<String>,
    val text: String,
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
        val sectionStateStack = ArrayList<SectionState>()
        var section = SectionState.root(docHash)
        var position = 0
        var tableOrdinal = 0
        var listOrdinal = 0

        fun add(
            page: Int,
            body: String,
            table: Boolean,
            kind: String,
            caption: String = "",
            tableId: String = "",
            tableNumber: String = "",
            listId: String = "",
            listItemStart: Int = 0,
            listItemCount: Int = 0,
            listComplete: Boolean = false,
            tableCells: List<CanonicalTableCell> = emptyList(),
            tableFragmentGroupId: String = "",
        ) {
            val clean = body.trim()
            if (clean.isEmpty()) return
            val retrieval = retrievalText(section.path, clean, table, caption)
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
                parentSectionId = section.parentSectionId,
                positionInSection = position++,
                contentKind = kind,
                identifierAtoms = IdentifierAtoms.extract(retrieval),
                tableCaption = caption,
                tableId = tableId,
                tableNumber = tableNumber,
                listId = listId,
                listItemStart = listItemStart,
                listItemCount = listItemCount,
                listComplete = listComplete,
                tableCells = tableCells,
                tableFragmentGroupId = tableFragmentGroupId,
            )
        }

        for (page in pages) {
            for (segment in page.segments) {
                when (segment) {
                    is Segment.Heading -> {
                        while (sectionStack.isNotEmpty() && closesCurrentHeading(sectionStack, segment)) {
                            sectionStack.removeAt(sectionStack.lastIndex)
                            sectionStateStack.removeAt(sectionStateStack.lastIndex)
                        }
                        sectionStack += segment
                        ordinal++
                        section = SectionState.from(
                            docHash, sectionStack, ordinal,
                            parentSectionId = sectionStateStack.lastOrNull()?.id.orEmpty(),
                        )
                        sectionStateStack += section
                        position = 0
                        add(page.pageNumber, segment.text, table = false, kind = "HEADING")
                    }
                    is Segment.Paragraph -> {
                        for (text in chunkParagraph(segment.text)) {
                            add(page.pageNumber, text, table = false, kind = "PARAGRAPH")
                        }
                    }
                    is Segment.ListBlock -> {
                        listOrdinal++
                        val id = SectionState.stableId("$docHash|${section.id}|${page.pageNumber}|list|$listOrdinal")
                        var itemStart = 0
                        val listChunks = chunkList(segment)
                        for ((chunkOffset, text) in listChunks.withIndex()) {
                            val itemCount = listItemStartOffsets(text).size
                            add(
                                page.pageNumber, text, table = false, kind = "LIST",
                                listId = id, listItemStart = itemStart, listItemCount = itemCount,
                                listComplete = chunkOffset == listChunks.lastIndex,
                            )
                            itemStart += itemCount
                        }
                    }
                    is Segment.Table -> {
                        tableOrdinal++
                        val number = segment.tableNumber.ifBlank { TableIdentity.numberFromCaption(segment.caption) }
                        val tableId = TableIdentity.id(docHash, segment.caption, number, page.pageNumber, tableOrdinal)
                        var rowStart = 0
                        for ((pieceIndex, text) in chunkTable(segment).withIndex()) {
                            val cells = canonicalTableCells(text, rowStart)
                            rowStart += text.lines().map(String::trim).count { it.startsWith("|") }
                                .minus(2).coerceAtLeast(0)
                            add(
                                page.pageNumber, text, table = true, kind = "TABLE",
                                caption = segment.caption, tableId = tableId, tableNumber = number,
                                tableCells = cells,
                                tableFragmentGroupId = SectionState.stableId("$tableId|piece|$pieceIndex"),
                            )
                        }
                    }
                }
            }
        }
        return normalizeListChains(linkContinuations(mergeTiny(out)))
    }

    /**
     * PART is level 2 in specification manuals (SECTION -> PART), but level 1 in
     * many books (PART -> CHAPTER). Preserve the latter only when PART itself is
     * a root node; a PART nested below a CHAPTER must still close at the next
     * CHAPTER boundary.
     */
    private fun closesCurrentHeading(stack: List<Segment.Heading>, incoming: Segment.Heading): Boolean {
        val current = stack.last()
        val bookStylePartRoot = stack.first().kind == Segment.Heading.KIND_PART
        if (bookStylePartRoot && incoming.kind == Segment.Heading.KIND_PART) return true
        if (bookStylePartRoot && incoming.kind == Segment.Heading.KIND_CHAPTER) {
            return stack.size > 1
        }
        return current.level >= incoming.level
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
        return normalizeListChains(linkContinuations(indexed))
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
        val pieceStarts = ArrayList<Int>()
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
            if (text.isNotEmpty()) {
                pieceStarts += offset
                out += chunk.copy(
                    text = text,
                    bodyText = text,
                    retrievalText = retrievalFor(chunk, text),
                    script = ScriptDetector.detect(text),
                    identifierAtoms = IdentifierAtoms.extract(retrievalFor(chunk, text)),
                )
            }
            offset = end
            while (offset < body.length && body[offset].isWhitespace()) offset++
        }
        val withCanonicalCells = out.mapIndexed { pieceIndex, piece ->
            piece.copy(tableCells = if (pieceIndex == 0) chunk.tableCells else emptyList())
        }
        if (chunk.listId.isBlank()) return withCanonicalCells
        val itemStarts = listItemStartOffsets(body)
        var itemStart = chunk.listItemStart
        return withCanonicalCells.mapIndexed { pieceIndex, piece ->
            val rangeEnd = pieceStarts.getOrNull(pieceIndex + 1) ?: body.length
            val itemCount = itemStarts.count { it in pieceStarts[pieceIndex] until rangeEnd }
            piece.copy(
                listItemStart = itemStart.also { itemStart += itemCount },
                listItemCount = itemCount,
                listComplete = chunk.listComplete && pieceIndex == withCanonicalCells.lastIndex,
            )
        }
    }

    internal fun mergeTiny(chunks: List<Chunk>): List<Chunk> {
        if (chunks.isEmpty()) return chunks
        val merged = ArrayList<Chunk>()
        var acc: Chunk? = null
        fun flush() {
            val pending = acc ?: return
            val previous = merged.lastOrNull()
            val startsAsContinuation = pending.bodyText.trimStart().firstOrNull()?.isLowerCase() == true
            if (pending.text.length < minKeepChars && startsAsContinuation &&
                previous?.contentKind == "HEADING" && previous.pageNumber == pending.pageNumber &&
                previous.sectionId == pending.sectionId
            ) {
                // A wrapped first sentence can be separated from its numbered heading by PDF
                // geometry (for example a final one-word line). Preserve it on the heading rather
                // than dropping the short orphan below. This keeps the source statement complete.
                val body = previous.bodyText.trimEnd() + " " + pending.bodyText.trim()
                merged[merged.lastIndex] = previous.copy(
                    text = body,
                    bodyText = body,
                    retrievalText = retrievalFor(previous, body),
                    identifierAtoms = IdentifierAtoms.extract(retrievalFor(previous, body)),
                )
            } else {
                merged += pending
            }
            acc = null
        }
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

    internal fun canonicalTableCells(body: String, rowStart: Int = 0): List<CanonicalTableCell> {
        val lines = body.lines().map(String::trim).filter { it.startsWith("|") }
        if (lines.size < 3) return emptyList()
        val headers = lines.first().trim('|').split('|').map(String::trim)
        return lines.drop(2).flatMapIndexed { rowOffset, line ->
            val values = line.trim('|').split('|').map(String::trim)
            if (values.size < 2) return@flatMapIndexed emptyList()
            val rowHeader = values.firstOrNull().orEmpty()
            values.mapIndexedNotNull { columnIndex, value ->
                value.takeIf(String::isNotBlank)?.let {
                    CanonicalTableCell(
                        rowStart + rowOffset,
                        columnIndex,
                        rowHeader,
                        listOfNotNull(headers.getOrNull(columnIndex)?.takeIf(String::isNotBlank)),
                        value,
                    )
                }
            }
        }
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

    private fun listItemStartOffsets(text: String): List<Int> {
        val bullets = BULLET_ITEM_START.findAll(text).map { it.groups[1]!!.range.first }.toList()
        val labelled = LABELLED_ITEM_START.findAll(text).map { it.groups[1]!!.range.first }.toList()
        val firstBullet = bullets.firstOrNull()
        val firstLabel = labelled.firstOrNull()
        // A bullet-led block can end with the next subsection label. That trailing label is a
        // boundary, not another item in the bullet list. Number/letter-led blocks may contain
        // nested bullets, so retain both marker families there.
        val markers = when {
            firstBullet != null && (firstLabel == null || firstBullet < firstLabel) -> bullets
            firstLabel != null -> (bullets + labelled).sorted()
            else -> emptyList()
        }
        if (markers.isNotEmpty()) return markers
        return buildList {
            var lineStart = 0
            text.forEachIndexed { index, char ->
                if (char == '\n') {
                    if (text.substring(lineStart, index).isNotBlank()) add(lineStart)
                    lineStart = index + 1
                }
            }
            if (text.substring(lineStart).isNotBlank()) add(lineStart)
        }
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

    /** Coalesces page-split list blocks into one logical list and assigns global item ordinals. */
    private fun normalizeListChains(chunks: List<Chunk>): List<Chunk> {
        if (chunks.none { it.listId.isNotBlank() }) return chunks
        val canonical = HashMap<String, String>()
        chunks.forEach { chunk ->
            if (chunk.listId.isBlank()) return@forEach
            val previous = chunk.continuesFromChunkIndex?.let(chunks::getOrNull)
            if (previous?.listId?.isNotBlank() == true) {
                canonical[chunk.listId] = canonical[previous.listId] ?: previous.listId
            } else {
                // Later token fragments on this page must not overwrite the canonical id that
                // its first fragment inherited from the previous page.
                canonical.putIfAbsent(chunk.listId, chunk.listId)
            }
        }
        val itemOffsets = HashMap<String, Int>()
        val lastIndexByList = chunks.withIndex().filter { it.value.listId.isNotBlank() }
            .groupBy { canonical[it.value.listId] ?: it.value.listId }
            .mapValues { (_, values) -> values.maxOf { it.index } }
        return chunks.mapIndexed { index, chunk ->
            if (chunk.listId.isBlank()) return@mapIndexed chunk
            val id = canonical[chunk.listId] ?: chunk.listId
            val start = itemOffsets[id] ?: 0
            itemOffsets[id] = start + chunk.listItemCount
            chunk.copy(
                listId = id,
                listItemStart = start,
                listComplete = index == lastIndexByList[id],
            )
        }
    }

    private fun retrievalFor(chunk: Chunk, body: String): String =
        retrievalText(chunk.sectionPath, body.trim(), chunk.isTable, chunk.tableCaption)

    /**
     * Text that is embedded and keyword-indexed. Body text is used as-is under its section path;
     * a table chunk is additionally rendered as header-qualified row facts (`Risk: 21-25;
     * Description: …; Control: Very high`) so a question phrased in column terms lands on the row.
     */
    internal fun retrievalText(sectionPath: String, body: String, table: Boolean, caption: String): String {
        val parts = ArrayList<String>()
        if (sectionPath.isNotBlank()) parts += sectionPath
        if (caption.isNotBlank()) parts += caption
        if (table) {
            val facts = tableRowFacts(body)
            parts += if (facts.isNotBlank()) facts else body
        } else parts += body
        return parts.joinToString("\n")
    }

    /** Renders markdown table rows as `Header: value; …` lines; the first cell names the row when the header's first cell is blank. */
    internal fun tableRowFacts(markdown: String): String {
        val lines = markdown.lines().map(String::trim).filter { it.startsWith("|") }
        if (lines.size < 2) return ""
        fun cells(line: String) = line.trim().trim('|').split('|').map(String::trim)
        val header = cells(lines[0])
        val rows = lines.drop(1).filterNot { line -> cells(line).all { it.isEmpty() || it.all { ch -> ch == '-' || ch == ':' } } }
        if (rows.isEmpty()) return ""
        val matrix = header.firstOrNull().isNullOrBlank()
        return rows.joinToString("\n") { line ->
            val values = cells(line)
            val pairs = values.indices.mapNotNull { k ->
                val value = values[k]
                val name = header.getOrNull(k).orEmpty()
                when {
                    value.isBlank() -> null
                    k == 0 && matrix -> null
                    name.isBlank() -> value
                    else -> "$name: $value"
                }
            }
            val label = if (matrix) values.firstOrNull().orEmpty() + " — " else ""
            label + pairs.joinToString("; ")
        }
    }

    private data class SectionState(
        val id: String,
        val specificationNumber: String,
        val number: String,
        val title: String,
        val path: String,
        val level: Int,
        val kind: String = "",
        val printedNumber: String = "",
        val parentSectionId: String = "",
    ) {
        companion object {
            fun root(docHash: String) = SectionState(stableId("$docHash|root"), "", "", "", "", 0)

            fun from(docHash: String, stack: List<Segment.Heading>, ordinal: Int, parentSectionId: String): SectionState {
                val leaf = stack.last()
                val specification = stack.asReversed().firstNotNullOfOrNull { it.specificationNumber } ?: ""
                val path = stack.joinToString(" > ") { it.text }
                val key = "$docHash|$specification|${leaf.number.orEmpty()}|${leaf.title}|$ordinal"
                return SectionState(
                    stableId(key), specification, leaf.number.orEmpty(), leaf.title, path, leaf.level,
                    leaf.kind, leaf.printedNumber, parentSectionId,
                )
            }

            fun stableId(value: String): String = MessageDigest.getInstance("SHA-256")
                .digest(value.toByteArray())
                .take(12)
                .joinToString("") { "%02x".format(it) }
        }
    }

    companion object {
        private val LABELLED_ITEM_START = Regex(
            "(?m)(?:^|\\s+)(((?:\\d{1,3}|[A-Za-z]|[ivxl]{2,6}|[IVXL]{2,6})[.)]|\\([A-Za-z0-9]{1,4}\\))\\s+)",
        )
        private val BULLET_ITEM_START = Regex(
            "(?m)(?:^|\\s+)([•▪■●○◦‣⁃➢➤►✓✔➔→*\\uE000-\\uF8FF-]\\s+)",
        )
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
