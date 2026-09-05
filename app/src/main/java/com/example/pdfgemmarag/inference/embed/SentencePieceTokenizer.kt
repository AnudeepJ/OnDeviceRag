package com.example.pdfgemmarag.inference.embed

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.text.Normalizer
import java.util.PriorityQueue

/**
 * Pure-Kotlin SentencePiece BPE encoder that reads the standard `tokenizer.model` protobuf.
 *
 * Why not DJL: `ai.djl.sentencepiece` downloads desktop JNI binaries at runtime and has no Android
 * ABI. Gemma's tokenizer (shared by EmbeddingGemma) is a BPE model with byte fallback, no dummy
 * prefix, whitespace escaped to U+2581 and an identity normalizer, all of which this class handles.
 * If a model ships a precompiled charsmap we approximate with NFKC and flag it in [normalizerNote];
 * the on-device parity test against the Python reference must be green before shipping.
 */
class SentencePieceTokenizer private constructor(
    private val pieces: List<String>,
    private val scores: FloatArray,
    private val types: IntArray,
    private val pieceToId: HashMap<String, Int>,
    val unkId: Int,
    val bosId: Int,
    val eosId: Int,
    val padId: Int,
    private val addDummyPrefix: Boolean,
    private val removeExtraWhitespaces: Boolean,
    private val escapeWhitespaces: Boolean,
    private val byteFallback: Boolean,
    private val nfkcApprox: Boolean,
    val modelType: Int,
) {
    val vocabSize: Int get() = pieces.size
    val normalizerNote: String
        get() = if (nfkcApprox) "precompiled charsmap present; using NFKC approximation" else "identity normalizer"

    private val byteIds: IntArray by lazy {
        IntArray(256) { b -> pieceToId["<0x%02X>".format(b)] ?: unkId }
    }

    fun idToPiece(id: Int): String = pieces[id]

    /** User-defined symbols (e.g. `<start_of_turn>`) are matched verbatim before BPE, longest first. */
    private val userDefined: Map<Int, HashSet<String>> by lazy {
        val byLen = HashMap<Int, HashSet<String>>()
        pieces.forEachIndexed { id, p -> if (types[id] == TYPE_USER_DEFINED) byLen.getOrPut(p.length) { HashSet() }.add(p) }
        byLen
    }
    private val userDefinedLengths: IntArray by lazy { userDefined.keys.sortedDescending().toIntArray() }

    /** Encodes [text] to ids without BOS/EOS. */
    fun encode(text: String): IntArray {
        val normalized = normalize(text)
        if (normalized.isEmpty()) return IntArray(0)
        val symbols = ArrayList<String>()
        var segStart = 0
        var i = 0
        while (i < normalized.length) {
            val match = matchUserDefined(normalized, i)
            if (match != null) {
                if (i > segStart) symbols += bpe(normalized.substring(segStart, i))
                symbols += match
                i += match.length
                segStart = i
            } else {
                i += Character.charCount(normalized.codePointAt(i))
            }
        }
        if (segStart < normalized.length) symbols += bpe(normalized.substring(segStart))
        val out = ArrayList<Int>(symbols.size + 8)
        for (s in symbols) {
            val id = pieceToId[s]
            if (id != null && types[id] != TYPE_UNUSED) {
                out += id
            } else if (byteFallback) {
                for (b in s.toByteArray(StandardCharsets.UTF_8)) out += byteIds[b.toInt() and 0xFF]
            } else {
                out += unkId
            }
        }
        return out.toIntArray()
    }

    /** Encodes with control tokens and truncates to [maxLen], always preserving EOS when requested. */
    fun encodeForModel(text: String, maxLen: Int, addBos: Boolean = true, addEos: Boolean = true): IntArray {
        val body = encode(text)
        val reserve = (if (addBos) 1 else 0) + (if (addEos) 1 else 0)
        val keep = minOf(body.size, maxLen - reserve)
        val out = IntArray(keep + reserve)
        var i = 0
        if (addBos) out[i++] = bosId
        System.arraycopy(body, 0, out, i, keep); i += keep
        if (addEos) out[i] = eosId
        return out
    }

    fun decode(ids: IntArray): String {
        val sb = StringBuilder()
        val pendingBytes = ArrayList<Byte>()
        fun flushBytes() {
            if (pendingBytes.isNotEmpty()) {
                sb.append(String(pendingBytes.toByteArray(), StandardCharsets.UTF_8)); pendingBytes.clear()
            }
        }
        for (id in ids) {
            if (id < 0 || id >= pieces.size) continue
            val p = pieces[id]
            if (types[id] == TYPE_BYTE) {
                pendingBytes += p.substring(3, 5).toInt(16).toByte(); continue
            }
            flushBytes()
            if (types[id] == TYPE_CONTROL) continue
            sb.append(p)
        }
        flushBytes()
        return sb.toString().replace(SPACE_MARKER, ' ').let { if (addDummyPrefix) it.removePrefix(" ") else it }
    }

    private val userDefinedFirstChars: Set<Char> by lazy { userDefined.values.flatten().map { it[0] }.toHashSet() }

    private fun matchUserDefined(s: String, at: Int): String? {
        if (s[at] !in userDefinedFirstChars) return null
        for (len in userDefinedLengths) {
            if (at + len > s.length) continue
            val candidate = s.substring(at, at + len)
            if (userDefined[len]?.contains(candidate) == true) return candidate
        }
        return null
    }

    // ---------------------------------------------------------------- normalization

    private fun normalize(text: String): String {
        var t = if (nfkcApprox) Normalizer.normalize(text, Normalizer.Form.NFKC) else text
        if (removeExtraWhitespaces) t = t.trim().replace(Regex(" +"), " ")
        if (addDummyPrefix && t.isNotEmpty()) t = " $t"
        if (escapeWhitespaces) t = t.replace(' ', SPACE_MARKER)
        return t
    }

    // ---------------------------------------------------------------- BPE

    private class Symbol(var text: String, var prev: Int, var next: Int, var alive: Boolean = true)
    private class Merge(val left: Int, val right: Int, val score: Float, val size: Int)

    /** Greedy merge by piece score, ties broken by leftmost position, matching sentencepiece's bpe_model.cc. */
    private fun bpe(input: String): List<String> {
        val symbols = ArrayList<Symbol>()
        var i = 0
        while (i < input.length) {
            val cp = input.codePointAt(i)
            val n = Character.charCount(cp)
            symbols += Symbol(input.substring(i, i + n), symbols.size - 1, symbols.size + 1)
            i += n
        }
        if (symbols.isEmpty()) return emptyList()
        symbols.last().next = -1

        val agenda = PriorityQueue<Merge> { a, b ->
            if (a.score != b.score) if (a.score > b.score) -1 else 1 else a.left.compareTo(b.left)
        }
        fun maybeAdd(left: Int, right: Int) {
            if (left < 0 || right < 0) return
            val merged = symbols[left].text + symbols[right].text
            val id = pieceToId[merged] ?: return
            if (types[id] == TYPE_UNUSED) return
            agenda += Merge(left, right, scores[id], merged.length)
        }
        for (k in 0 until symbols.size - 1) maybeAdd(k, k + 1)

        while (agenda.isNotEmpty()) {
            val m = agenda.poll() ?: break
            val l = symbols[m.left]; val r = symbols[m.right]
            if (!l.alive || !r.alive || l.next != m.right) continue
            if (l.text.length + r.text.length != m.size) continue
            l.text += r.text
            r.alive = false
            l.next = r.next
            if (r.next >= 0) symbols[r.next].prev = m.left
            maybeAdd(l.prev, m.left)
            maybeAdd(m.left, l.next)
        }
        val out = ArrayList<String>()
        var cur = 0
        while (cur >= 0) {
            val s = symbols[cur]
            if (s.alive) out += s.text
            cur = s.next
        }
        return out
    }

    companion object {
        const val SPACE_MARKER = '\u2581'
        const val TYPE_NORMAL = 1
        const val TYPE_UNKNOWN = 2
        const val TYPE_CONTROL = 3
        const val TYPE_USER_DEFINED = 4
        const val TYPE_UNUSED = 5
        const val TYPE_BYTE = 6
        const val MODEL_UNIGRAM = 1
        const val MODEL_BPE = 2

        fun load(file: File): SentencePieceTokenizer = parse(file.readBytes())

        fun parse(bytes: ByteArray): SentencePieceTokenizer {
            val pieces = ArrayList<String>()
            val scores = ArrayList<Float>()
            val types = ArrayList<Int>()
            var addDummyPrefix = true
            var removeExtraWhitespaces = true
            var escapeWhitespaces = true
            var charsmapPresent = false
            var byteFallback = false
            var modelType = MODEL_BPE
            var unkId = 0; var bosId = 1; var eosId = 2; var padId = -1

            val root = ProtoReader(bytes)
            while (root.hasMore()) {
                val (field, wire) = root.tag()
                when {
                    field == 1 && wire == 2 -> { // SentencePiece
                        val sub = ProtoReader(root.bytes())
                        var piece = ""; var score = 0f; var type = TYPE_NORMAL
                        while (sub.hasMore()) {
                            val (f, w) = sub.tag()
                            when {
                                f == 1 && w == 2 -> piece = sub.string()
                                f == 2 && w == 5 -> score = sub.float()
                                f == 3 && w == 0 -> type = sub.varint().toInt()
                                else -> sub.skip(w)
                            }
                        }
                        pieces += piece; scores += score; types += type
                    }
                    field == 2 && wire == 2 -> { // TrainerSpec
                        val sub = ProtoReader(root.bytes())
                        while (sub.hasMore()) {
                            val (f, w) = sub.tag()
                            when {
                                f == 3 && w == 0 -> modelType = sub.varint().toInt()
                                f == 35 && w == 0 -> byteFallback = sub.varint() != 0L
                                f == 40 && w == 0 -> unkId = sub.varint().toInt()
                                f == 41 && w == 0 -> bosId = sub.varint().toInt()
                                f == 42 && w == 0 -> eosId = sub.varint().toInt()
                                f == 43 && w == 0 -> padId = sub.varint().toInt()
                                else -> sub.skip(w)
                            }
                        }
                    }
                    field == 3 && wire == 2 -> { // NormalizerSpec
                        val sub = ProtoReader(root.bytes())
                        while (sub.hasMore()) {
                            val (f, w) = sub.tag()
                            when {
                                f == 2 && w == 2 -> charsmapPresent = sub.bytes().isNotEmpty()
                                f == 3 && w == 0 -> addDummyPrefix = sub.varint() != 0L
                                f == 4 && w == 0 -> removeExtraWhitespaces = sub.varint() != 0L
                                f == 5 && w == 0 -> escapeWhitespaces = sub.varint() != 0L
                                else -> sub.skip(w)
                            }
                        }
                    }
                    else -> root.skip(wire)
                }
            }
            require(pieces.isNotEmpty()) { "Not a SentencePiece model: no pieces" }
            require(modelType == MODEL_BPE) { "Only BPE SentencePiece models are supported (type=$modelType)" }
            val map = HashMap<String, Int>(pieces.size * 2)
            pieces.forEachIndexed { idx, p -> map.putIfAbsent(p, idx) }
            // Some exports leave byte_fallback unset while still shipping <0x..> pieces.
            if (!byteFallback && map.containsKey("<0x00>") && map.containsKey("<0xFF>")) byteFallback = true
            return SentencePieceTokenizer(
                pieces, scores.toFloatArray(), types.toIntArray(), map,
                unkId, bosId, eosId, padId,
                addDummyPrefix, removeExtraWhitespaces, escapeWhitespaces, byteFallback, charsmapPresent, modelType,
            )
        }
    }

    /** Minimal protobuf wire-format reader (varint, fixed32/64, length-delimited). */
    private class ProtoReader(private val buf: ByteArray, private var pos: Int = 0, private val end: Int = buf.size) {
        fun hasMore() = pos < end
        fun tag(): Pair<Int, Int> { val v = varint().toInt(); return (v ushr 3) to (v and 7) }
        fun varint(): Long {
            var result = 0L; var shift = 0
            while (true) {
                val b = buf[pos++].toInt()
                result = result or ((b and 0x7F).toLong() shl shift)
                if (b and 0x80 == 0) return result
                shift += 7
            }
        }
        fun bytes(): ByteArray { val len = varint().toInt(); val out = buf.copyOfRange(pos, pos + len); pos += len; return out }
        fun string(): String = String(bytes(), StandardCharsets.UTF_8)
        fun float(): Float { val f = ByteBuffer.wrap(buf, pos, 4).order(ByteOrder.LITTLE_ENDIAN).float; pos += 4; return f }
        fun skip(wire: Int) {
            when (wire) {
                0 -> varint()
                1 -> pos += 8
                2 -> { val len = varint().toInt(); pos += len }
                5 -> pos += 4
                else -> throw IllegalStateException("Unsupported wire type $wire")
            }
        }
    }
}
