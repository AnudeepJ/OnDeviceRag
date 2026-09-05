package com.example.pdfgemmarag.inference.embed

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Exercises the protobuf reader and the BPE merge loop against a hand-built model. Parity against
 * the real Gemma `tokenizer.model` is an on-device self-test (needs the 4.7 MB file).
 */
class SentencePieceTokenizerTest {

    // ---- minimal protobuf writer -------------------------------------------------------------
    private class Proto {
        val out = ByteArrayOutputStream()
        fun varint(v: Long) { var x = v; while (true) { val b = (x and 0x7F).toInt(); x = x ushr 7; if (x == 0L) { out.write(b); return } else out.write(b or 0x80) } }
        fun tag(field: Int, wire: Int) = varint(((field shl 3) or wire).toLong())
        fun string(field: Int, s: String) { tag(field, 2); val b = s.toByteArray(); varint(b.size.toLong()); out.write(b) }
        fun bytes(field: Int, b: ByteArray) { tag(field, 2); varint(b.size.toLong()); out.write(b) }
        fun float(field: Int, f: Float) { tag(field, 5); out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(f).array()) }
        fun int(field: Int, v: Int) { tag(field, 0); varint(v.toLong()) }
        fun bool(field: Int, v: Boolean) = int(field, if (v) 1 else 0)
        fun bytes() = out.toByteArray()
    }

    private fun piece(text: String, score: Float, type: Int = SentencePieceTokenizer.TYPE_NORMAL) =
        Proto().apply { string(1, text); float(2, score); int(3, type) }.bytes()

    private fun model(addDummyPrefix: Boolean = false): ByteArray {
        val m = Proto()
        val pieces = mutableListOf(
            piece("<pad>", 0f, SentencePieceTokenizer.TYPE_CONTROL),
            piece("<eos>", 0f, SentencePieceTokenizer.TYPE_CONTROL),
            piece("<bos>", 0f, SentencePieceTokenizer.TYPE_CONTROL),
            piece("<unk>", 0f, SentencePieceTokenizer.TYPE_UNKNOWN),
        )
        for (b in 0 until 256) pieces += piece("<0x%02X>".format(b), 0f, SentencePieceTokenizer.TYPE_BYTE)
        // Vocabulary: higher score merges first. "▁he" + "llo" vs "▁hel" + "lo": scores decide.
        val vocab = listOf(
            "▁" to -1f, "h" to -2f, "e" to -2f, "l" to -2f, "o" to -2f, "w" to -2f, "r" to -2f, "d" to -2f,
            "▁h" to -3f, "he" to -3.5f, "▁he" to -1.5f, "ll" to -1.2f, "llo" to -1.1f, "▁hello" to -0.5f,
            "▁w" to -3f, "or" to -3f, "▁wor" to -2.5f, "ld" to -2.8f, "▁world" to -0.6f,
            "東" to -4f, "京" to -4f, "東京" to -1f,
        )
        for ((t, s) in vocab) pieces += piece(t, s)
        for (p in pieces) m.bytes(1, p)
        val trainer = Proto().apply { int(3, SentencePieceTokenizer.MODEL_BPE); bool(35, true); int(40, 3); int(41, 2); int(42, 1); int(43, 0) }
        m.bytes(2, trainer.bytes())
        val normalizer = Proto().apply { string(1, "identity"); bool(3, addDummyPrefix); bool(4, false); bool(5, true) }
        m.bytes(3, normalizer.bytes())
        return m.bytes()
    }

    private val tok = SentencePieceTokenizer.parse(model())

    private fun id(piece: String) = (0 until tok.vocabSize).first { tok.idToPiece(it) == piece }

    @Test
    fun `control ids come from trainer spec`() {
        assertEquals(0, tok.padId); assertEquals(1, tok.eosId); assertEquals(2, tok.bosId); assertEquals(3, tok.unkId)
    }

    @Test
    fun `greedy merges by score produce whole-word pieces`() {
        val ids = tok.encode(" hello world")
        assertEquals(listOf("▁hello", "▁world"), ids.map { tok.idToPiece(it) })
        // Without a leading space the first word cannot take the ▁-prefixed piece.
        assertEquals(listOf("he", "llo", "▁world"), tok.encode("hello world").map { tok.idToPiece(it) })
    }

    @Test
    fun `no dummy prefix means a leading word has no space marker`() {
        // First "hello" has no leading space: merges to he + llo (no ▁hello since no ▁).
        val ids = tok.encode("hello")
        assertEquals(listOf("he", "llo"), ids.map { tok.idToPiece(it) })
    }

    @Test
    fun `dummy prefix option prepends a space`() {
        val t2 = SentencePieceTokenizer.parse(model(addDummyPrefix = true))
        assertEquals(listOf("▁hello"), t2.encode("hello").map { t2.idToPiece(it) })
        assertEquals("hello", t2.decode(t2.encode("hello")))
    }

    @Test
    fun `unknown characters fall back to utf-8 bytes`() {
        val ids = tok.encode("東京ü")
        val pieces = ids.map { tok.idToPiece(it) }
        assertEquals("東京", pieces[0])
        assertEquals(listOf("<0xC3>", "<0xBC>"), pieces.drop(1))
        assertEquals("東京ü", tok.decode(ids))
    }

    @Test
    fun `encodeForModel adds bos eos and truncates`() {
        val ids = tok.encodeForModel(" hello world hello world", maxLen = 4)
        assertEquals(4, ids.size)
        assertEquals(tok.bosId, ids.first())
        assertEquals(tok.eosId, ids.last())
        assertArrayEquals(intArrayOf(id("▁hello"), id("▁world")), ids.copyOfRange(1, 3))
    }

    @Test
    fun `round trip preserves whitespace runs`() {
        val s = "hello  world"
        assertEquals(s, tok.decode(tok.encode(s)))
    }
}
