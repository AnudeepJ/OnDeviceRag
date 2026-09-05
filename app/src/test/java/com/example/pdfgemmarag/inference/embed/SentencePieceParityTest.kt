package com.example.pdfgemmarag.inference.embed

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.security.MessageDigest

/**
 * Spike 2 parity check: the pure-Kotlin tokenizer must produce exactly the ids the Python
 * `sentencepiece` reference produces for the real Gemma tokenizer on 50 Latin/CJK strings.
 *
 * Fixtures: `src/test/resources/tokenizer_fixtures.json` (see `scripts/gen_tokenizer_fixtures.py`).
 * Model: `$SPM_MODEL`, or `~/.cache/ondevice-rag/tokenizer.model` (4.7 MB, not committed).
 * The test is skipped (not failed) when the model file is absent so CI without the file stays green.
 */
class SentencePieceParityTest {

    private val fixtures: JSONObject by lazy {
        val stream = checkNotNull(javaClass.classLoader?.getResourceAsStream("tokenizer_fixtures.json")) { "fixtures missing" }
        JSONObject(stream.reader().readText())
    }

    private fun locateModel(): File? {
        val candidates = listOfNotNull(
            System.getenv("SPM_MODEL")?.let(::File),
            File(System.getProperty("user.home"), ".cache/ondevice-rag/tokenizer.model"),
        )
        return candidates.firstOrNull { it.isFile }
    }

    private fun loadVerified(): SentencePieceTokenizer {
        val file = locateModel()
        assumeTrue("tokenizer.model not available; skipping parity test", file != null)
        val bytes = file!!.readBytes()
        val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        assertEquals("fixture/model mismatch", fixtures.getString("model_sha256"), sha)
        return SentencePieceTokenizer.parse(bytes)
    }

    @Test
    fun `special ids and vocab match the reference`() {
        val tok = loadVerified()
        assertEquals(fixtures.getInt("vocab_size"), tok.vocabSize)
        assertEquals(fixtures.getInt("bos"), tok.bosId)
        assertEquals(fixtures.getInt("eos"), tok.eosId)
        assertEquals(fixtures.getInt("pad"), tok.padId)
        assertEquals(fixtures.getInt("unk"), tok.unkId)
    }

    @Test
    fun `encode matches python sentencepiece on all fixtures`() {
        val tok = loadVerified()
        val cases = fixtures.getJSONArray("cases")
        val failures = StringBuilder()
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val text = c.getString("text")
            val expected = c.getJSONArray("ids").let { arr -> List(arr.length()) { arr.getInt(it) } }
            val got = tok.encode(text).toList()
            if (got != expected) {
                val expPieces = expected.map(tok::idToPiece)
                val gotPieces = got.map(tok::idToPiece)
                failures.appendLine("'${text.replace("\n", "\\n")}'\n   expected $expPieces\n   got      $gotPieces")
            }
        }
        assertEquals("tokenizer parity failures:\n$failures", "", failures.toString())
    }

    @Test
    fun `decode matches python sentencepiece on all fixtures`() {
        val tok = loadVerified()
        val cases = fixtures.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val ids = c.getJSONArray("ids").let { arr -> IntArray(arr.length()) { arr.getInt(it) } }
            assertEquals("decode of '${c.getString("text")}'", c.getString("decoded"), tok.decode(ids))
        }
    }
}
