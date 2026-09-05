package com.example.pdfgemmarag.inference.service

import android.content.Context
import android.os.SystemClock
import androidx.appsearch.app.EmbeddingVector
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.eval.RetrievalEval
import com.example.pdfgemmarag.eval.RetrievalEvalDataset
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import com.example.pdfgemmarag.inference.embed.SentencePieceTokenizer
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.PdfChunkDocument

/**
 * On-device execution of the Phase 0 spikes that need real hardware. Output is shown on the
 * Diagnostics screen and should be pasted into `docs/SPIKES.md` for each target device.
 *
 * - Spike 1 (JNI coexistence): implicitly verified by this process having loaded both
 *   `liblitertlm_jni.so` and `liblitert_jni.so`/`libLiteRt.so` without a linker failure.
 * - Spike 2 (EmbeddingGemma + tokenizer): tokenizer fixtures + embedding latency + cosine sanity.
 * - Spike 3 (AppSearch hybrid): feature support, put/search/remove, CJK keyword hit, latency.
 */
class SelfTest(
    private val context: Context,
    private val embedderProvider: suspend () -> EmbeddingGemmaEmbedder,
    private val storeProvider: suspend () -> AppSearchVectorStore,
) {

    suspend fun run(): String {
        val out = StringBuilder()
        out.appendLine("== Spike 1: native coexistence ==")
        out.appendLine(nativeLibs())
        out.appendLine()
        out.appendLine("== Spike 2: EmbeddingGemma + SentencePiece ==")
        runCatching { embedderChecks(out) }.onFailure { out.appendLine("FAILED: ${it.message}") }
        out.appendLine()
        out.appendLine("== Spike 3: AppSearch 1.1.0 hybrid ==")
        runCatching { appSearchChecks(out) }.onFailure { out.appendLine("FAILED: ${it.message}") }
        out.appendLine()
        out.appendLine("== Retrieval eval (page-hit@5) ==")
        runCatching { retrievalEval(out) }.onFailure { out.appendLine("FAILED: ${it.message}") }
        return out.toString()
    }

    private fun nativeLibs(): String {
        // Explicitly resolve both JNI shims against the single packaged libLiteRt.so.
        val results = listOf("LiteRt", "litertlm_jni", "litert_jni").map { lib ->
            "  $lib: " + runCatching { System.loadLibrary(lib); "loaded" }.getOrElse { "FAILED ${it.message}" }
        }
        // Libraries are mmapped from base.apk (useLegacyPackaging=false), so maps show the APK, not .so names.
        val maps = runCatching { java.io.File("/proc/self/maps").readLines() }.getOrDefault(emptyList())
        val soPaths = maps.mapNotNull { line -> Regex("(/[^ ]+\\.so)$").find(line)?.groupValues?.get(1) }
            .filter { it.contains("litert", ignoreCase = true) || it.contains("PDFNet") || it.contains("mlkit", true) }
            .toSortedSet()
        return results.joinToString("\n") + "\n  extracted .so paths: ${if (soPaths.isEmpty()) "none (all in-APK)" else soPaths}"
    }

    private suspend fun embedderChecks(out: StringBuilder) {
        if (!ModelPaths.embeddingReady(context)) { out.appendLine("SKIPPED: embedding model/tokenizer not installed"); return }
        val t0 = SystemClock.elapsedRealtime()
        val embedder = embedderProvider()
        out.appendLine("embedder backend=${embedder.backend} seq=${embedder.sequenceLength} init=${SystemClock.elapsedRealtime() - t0}ms")
        val tok: SentencePieceTokenizer = embedder.tokenizer
        out.appendLine("tokenizer vocab=${tok.vocabSize} bos=${tok.bosId} eos=${tok.eosId} pad=${tok.padId} unk=${tok.unkId} (${tok.normalizerNote})")

        // Tokenizer round-trip fixtures. Reference ids for these strings come from the Python
        // `sentencepiece` package; fill TOKENIZER_FIXTURES with them to turn this into a hard parity check.
        val samples = listOf(
            "Hello world", "The quick brown fox jumps over the lazy dog.", "東京都は日本の首都です。",
            "机器学习是人工智能的一个分支。", "안녕하세요, 만나서 반갑습니다.", "Straße Ärger café naïve 12345 €", "  leading spaces\tand\ttabs",
        )
        var roundTripOk = 0
        for (s in samples) {
            val ids = tok.encode(s)
            val back = tok.decode(ids)
            val ok = back == s
            if (ok) roundTripOk++
            out.appendLine("  ${if (ok) "ok " else "DIFF"} tokens=${ids.size.toString().padStart(3)} '${s.take(30)}'${if (!ok) " -> '$back'" else ""}")
        }
        out.appendLine("round-trip $roundTripOk/${samples.size}")
        tokenizerParity(tok, out)

        // Cosine sanity + latency
        val q = embedder.embedQuery("What is the capital of Japan?")
        val t1 = SystemClock.elapsedRealtime()
        val a = embedder.embedDocument("Tokyo is the capital city of Japan and its largest metropolis.")
        val perChunk = SystemClock.elapsedRealtime() - t1
        val b = embedder.embedDocument("The recipe calls for two cups of flour and one egg.")
        val c = embedder.embedDocument("東京は日本の首都であり、最大の都市です。")
        val ca = EmbeddingGemmaEmbedder.cosine(q, a); val cb = EmbeddingGemmaEmbedder.cosine(q, b); val cc = EmbeddingGemmaEmbedder.cosine(q, c)
        out.appendLine("cosine(q, relevant-en)=${"%.3f".format(ca)} cosine(q, unrelated)=${"%.3f".format(cb)} cosine(q, relevant-ja)=${"%.3f".format(cc)}")
        out.appendLine("sanity ${if (ca > cb && cc > cb) "ok" else "FAILED"}; latency/chunk=${perChunk}ms dim=${a.size}")
    }

    private suspend fun appSearchChecks(out: StringBuilder) {
        val store = storeProvider()
        out.appendLine(store.features.toString().prependIndent("  "))
        out.appendLine("hybridOk=${store.features.hybridOk}")
        val ns = "selftest-${System.currentTimeMillis()}"
        // Pseudo-random unit vectors are near-orthogonal, so keyword-only queries are not swamped by
        // accidental vector similarity (a patterned generator made every doc score ~0.99 against every query).
        fun vec(seed: Int): FloatArray {
            val r = kotlin.random.Random(seed)
            val v = FloatArray(512) { r.nextFloat() - 0.5f }
            EmbeddingGemmaEmbedder.l2Normalize(v); return v
        }
        val docs = listOf(
            "東京都は日本の首都です。人口は約千四百万人。" to 1,
            "北京是中华人民共和国的首都。" to 2,
            "Paris is the capital of France." to 3,
            "| Year | Revenue |\n| 2023 | 4.2M |" to 4,
        ).mapIndexed { i, (text, seed) ->
            PdfChunkDocument().apply {
                namespace = ns; id = "$ns:$i"; creationTimestampMillis = System.currentTimeMillis()
                this.text = text; pageNumber = i + 1; chunkIndex = i; docName = "selftest"; script = "MIXED"; pageCount = 4
                embedding = EmbeddingVector(vec(seed), EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
            }
        }
        val t0 = SystemClock.elapsedRealtime()
        store.putChunks(docs)
        out.appendLine("put ${docs.size} docs in ${SystemClock.elapsedRealtime() - t0}ms")

        suspend fun query(label: String, text: String, seed: Int, expectId: String) {
            val t = SystemClock.elapsedRealtime()
            val hits = store.search(ns, text, vec(seed), topK = 3, similarityFloor = 0.0)
            val top = hits.firstOrNull()
            out.appendLine("  $label -> top=${top?.chunkId?.substringAfter(':')} score=${"%.3f".format(top?.score ?: 0.0)} hits=${hits.size} ${SystemClock.elapsedRealtime() - t}ms ${if (top?.chunkId == expectId) "ok" else "UNEXPECTED"}")
        }
        query("vector-only (seed 3)", "zzzz", 3, "$ns:2")
        query("ja keyword 首都", "首都", 99, "$ns:0")
        query("zh keyword 北京", "北京", 99, "$ns:1")
        query("en keyword Paris", "Paris", 99, "$ns:2")
        query("table keyword Revenue", "Revenue", 99, "$ns:3")
        val cit = store.getCitation("$ns:0")
        out.appendLine("getCitation ${if (cit?.pageNumber == 1) "ok" else "FAILED"}")
        store.removeDocument(ns)
        val after = store.search(ns, "首都", vec(1), 3, 0.0)
        out.appendLine("removeByNamespace ${if (after.isEmpty()) "ok" else "FAILED (${after.size} left)"}")
    }

    private suspend fun retrievalEval(out: StringBuilder) {
        if (!ModelPaths.embeddingReady(context)) { out.appendLine("SKIPPED: embedding model not installed"); return }
        val embedder = embedderProvider()
        val store = storeProvider()
        val ns = RetrievalEvalDataset.DOC_HASH
        store.removeDocument(ns)
        val now = System.currentTimeMillis()
        val docs = RetrievalEvalDataset.chunks().map { c ->
            PdfChunkDocument().apply {
                namespace = ns
                id = "$ns:${c.chunkIndex}"
                creationTimestampMillis = now
                text = c.text
                pageNumber = c.page
                chunkIndex = c.chunkIndex
                docName = RetrievalEvalDataset.DOC_NAME
                script = c.lang
                pageCount = 250
                embedding = EmbeddingVector(embedder.embedDocument(c.text), EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
            }
        }
        val tPut = SystemClock.elapsedRealtime()
        store.putChunks(docs)
        out.appendLine("indexed ${docs.size} chunks in ${SystemClock.elapsedRealtime() - tPut}ms")
        try {
            val report = RetrievalEval.run(store, ns, embed = { embedder.embedQuery(it) })
            out.append(report.toString())
        } finally {
            store.removeDocument(ns)
        }
    }

    /** Hard parity check against Python `sentencepiece` ids (assets/tokenizer_fixtures.json, 50 CJK/Latin strings). */
    private fun tokenizerParity(tok: SentencePieceTokenizer, out: StringBuilder) {
        val json = runCatching { context.assets.open("tokenizer_fixtures.json").bufferedReader().readText() }
            .getOrElse { out.appendLine("parity fixtures missing: ${it.message}"); return }
        val root = org.json.JSONObject(json)
        if (root.optInt("vocab_size") != tok.vocabSize) {
            out.appendLine("parity SKIPPED: fixtures are for vocab=${root.optInt("vocab_size")}, installed tokenizer vocab=${tok.vocabSize}")
            return
        }
        val cases = root.getJSONArray("cases")
        var ok = 0
        val t0 = SystemClock.elapsedRealtime()
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val expected = c.getJSONArray("ids").let { a -> List(a.length()) { a.getInt(it) } }
            val got = tok.encode(c.getString("text")).toList()
            if (got == expected) ok++ else out.appendLine("  MISMATCH '${c.getString("text").take(40)}' got=${got.take(12)} expected=${expected.take(12)}")
        }
        out.appendLine("parity $ok/${cases.length()} vs python sentencepiece (${SystemClock.elapsedRealtime() - t0}ms)")
    }
}
