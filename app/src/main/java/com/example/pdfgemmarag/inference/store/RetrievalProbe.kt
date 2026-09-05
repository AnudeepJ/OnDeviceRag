package com.example.pdfgemmarag.inference.store

import android.os.SystemClock
import android.util.Log
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder

/**
 * Runs the Baytown Division 3 test questions (and a few keyword probes) against whatever is
 * already in AppSearch. Does not generate with Gemma — retrieval only.
 */
object RetrievalProbe {

    data class Case(val label: String, val question: String, val expectAny: List<String>)

    val BAYTOWN_CASES = listOf(
        Case(
            "Q1 city+date",
            "What city published these Division 3 concrete specifications, and what is the date?",
            listOf("baytown", "03/2020", "2020"),
        ),
        Case(
            "Q2 formwork payment",
            "Is concrete formwork paid as a separate bid item?",
            listOf("formwork", "payment", "unit price", "no separate"),
        ),
        Case(
            "Q3 plywood standard",
            "What plywood standard and class is required for formwork?",
            listOf("ps 1", "ps1", "class 1", "plywood"),
        ),
        Case("KW Baytown", "Baytown", listOf("baytown")),
        Case("KW formwork", "formwork", listOf("formwork")),
        Case("KW plywood", "plywood", listOf("plywood")),
    )

    suspend fun run(
        store: AppSearchVectorStore,
        embed: (String) -> FloatArray,
        docHash: String? = null,
    ): String {
        val out = StringBuilder()
        val docs = store.listDocuments()
        out.appendLine("indexed documents: ${docs.size}")
        if (docs.isEmpty()) {
            out.appendLine("INDEX EMPTY — Room may still show the PDF. Re-index after the schema-wipe fix.")
            Log.w(TAG, out.toString())
            return out.toString()
        }
        val targets = if (docHash.isNullOrBlank()) docs else docs.filter { it.docHash == docHash }
        for (doc in targets) {
            out.appendLine()
            out.appendLine("== ${doc.displayName} hash=${doc.docHash.take(12)} pages=${doc.pageCount} chunks=${doc.chunkCount} ==")
            val samples = store.sampleChunks(doc.docHash, 3)
            if (samples.isEmpty()) {
                out.appendLine("NAMESPACE EMPTY (listDocuments saw metadata but no chunks)")
                continue
            }
            samples.forEach { c ->
                out.appendLine("  sample p${c.pageNumber} c${c.chunkIndex}: ${c.text.take(140).replace('\n', ' ')}")
            }
            for (c in BAYTOWN_CASES) {
                val t0 = SystemClock.elapsedRealtime()
                val hits = store.search(doc.docHash, c.question, embed(c.question), topK = 8, similarityFloor = 0.3)
                val ms = SystemClock.elapsedRealtime() - t0
                val blob = hits.joinToString("\n") { it.text.lowercase() }
                val matched = c.expectAny.filter { blob.contains(it) }
                val verdict = when {
                    hits.isEmpty() -> "EMPTY"
                    matched.isNotEmpty() -> "HIT ${matched.joinToString(",")}"
                    else -> "MISS (no expected token in top ${hits.size})"
                }
                out.appendLine("  ${c.label}: $verdict ${hits.size} hits ${ms}ms terms=${HybridQuery.keywordTerms(c.question)}")
                hits.take(3).forEach { h ->
                    out.appendLine("    p${h.pageNumber} @${"%.3f".format(h.score)} ${h.text.take(100).replace('\n', ' ')}")
                }
            }
        }
        val report = out.toString()
        Log.i(TAG, report)
        return report
    }

    private const val TAG = "RetrievalProbe"
}
