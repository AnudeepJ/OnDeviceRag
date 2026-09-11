package com.example.pdfgemmarag.eval

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore

/**
 * Retrieval evaluation: page-hit@K over a fixed multilingual Q/A set.
 *
 * A question scores a hit@K when any of the top-K retrieved pages is in [EvalQuestion.expectedPages].
 * Tune [similarityFloor] and [keywordWeight] against [EvalReport.pageHitAt5] per language.
 */
object RetrievalEval {

    data class QuestionResult(
        val id: String,
        val lang: String,
        val question: String,
        val expectedPages: Set<Int>,
        val retrievedPages: List<Int>,
        val hitAt1: Boolean,
        val hitAt5: Boolean,
        val reciprocalRank: Double,
        val latencyMs: Long = 0,
    )

    data class LangSummary(val lang: String, val n: Int, val hitAt1: Double, val hitAt5: Double, val mrr: Double) {
        override fun toString() =
            "$lang n=$n hit@1=${pct(hitAt1)} hit@5=${pct(hitAt5)} mrr=${"%.3f".format(mrr)}"
    }

    data class EvalReport(
        val results: List<QuestionResult>,
        val byLanguage: List<LangSummary>,
        val overallHitAt1: Double,
        val overallHitAt5: Double,
        val overallMrr: Double,
        val elapsedMs: Long,
    ) {
        override fun toString(): String = buildString {
            appendLine("retrieval eval ${results.size} questions in ${elapsedMs}ms")
            appendLine("overall hit@1=${pct(overallHitAt1)} hit@5=${pct(overallHitAt5)} mrr=${"%.3f".format(overallMrr)}")
            byLanguage.forEach { appendLine("  $it") }
            results.filter { !it.hitAt5 }.forEach {
                appendLine("  MISS ${it.id} expected=${it.expectedPages} got=${it.retrievedPages}")
            }
        }
    }

    fun pageHitAtK(retrievedPages: List<Int>, expectedPages: Set<Int>, k: Int): Boolean {
        if (expectedPages.isEmpty()) return false
        return retrievedPages.take(k).any { it in expectedPages }
    }

    fun reciprocalRank(retrievedPages: List<Int>, expectedPages: Set<Int>): Double {
        val idx = retrievedPages.indexOfFirst { it in expectedPages }
        return if (idx < 0) 0.0 else 1.0 / (idx + 1)
    }

    fun score(question: RetrievalEvalDataset.Question, citations: List<Citation>, latencyMs: Long = 0): QuestionResult {
        val pages = citations.map { it.pageNumber }
        return QuestionResult(
            id = question.id,
            lang = question.lang,
            question = question.question,
            expectedPages = question.expectedPages,
            retrievedPages = pages,
            hitAt1 = pageHitAtK(pages, question.expectedPages, 1),
            hitAt5 = pageHitAtK(pages, question.expectedPages, 5),
            reciprocalRank = reciprocalRank(pages, question.expectedPages),
            latencyMs = latencyMs,
        )
    }

    fun summarise(results: List<QuestionResult>, elapsedMs: Long): EvalReport {
        fun List<QuestionResult>.hit1() = if (isEmpty()) 0.0 else count { it.hitAt1 }.toDouble() / size
        fun List<QuestionResult>.hit5() = if (isEmpty()) 0.0 else count { it.hitAt5 }.toDouble() / size
        fun List<QuestionResult>.mrr() = if (isEmpty()) 0.0 else sumOf { it.reciprocalRank } / size
        val byLang = results.groupBy { it.lang }.entries.sortedBy { it.key }.map { (lang, rows) ->
            LangSummary(lang, rows.size, rows.hit1(), rows.hit5(), rows.mrr())
        }
        return EvalReport(results, byLang, results.hit1(), results.hit5(), results.mrr(), elapsedMs)
    }

    /**
     * Runs the full dataset against [store]. [embed] must produce L2-normalised query vectors with
     * [com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder.MODEL_SIGNATURE].
     */
    suspend fun run(
        store: AppSearchVectorStore,
        docHash: String,
        embed: suspend (String) -> FloatArray,
        topK: Int = 5,
        similarityFloor: Double = 0.3,
        keywordWeight: Double = 0.05,
        questions: List<RetrievalEvalDataset.Question> = RetrievalEvalDataset.questions(),
        semanticWeight: Double = 1.0,
        localRerankWeight: Double = 0.35,
    ): EvalReport {
        val t0 = System.currentTimeMillis()
        val results = questions.map { q ->
            val queryStarted = System.nanoTime()
            val hits = store.search(
                docHash, q.question, embed(q.question), topK, similarityFloor, keywordWeight,
                semanticWeight = semanticWeight, localRerankWeight = localRerankWeight,
            )
            score(q, hits, (System.nanoTime() - queryStarted) / 1_000_000)
        }
        return summarise(results, System.currentTimeMillis() - t0)
    }

    private fun pct(v: Double) = "${"%.1f".format(v * 100)}%"
}
