package com.example.pdfgemmarag.inference.llm

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.inference.ocr.ScriptDetector

/**
 * Builds the grounded prompt from ranked chunks under a hard token budget.
 *
 * The Kotlin LiteRT-LM SDK does not expose the tokenizer, so token counts are estimated per script:
 * CJK ~1.1 tokens/char, everything else ~0.28 tokens/char (4 chars/token) with a safety margin.
 */
class ContextAssembler(
    private val contextTokenBudget: Int = 1600,
    private val maxChunks: Int = 4,
) {

    data class Assembled(val prompt: String, val citations: List<Citation>, val approxTokens: Int)

    fun assemble(question: String, ranked: List<Citation>): Assembled {
        val deduped = dedupe(ranked)
        val selected = ArrayList<Citation>()
        var used = 0
        for (c in deduped) {
            if (selected.size >= maxChunks) break
            val cost = estimateTokens(c.text) + 12 // header overhead per chunk
            if (used + cost > contextTokenBudget) continue // drop lowest-ranked-first: list is already ranked
            selected += c
            used += cost
        }
        val sb = StringBuilder()
        sb.append("Answer the question using only the document excerpts below. ")
        sb.append("Cite the page of every fact you use in the form [Page N]. ")
        sb.append("If the excerpts do not contain the answer, say that the document does not cover it. ")
        sb.append("Reply in the language of the question.\n\n")
        sb.append("Excerpts:\n")
        selected.forEachIndexed { i, c ->
            sb.append("[").append(i + 1).append("] (Page ").append(c.pageNumber).append(")\n")
            sb.append(c.text.trim()).append("\n\n")
        }
        sb.append("Question: ").append(question.trim())
        return Assembled(sb.toString(), selected, used + estimateTokens(question) + 80)
    }

    /**
     * Adjacent chunks from the same page overlap by ~12%; when both are retrieved the lower-ranked one
     * mostly repeats the other. Keep the first (higher-ranked) of any pair of neighbours.
     */
    internal fun dedupe(ranked: List<Citation>): List<Citation> {
        val kept = ArrayList<Citation>()
        val seenText = HashSet<String>()
        for (c in ranked) {
            val key = c.text.trim().take(200)
            if (!seenText.add(key)) continue
            val neighbour = kept.any { k ->
                k.docHash == c.docHash && k.pageNumber == c.pageNumber && kotlin.math.abs(k.chunkIndex - c.chunkIndex) == 1 &&
                    overlapRatio(k.text, c.text) > 0.35
            }
            if (!neighbour) kept += c
        }
        return kept
    }

    private fun overlapRatio(a: String, b: String): Double {
        val ta = a.split(Regex("\\s+")).filter { it.length > 2 }.toSet()
        val tb = b.split(Regex("\\s+")).filter { it.length > 2 }.toSet()
        if (ta.isEmpty() || tb.isEmpty()) return 0.0
        return ta.intersect(tb).size.toDouble() / minOf(ta.size, tb.size)
    }

    companion object {
        fun estimateTokens(text: String): Int {
            if (text.isEmpty()) return 0
            val stats = ScriptDetector.stats(text)
            val cjkChars = stats.cjk
            val otherChars = text.length - cjkChars
            return (cjkChars * 1.1 + otherChars * 0.28).toInt() + 1
        }
    }
}
