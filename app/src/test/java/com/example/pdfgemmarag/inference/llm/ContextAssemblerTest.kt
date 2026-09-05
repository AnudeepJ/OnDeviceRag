package com.example.pdfgemmarag.inference.llm

import com.example.pdfgemmarag.core.model.Citation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextAssemblerTest {
    private fun c(idx: Int, page: Int, text: String, score: Double = 1.0 - idx * 0.01) =
        Citation("doc:$idx", "doc", page, idx, score, text)

    @Test
    fun `budget drops lowest ranked chunks first`() {
        // 1250 chars -> ~351 tokens (+12 header) each; prefix differs so dedupe keeps every chunk
        val ranked = (0 until 40).map { c(it * 3, it, "chunk$it " + "word ".repeat(250)) }
        val assembled = ContextAssembler(contextTokenBudget = 1500, maxChunks = 20).assemble("q", ranked)
        assertEquals(4, assembled.citations.size)
        assertEquals(ranked.take(assembled.citations.size).map { it.chunkId }, assembled.citations.map { it.chunkId })
        assertTrue(assembled.approxTokens <= 1500 + 300)
    }

    @Test
    fun `adjacent overlapping chunks are deduped`() {
        val shared = "the contract identifies performance obligations and allocates the transaction price to each obligation"
        val a = c(10, 4, "Revenue recognition. $shared based on standalone selling prices.")
        val b = c(11, 4, "$shared based on standalone selling prices. Disclosures follow in note 5.")
        val far = c(30, 9, "Completely different content about employee headcount and office leases.")
        val out = ContextAssembler().dedupe(listOf(a, b, far))
        assertEquals(listOf("doc:10", "doc:30"), out.map { it.chunkId })
    }

    @Test
    fun `prompt numbers excerpts with pages and ends with the question`() {
        val assembled = ContextAssembler().assemble("What was revenue?", listOf(c(0, 12, "Revenue was 4.2M."), c(1, 40, "Margin was 31%.")))
        assertTrue(assembled.prompt.contains("[1] (Page 12)"))
        assertTrue(assembled.prompt.contains("[2] (Page 40)"))
        assertTrue(assembled.prompt.trimEnd().endsWith("Question: What was revenue?"))
    }

    @Test
    fun `keyword overlap outranks a high-score off-topic header`() {
        val junk = c(0, 1, "FORMWORK AND CONCRETE DIVISION 3 TECHNICAL SPECIFICATIONS HEADER ONLY REPEATED", 0.99)
        val hit = c(1, 3, "C. Plywood: Conform to PS 1, Class 1. D. Lumber: Conform to PS 20.", 0.20)
        val assembled = ContextAssembler().assemble("What plywood standard and class is required for formwork?", listOf(junk, hit))
        assertTrue(assembled.citations.first().text.contains("PS 1"))
    }

    @Test
    fun `token estimate scales by script`() {
        val latin = ContextAssembler.estimateTokens("a".repeat(400))
        val cjk = ContextAssembler.estimateTokens("東".repeat(400))
        assertTrue(latin in 100..130)
        assertTrue(cjk in 430..450)
    }
}
