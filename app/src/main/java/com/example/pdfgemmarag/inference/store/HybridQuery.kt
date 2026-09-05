package com.example.pdfgemmarag.inference.store

/**
 * Builds the AppSearch list-filter query for hybrid retrieval.
 *
 * Passing the raw question to `getSearchStringParameter(0)` makes Icing AND every token. A
 * natural-language question like "What city published these Division 3 specifications?" then
 * requires *every* word (what, city, published, these, …) to appear in one chunk, so keyword
 * search returns nothing. Distinctive terms are OR'ed instead; semantic search covers paraphrase.
 */
object HybridQuery {

    val STOPWORDS = setOf(
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "how", "i", "if", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
        "so", "than", "that", "the", "their", "these", "this", "those", "to", "was", "we",
        "were", "what", "when", "where", "which", "who", "why", "will", "with", "you",
        "about", "any", "been", "being", "both", "did", "each", "had", "has", "have", "her",
        "him", "his", "into", "just", "may", "more", "most", "our", "out", "over", "own",
        "same", "some", "such", "them", "then", "there", "they", "too", "use", "used",
        "using", "very", "your",
    )

    private val tokenRe = Regex("""[\p{L}\p{N}]+""")

    fun keywordTerms(question: String, maxTerms: Int = 12): List<String> {
        val seen = LinkedHashSet<String>()
        for (m in tokenRe.findAll(question.lowercase())) {
            val t = m.value
            if (t in STOPWORDS) continue
            val cjk = t.any { it.code > 0x2E7F }
            if (!cjk && t.length < 3) continue
            seen += t
            if (seen.size >= maxTerms) break
        }
        return seen.toList()
    }

    /**
     * @param similarityFloor cosine below which vector hits are dropped
     * @param vectorLimit max hits the `semanticSearch` function itself may contribute
     */
    fun build(terms: List<String>, similarityFloor: Double, vectorLimit: Int): String {
        val semantic = "semanticSearch(getEmbeddingParameter(0), $similarityFloor, $vectorLimit)"
        if (terms.isEmpty()) return semantic
        val keyword = terms.indices.joinToString(" OR ") { "getSearchStringParameter($it)" }
        return "($keyword) OR $semantic"
    }
}
