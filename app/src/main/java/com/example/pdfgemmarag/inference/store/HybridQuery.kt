package com.example.pdfgemmarag.inference.store

import com.example.pdfgemmarag.inference.chunk.IdentifierAtoms

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
        // Extract compound IDs before the ordinary tokenizer splits punctuation. Leading-zero
        // specification numbers and dotted section paths are never subject to min token length.
        for (identifier in IdentifierAtoms.extract(question)) {
            seen += identifier
            if (seen.size >= maxTerms) return seen.toList()
        }
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
    fun build(
        terms: List<String>,
        similarityFloor: Double,
        vectorLimit: Int,
        requiredPropertyTerm: Pair<String, String>? = null,
    ): String {
        val semantic = "semanticSearch(getEmbeddingParameter(0), $similarityFloor, $vectorLimit)"
        val retrieval = if (terms.isEmpty()) semantic else {
            val keyword = terms.indices.joinToString(" OR ") { "getSearchStringParameter($it)" }
            "($keyword) OR $semantic"
        }
        val required = requiredPropertyTerm ?: return retrieval
        require(PROPERTY_NAME.matches(required.first)) { "Unsafe AppSearch property name" }
        require(PROPERTY_TERM.matches(required.second)) { "Unsafe AppSearch property term" }
        return "${required.first}:${required.second} AND ($retrieval)"
    }

    private val PROPERTY_NAME = Regex("[A-Za-z][A-Za-z0-9_.]*")
    private val PROPERTY_TERM = Regex("[A-Za-z0-9_-]+")
}
