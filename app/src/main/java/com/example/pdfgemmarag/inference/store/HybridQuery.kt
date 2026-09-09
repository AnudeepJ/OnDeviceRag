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
            // Numbers in a question ("between 21 and 25", "Class 4") are strong evidence
            // anchors even when short; ordinary words need three letters to be distinctive.
            val numeric = t.all(Char::isDigit)
            if (!cjk && !numeric && t.length < 3) continue
            if (numeric && t.length < 2) continue
            seen += t
            if (seen.size >= maxTerms) break
        }
        return seen.toList()
    }

    /**
     * Light, language-neutral stem for prefix matching: `trenches` -> `trench`, `required` ->
     * `requir`, `generally` -> `general`. Only applied to plain Latin words of five or more letters
     * so short words and identifiers keep exact matching. Returns null when no prefix form helps.
     */
    fun prefixTerm(term: String): String? {
        if (term.length < 5 || !term.all { it in 'a'..'z' }) return null
        var stem = term
        for (suffix in SUFFIXES) {
            if (stem.length - suffix.length >= 4 && stem.endsWith(suffix)) {
                stem = stem.removeSuffix(suffix)
                break
            }
        }
        return if (stem.length >= 4) "$stem*" else null
    }

    private const val ANCHOR_WEIGHT = 2.0

    /** Tokens written as acronyms (all capitals) or numbers in the original question. */
    fun anchorTerms(question: String): Set<String> = tokenRe.findAll(question)
        .map { it.value }
        .filter { token -> token.length >= 2 && (token.all(Char::isDigit) || (token.all { it.isUpperCase() || it.isDigit() } && token.any(Char::isLetter))) }
        .map { it.lowercase() }
        .toSet()

    private val SUFFIXES = listOf("ations", "ation", "ings", "ing", "ied", "ies", "ed", "es", "ly", "s")

    /**
     * Local re-rank after AppSearch: adds a bounded bonus for the query terms (by stem prefix)
     * present in the chunk's section path and text, weighted by how rare each term is among the
     * fetched candidates (an IDF estimate over the candidate set). Generic words shared by most
     * candidates contribute little; a distinctive word present in one or two candidates lifts them.
     * The semantic score still dominates.
     */
    fun <T> rerank(
        hits: List<T>,
        terms: List<String>,
        weight: Double,
        score: (T) -> Double,
        text: (T) -> String,
        anchors: Set<String> = emptySet(),
    ): List<Pair<T, Double>> {
        if (terms.isEmpty() || hits.isEmpty()) return hits.map { it to score(it) }
        val stems = terms.map { term -> prefixTerm(term)?.removeSuffix("*") ?: term }.distinct()
        val tokenSets = hits.map { hit -> tokenRe.findAll(text(hit).lowercase()).map { it.value }.toHashSet() }
        fun matches(tokens: Set<String>, stem: String) = tokens.any { it == stem || (stem.length >= 4 && it.startsWith(stem)) }
        val idf = stems.associateWith { stem ->
            val df = tokenSets.count { matches(it, stem) }
            val base = Math.log((hits.size + 1.0) / (df + 1.0)) + 0.1
            // Acronyms and numbers in a question (HIRA, 03300, 21) are its most specific tokens.
            if (stem in anchors) base * ANCHOR_WEIGHT else base
        }
        val total = idf.values.sum().coerceAtLeast(1e-6)
        return hits.mapIndexed { index, hit ->
            val covered = stems.filter { matches(tokenSets[index], it) }.sumOf { idf.getValue(it) }
            hit to (score(hit) + weight * covered / total)
        }.sortedByDescending { it.second }
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
        withPrefixes: Boolean = true,
    ): String {
        val semantic = "semanticSearch(getEmbeddingParameter(0), $similarityFloor, $vectorLimit)"
        val retrieval = if (terms.isEmpty()) semantic else {
            val exact = terms.indices.map { "getSearchStringParameter($it)" }
            // Prefix forms are inlined (tokens are alphanumeric, so no escaping is needed) and
            // widen keyword recall to morphological variants.
            val prefixes = if (withPrefixes) terms.mapNotNull(::prefixTerm).distinct() else emptyList()
            val keyword = (exact + prefixes).joinToString(" OR ")
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
