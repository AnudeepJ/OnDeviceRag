package com.example.pdfgemmarag.inference.pdf

import java.security.MessageDigest
import java.util.Locale

/**
 * Document-neutral table identity. A caption number is optional; uncaptioned grids still receive a
 * stable id from the document hash, first page and ordinal so the manifest can name them.
 */
object TableIdentity {
    /** `Table 5.1`, `TABLE (1.1)`, `Table (1.4):`. The closing paren is optional. */
    private val NUMBER = Regex("(?i)^table\\s*\\(?\\s*([0-9][0-9a-z.\\-]*)\\s*\\)?")
    private const val MAX_CAPTION_CHARS = 120

    fun looksLikeCaption(text: String): Boolean = NUMBER.containsMatchIn(text.trim())
    private val FILLER = setOf("table", "the", "and", "of", "for", "to", "a", "an", "in")

    /**
     * Caption text to store on the grid. A header row sometimes shares the line with
     * `Table (1.1)`; keep the number and a short title, drop the following column labels.
     */
    fun captionFrom(text: String): String? {
        val trimmed = text.trim().replace(Regex("[\\s\\u00a0]+"), " ")
        if (!looksLikeCaption(trimmed)) return null
        if (trimmed.length <= MAX_CAPTION_CHARS) return trimmed
        val match = NUMBER.find(trimmed) ?: return null
        val rest = trimmed.substring(match.range.last + 1).trimStart(':', '-', '–', '—', ' ').trim()
        val title = rest.split(' ').filter { it.isNotBlank() }.takeWhile { word ->
            word.length > 2 && !word.matches(Regex("\\(?[0-9]+\\)?"))
        }.take(8).joinToString(" ")
        return (match.value + if (title.isBlank()) "" else " $title").trim().take(MAX_CAPTION_CHARS)
    }

    fun numberFromCaption(caption: String): String =
        NUMBER.find(caption.trim())?.groupValues?.get(1)?.uppercase(Locale.ROOT).orEmpty()

    fun id(docHash: String, caption: String, number: String, firstPage: Int, ordinal: Int): String {
        val key = "$docHash|${number.ifBlank { caption.trim().lowercase(Locale.ROOT) }}|$firstPage|$ordinal"
        return MessageDigest.getInstance("SHA-256")
            .digest(key.toByteArray())
            .take(12)
            .joinToString("") { "%02x".format(it) }
    }

    fun aliases(caption: String, number: String): List<String> {
        val words = caption.lowercase(Locale.ROOT)
            .replace(Regex("[^a-z0-9 ]"), " ")
            .split(' ')
            .filter { it.length >= 3 && it !in FILLER }
        return (words + listOfNotNull(number.lowercase(Locale.ROOT).takeIf { it.isNotBlank() })).distinct()
    }

    fun headersFromMarkdown(body: String): List<String> {
        val line = body.lineSequence().firstOrNull { it.trim().startsWith("|") } ?: return emptyList()
        return line.trim().trim('|').split('|').map(String::trim)
            .filter { it.isNotEmpty() && it.any { ch -> ch != '-' && ch != ':' } }
    }
}
