package com.example.pdfgemmarag.inference.pdf

import java.security.MessageDigest
import java.util.Locale

/**
 * Document-neutral table identity. A caption number is optional; uncaptioned grids still receive a
 * stable id from the document hash, first page and ordinal so the manifest can name them.
 */
object TableIdentity {
    private val NUMBER = Regex("(?i)^table\\s*\\(?\\s*([0-9][0-9a-z.\\-]*)")

    fun looksLikeCaption(text: String): Boolean = NUMBER.containsMatchIn(text.trim())
    private val FILLER = setOf("table", "the", "and", "of", "for", "to", "a", "an", "in")

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
