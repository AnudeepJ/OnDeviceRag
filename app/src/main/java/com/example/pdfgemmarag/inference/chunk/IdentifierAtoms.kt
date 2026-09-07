package com.example.pdfgemmarag.inference.chunk

import java.text.Normalizer
import java.util.Locale

/** Extracts searchable identifiers before ordinary punctuation tokenization can split them. */
object IdentifierAtoms {
    private val atom = Regex(
        "(?<![\\p{L}\\p{N}])(?:[A-Z]{2,12}\\s+[A-Z]?\\d[\\d.-]*|\\d+(?:[.-]\\d+)+|0\\d{2,})(?![\\p{L}\\p{N}])",
    )

    fun extract(text: String): List<String> {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
        return atom.findAll(normalized)
            .map { it.value.trim().lowercase(Locale.ROOT).replace(Regex("\\s+"), " ") }
            .distinct()
            .toList()
    }
}
