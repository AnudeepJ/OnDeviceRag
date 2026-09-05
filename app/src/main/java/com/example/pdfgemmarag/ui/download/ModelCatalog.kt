package com.example.pdfgemmarag.ui.download

import com.example.pdfgemmarag.BuildConfig
import com.example.pdfgemmarag.core.model.ModelPaths

/**
 * Downloadable artifacts. The Gemma and EmbeddingGemma weights are gated on Hugging Face, so they
 * must be mirrored on a CDN you control; set `MODEL_CDN_BASE_URL` (and the SHA-256 of each file)
 * in `~/.gradle/gradle.properties` or CI secrets. Files with an empty SHA install with a warning.
 */
data class CatalogEntry(
    val id: String,
    val displayName: String,
    val fileName: String,
    val kind: Kind,
    val sizeBytes: Long,
    val sha256: String,
    /** Minimum device RAM (GB) to offer this entry; 0 = always. */
    val minRamGb: Int,
    val description: String,
) {
    enum class Kind { LLM, EMBEDDING, TOKENIZER }

    val url: String get() = BuildConfig.MODEL_CDN_BASE_URL.trimEnd('/') + "/" + fileName
    val sizeMb: Long get() = sizeBytes shr 20
}

object ModelCatalog {
    val entries: List<CatalogEntry> = listOf(
        CatalogEntry(
            id = "gemma4-e2b",
            displayName = "Gemma 4 E2B (default)",
            fileName = "gemma-4-E2B-it.litertlm",
            kind = CatalogEntry.Kind.LLM,
            sizeBytes = 2_580L * 1024 * 1024,
            sha256 = BuildConfig.SHA256_GEMMA_E2B,
            minRamGb = 6,
            description = "2.6 GB. Runs on 6 GB+ devices. Best balance of speed and quality for RAG.",
        ),
        CatalogEntry(
            id = "gemma4-e4b",
            displayName = "Gemma 4 E4B",
            fileName = "gemma-4-E4B-it.litertlm",
            kind = CatalogEntry.Kind.LLM,
            sizeBytes = 3_650L * 1024 * 1024,
            sha256 = BuildConfig.SHA256_GEMMA_E4B,
            minRamGb = 12,
            description = "3.7 GB. Higher quality; needs 12 GB+ RAM.",
        ),
        CatalogEntry(
            id = "embeddinggemma",
            displayName = "EmbeddingGemma 300M (required)",
            fileName = ModelPaths.EMBEDDING_MODEL_FILE,
            kind = CatalogEntry.Kind.EMBEDDING,
            sizeBytes = 179L * 1024 * 1024,
            sha256 = BuildConfig.SHA256_EMBEDDING,
            minRamGb = 0,
            description = "179 MB. seq512 LiteRT export; produces the 512-d vectors stored in AppSearch.",
        ),
        CatalogEntry(
            id = "embeddinggemma-tokenizer",
            displayName = "EmbeddingGemma tokenizer (required)",
            fileName = ModelPaths.EMBEDDING_TOKENIZER_FILE,
            kind = CatalogEntry.Kind.TOKENIZER,
            sizeBytes = 4_700L * 1024,
            sha256 = BuildConfig.SHA256_TOKENIZER,
            minRamGb = 0,
            description = "4.7 MB SentencePiece model shared by Gemma and EmbeddingGemma.",
        ),
    )

    fun byFileName(fileName: String): CatalogEntry? = entries.firstOrNull { it.fileName == fileName }
}
