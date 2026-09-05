package com.example.pdfgemmarag.core.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class IndexingProgress(
    val docHash: String,
    val stage: Stage,
    val current: Int,
    val total: Int,
    val detail: String = "",
    /** Set while thermal throttling has paused the pipeline. */
    val paused: Boolean = false,
) : Parcelable {
    enum class Stage { PREPARING, EXTRACTING, OCR, CHUNKING, EMBEDDING, INDEXING, FINALIZING }

    val fraction: Float
        get() = if (total <= 0) 0f else (current.toFloat() / total).coerceIn(0f, 1f)
}
