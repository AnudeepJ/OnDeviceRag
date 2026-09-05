package com.example.pdfgemmarag.core.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/**
 * A retrieved chunk that supported an answer. Crosses the AIDL boundary; text is a single chunk
 * (~1-2 KB), far below Binder's 1 MB transaction limit.
 */
@Parcelize
data class Citation(
    val chunkId: String,
    val docHash: String,
    val pageNumber: Int,
    val chunkIndex: Int,
    val score: Double,
    /** Chunk text. Empty when only the reference is being transferred. */
    val text: String = "",
) : Parcelable
