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
    /** Explicit AppSearch namespace. Never infer this from [chunkId]. */
    val indexNamespace: String = docHash,
    val excerptId: String = "",
    val sectionId: String = "",
    val specificationNumber: String = "",
    val sectionNumber: String = "",
    val sectionTitle: String = "",
    val sectionPath: String = "",
    val contentKind: String = "PARAGRAPH",
    val continuesFromChunkIndex: Int = -1,
    val continuesToChunkIndex: Int = -1,
    val tableId: String = "",
    val tableNumber: String = "",
    val tableCaption: String = "",
) : Parcelable
