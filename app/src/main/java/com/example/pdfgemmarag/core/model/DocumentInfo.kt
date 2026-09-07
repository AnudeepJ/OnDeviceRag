package com.example.pdfgemmarag.core.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/** Summary of an indexed document as known by the :inference process (AppSearch namespace). */
@Parcelize
data class DocumentInfo(
    val docHash: String,
    val displayName: String,
    val pageCount: Int,
    val chunkCount: Int,
    val script: String,
    val indexVersion: Int = 1,
    val activeIndexNamespace: String = docHash,
) : Parcelable
