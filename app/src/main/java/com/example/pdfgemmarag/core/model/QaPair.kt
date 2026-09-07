package com.example.pdfgemmarag.core.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/** One prior turn. Source section metadata is trusted; the prior model answer is never RAG evidence. */
@Parcelize
data class QaPair(
    val question: String,
    val answer: String,
    val sourceSectionId: String = "",
) : Parcelable
