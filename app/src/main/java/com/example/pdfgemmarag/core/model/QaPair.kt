package com.example.pdfgemmarag.core.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/** One prior question/answer turn, with retrieved chunks already stripped from the answer. */
@Parcelize
data class QaPair(val question: String, val answer: String) : Parcelable
