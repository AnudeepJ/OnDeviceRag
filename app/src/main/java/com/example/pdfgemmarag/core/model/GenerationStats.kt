package com.example.pdfgemmarag.core.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class GenerationStats(
    val generationId: Long,
    val timeToFirstTokenMs: Long,
    val totalMs: Long,
    val outputChars: Int,
    val approxTokensPerSecond: Double,
    val retrievedChunks: Int,
    val contextTokensApprox: Int,
    val backend: String,
    val cancelled: Boolean,
    val visibleTimeToFirstTokenMs: Long = timeToFirstTokenMs,
    val groundingFailure: Boolean = false,
    /** Planner-owned scope for safe follow-up inheritance; never inferred from displayed chips. */
    val sourceSectionId: String = "",
    /** True when a FACT query recovered without its corrupt structure manifest. */
    val manifestFallback: Boolean = false,
) : Parcelable
