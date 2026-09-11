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
    /** Semicolon-separated grounding rejections (`VALUE_ABSENT:9m`, `UNIT_MISMATCH:40 psi`) for stage attribution. */
    val groundingReasons: String = "",
    /** Runtime-reported prefill throughput for the last turn; 0 when unavailable or deterministic. */
    val prefillTokensPerSecond: Double = 0.0,
    /** Runtime-reported decode throughput for the last turn; 0 when unavailable or deterministic. */
    val decodeTokensPerSecond: Double = 0.0,
    /** Prompt tokens as counted by the runtime for the last turn. */
    val prefillTokens: Int = 0,
    /** Planner intent that produced this answer (FACT, SECTION_SUMMARY, DOCUMENT_OVERVIEW, AMBIGUOUS_SECTION). */
    val intent: String = "",
    /** Deterministic lead that answered without the model, or empty when the model generated the answer. */
    val answeredBy: String = "",
    /** Expanded candidates before prompt budgeting. */
    val candidateChunks: Int = 0,
    val candidateChunkIds: String = "",
    /** `chunkId:origin:sourceChunkId` entries for retrieved and structurally expanded evidence. */
    val candidateProvenance: String = "",
    /** Stable comma-separated snapshots for stage-aware evaluation. */
    val selectedChunkIds: String = "",
    val selectedPages: String = "",
    val answerCitationChunkIds: String = "",
    /** Whether the selected evidence set was known complete for the requested scope. */
    val evidenceComplete: Boolean = false,
    /** Raw AppSearch/direct-scope result before structural expansion. */
    val rawCandidateChunks: Int = 0,
    val rawCandidateChunkIds: String = "",
    val rawCandidatePages: String = "",
    /** COMPLETE, PARTIAL, or UNKNOWN for the requested source scope. */
    val evidenceCompleteness: String = "UNKNOWN",
    /** Whether every expanded candidate fit the selected model context. */
    val selectionComplete: Boolean = false,
) : Parcelable
