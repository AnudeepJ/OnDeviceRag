package com.example.pdfgemmarag.core.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class EngineStatus(
    val state: State,
    /** "GPU", "CPU" or "" when no engine is loaded. */
    val backend: String = "",
    val modelPath: String = "",
    val modelName: String = "",
    val embedderLoaded: Boolean = false,
    val thermalStatus: Int = 0,
    val message: String = "",
) : Parcelable {
    enum class State { UNLOADED, LOADING, READY, FAILED }
}
