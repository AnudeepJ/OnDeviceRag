package com.example.pdfgemmarag.ui

import android.app.ActivityManager
import android.content.Context
import android.os.StatFs
import com.example.pdfgemmarag.BuildConfig
import java.io.File

/**
 * Device capability gating from the plan: < 6 GB RAM refuse, 6-8 GB E2B only and no chat while
 * indexing, >= 12 GB also offers E4B. Storage pre-flight: 2x model size for download + copy, 1 GB
 * before indexing.
 */
class DeviceGate(private val context: Context) {

    enum class Tier { UNSUPPORTED, LOW, MID, HIGH }

    val totalRamBytes: Long by lazy {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.totalMem
    }

    val totalRamGb: Double get() = totalRamBytes / (1024.0 * 1024.0 * 1024.0)

    val tier: Tier
        get() = when {
            totalRamGb < 5.5 -> Tier.UNSUPPORTED
            totalRamGb < 8.5 -> Tier.LOW
            totalRamGb < 11.5 -> Tier.MID
            else -> Tier.HIGH
        }

    /** On 6-8 GB devices the LLM and the indexing pipeline must not be resident at the same time. */
    val chatBlockedWhileIndexing: Boolean get() = tier == Tier.LOW && !BuildConfig.DEBUG

    /** Release builds refuse < 6 GB. Debug (emulator) still lets you load a small test LLM. */
    val canLoadLlm: Boolean get() = tier != Tier.UNSUPPORTED || BuildConfig.DEBUG

    fun freeBytes(dir: File = context.filesDir): Long {
        val s = StatFs(dir.absolutePath)
        return s.availableBlocksLong * s.blockSizeLong
    }

    fun hasSpaceForModel(sizeBytes: Long): Boolean = freeBytes() >= sizeBytes * 2 + SAFETY_MARGIN

    fun hasSpaceForIndexing(): Boolean = freeBytes() >= INDEXING_MIN_FREE

    fun describe(): String = "RAM ${"%.1f".format(totalRamGb)} GB (${tier.name.lowercase()}), free ${freeBytes() shr 20} MB"

    companion object {
        const val SAFETY_MARGIN = 512L * 1024 * 1024
        const val INDEXING_MIN_FREE = 1024L * 1024 * 1024
    }
}
