package com.example.pdfgemmarag.inference.ocr

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.example.pdfgemmarag.inference.pdf.Box
import com.example.pdfgemmarag.inference.pdf.LineBox
import com.example.pdfgemmarag.inference.pdf.PageLayout
import com.example.pdfgemmarag.inference.pdf.WordBox
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.google.mlkit.common.MlKit
import kotlinx.coroutines.tasks.await
import java.io.Closeable

/**
 * ML Kit on-device text recognition with per-document script routing.
 *
 * ML Kit ships one recogniser per script family; the Latin model returns garbage on CJK pages and
 * vice versa. The recogniser is chosen once per document: from already extracted text when the PDF
 * has any, otherwise by probing the first image-only page with the Japanese model (which reads
 * kana, kanji and Latin) and falling back to Korean when it finds nothing usable.
 */
class MlKitOcr(context: Context) : Closeable {
    private val clients = HashMap<Script, TextRecognizer>()

    init {
        // ML Kit's manifest provider initialises only the process in which it is created. OCR is
        // intentionally owned by :inference, so initialise it explicitly here rather than rely on
        // the UI process having started first. This is safe to call once in every app process.
        try {
            MlKit.initialize(context.applicationContext)
        } catch (alreadyInitialized: IllegalStateException) {
            // A host app (or ML Kit's manifest provider in the main process) may have initialized
            // the singleton before this library component is constructed. The explicit call is
            // still required in :inference, where that provider is not guaranteed to run.
            if (!alreadyInitialized.message.orEmpty().contains("already initialized", ignoreCase = true)) {
                throw alreadyInitialized
            }
            Log.d(TAG, "ML Kit was already initialized in this process")
        }
    }

    private fun client(script: Script): TextRecognizer = clients.getOrPut(script) {
        when (script) {
            Script.LATIN -> TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            Script.CHINESE -> TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())
            Script.JAPANESE -> TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build())
            Script.KOREAN -> TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
        }
    }

    /** Decides the recogniser for a document with no extractable text by probing one rendered page. */
    suspend fun probeScript(bitmap: Bitmap): Script {
        val jp = recognizeRaw(bitmap, Script.JAPANESE)
        val jpStats = ScriptDetector.stats(jp)
        if (jpStats.letters >= 20) {
            return when {
                jpStats.kana > 0 -> Script.JAPANESE
                jpStats.han > jpStats.latin -> Script.CHINESE
                else -> Script.LATIN
            }
        }
        val ko = recognizeRaw(bitmap, Script.KOREAN)
        val koStats = ScriptDetector.stats(ko)
        if (koStats.hangul >= 10) return Script.KOREAN
        return Script.LATIN
    }

    private suspend fun recognizeRaw(bitmap: Bitmap, script: Script): String =
        client(script).process(InputImage.fromBitmap(bitmap, 0)).await().text

    /**
     * Recognises [bitmap] and returns lines with word boxes in bitmap pixel space (same relative
     * geometry the table clusterer expects). The caller owns and recycles the bitmap.
     */
    suspend fun recognize(bitmap: Bitmap, script: Script, pageNumber: Int, pageWidthPts: Float, pageHeightPts: Float): PageLayout {
        val result = client(script).process(InputImage.fromBitmap(bitmap, 0)).await()
        val sx = pageWidthPts / bitmap.width
        val sy = pageHeightPts / bitmap.height
        val lines = ArrayList<LineBox>()
        for (block in result.textBlocks) {
            for (line in block.lines) {
                val words = line.elements.mapNotNull { el ->
                    val r = el.boundingBox ?: return@mapNotNull null
                    val t = el.text.trim()
                    if (t.isEmpty()) null else WordBox(t, Box(r.left * sx, r.top * sy, r.right * sx, r.bottom * sy))
                }
                if (words.isNotEmpty()) lines += LineBox(words)
            }
        }
        lines.sortWith(compareBy({ it.box.centerY }, { it.box.left }))
        Log.d(TAG, "OCR page $pageNumber ($script): ${lines.size} lines")
        return PageLayout(pageNumber, pageWidthPts, pageHeightPts, lines, hasImages = true, source = PageLayout.Source.OCR)
    }

    override fun close() {
        clients.values.forEach { runCatching { it.close() } }
        clients.clear()
    }

    companion object {
        private const val TAG = "MlKitOcr"
        /** Pages with fewer extractable characters than this and at least one large image go to OCR. */
        const val MIN_TEXT_CHARS = 50
    }
}
