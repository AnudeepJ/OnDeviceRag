package com.example.pdfgemmarag.inference.pdf

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Log
import com.pdftron.pdf.Element
import com.pdftron.pdf.ElementReader
import com.pdftron.pdf.PDFDoc
import com.pdftron.pdf.PDFDraw
import com.pdftron.pdf.PDFNet
import com.pdftron.pdf.Page
import com.pdftron.pdf.TextExtractor
import java.io.Closeable

/**
 * Apryse (PDFTron) wrapper: per-page word geometry via [TextExtractor] and page rasterisation for
 * OCR. All calls happen in the :inference process; PDFNet is initialised once per process.
 */
class AprysePdfExtractor(context: Context) {

    init {
        initialize(context)
    }

    /** Open document handle; pages are extracted lazily so 500-page PDFs never sit in memory at once. */
    inner class OpenDocument(path: String) : Closeable {
        private val doc = PDFDoc(path).also { it.initSecurityHandler() }
        val pageCount: Int = doc.pageCount

        /** Extracts positioned lines for [pageNumber] (1-based). */
        fun extractPage(pageNumber: Int): PageLayout {
            val page = doc.getPage(pageNumber)
            val width = page.pageWidth.toFloat()
            val height = page.pageHeight.toFloat()
            val lines = ArrayList<LineBox>()
            TextExtractor().use { te ->
                te.begin(page, null, TextExtractor.e_remove_hidden_text or TextExtractor.e_no_watermarks)
                var line = te.firstLine
                while (line != null && line.isValid) {
                    val words = ArrayList<WordBox>()
                    var word = line.firstWord
                    while (word != null && word.isValid) {
                        val text = TextNormalizer.normalize(word.string)
                        if (text.isNotBlank()) {
                            val r = word.bBox
                            r.normalize()
                            // PDF user space is y-up; flip so "top" is smaller than "bottom" like a screen.
                            words += WordBox(
                                text.trim(),
                                Box(r.x1.toFloat(), (height - r.y2).toFloat(), r.x2.toFloat(), (height - r.y1).toFloat()),
                            )
                        }
                        word = word.nextWord
                    }
                    if (words.isNotEmpty()) lines += LineBox(words)
                    line = line.nextLine
                }
            }
            // TextExtractor orders by flow, which keeps multi-column text separate but turns table
            // columns into vertical "lines"; RowReflow rebuilds true rows in those regions only.
            return PageLayout(pageNumber, width, height, RowReflow().reflow(lines, width), pageHasImages(page), PageLayout.Source.TEXT)
        }

        /** Renders a page for OCR at a DPI that keeps the long edge <= [maxLongEdgePx]. */
        fun renderPage(pageNumber: Int, maxLongEdgePx: Int = 2048): Bitmap {
            val page = doc.getPage(pageNumber)
            val longEdgePts = maxOf(page.pageWidth, page.pageHeight)
            val dpi = (maxLongEdgePx / longEdgePts * 72.0).coerceIn(72.0, 300.0)
            val draw = PDFDraw()
            try {
                draw.setDPI(dpi)
                draw.setPageTransparent(false)
                return draw.getBitmap(page, Bitmap.Config.ARGB_8888)
            } finally {
                draw.destroy()
            }
        }

        private fun pageHasImages(page: Page): Boolean {
            val reader = ElementReader()
            try {
                reader.begin(page)
                var el = reader.next()
                var depth = 0
                while (el != null) {
                    when (el.type) {
                        Element.e_image, Element.e_inline_image ->
                            if (el.imageWidth >= 200 && el.imageHeight >= 200) return true
                        Element.e_form -> if (depth < 4) { reader.formBegin(); depth++ }
                    }
                    el = reader.next()
                    while (el == null && depth > 0) { reader.end(); depth--; el = reader.next() }
                }
                reader.end()
            } catch (t: Throwable) {
                Log.w(TAG, "image scan failed on page ${page.index}: ${t.message}")
            } finally {
                reader.destroy()
            }
            return false
        }

        override fun close() {
            try { doc.close() } catch (t: Throwable) { Log.w(TAG, "close failed", t) }
        }
    }

    fun open(path: String): OpenDocument = OpenDocument(path)

    companion object {
        private const val TAG = "AprysePdfExtractor"
        @Volatile private var initialized = false

        @Synchronized
        fun initialize(context: Context) {
            if (initialized) return
            val key = licenseKey(context)
            if (key.isBlank()) Log.w(TAG, "No pdftron_license_key in manifest; Apryse will run in demo mode")
            PDFNet.initialize(context, com.pdftron.pdfnet.R.raw.pdfnet, key)
            initialized = true
            Log.i(TAG, "PDFNet ${PDFNet.getVersionString()} initialised")
        }

        fun licenseKey(context: Context): String = try {
            val ai = context.packageManager.getApplicationInfo(context.packageName, PackageManager.GET_META_DATA)
            ai.metaData?.getString("pdftron_license_key") ?: ""
        } catch (e: Exception) { "" }

        fun version(): String = try { PDFNet.getVersionString() } catch (t: Throwable) { "uninitialised" }
    }
}
