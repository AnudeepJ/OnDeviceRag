package com.example.pdfgemmarag.spikes

import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.inference.ocr.ScriptDetector
import com.example.pdfgemmarag.inference.pdf.AprysePdfExtractor
import com.example.pdfgemmarag.inference.pdf.HeaderFooterStripper
import com.example.pdfgemmarag.inference.pdf.Segment
import com.example.pdfgemmarag.inference.pdf.TableClusterer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Spike 5: Apryse `TextExtractor` output on CJK, multi-column and tabular pages, plus the
 * layout heuristics built on it. Test PDFs are generated on device with `android.graphics.pdf`
 * (Skia backend, embedded system fonts, real text objects) so the spike needs no fixtures.
 */
@RunWith(AndroidJUnit4::class)
class Spike5ApryseExtractionTest {

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val W = 595
    private val H = 842

    private fun paint(size: Float, typeface: Typeface = Typeface.DEFAULT) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = size; this.typeface = typeface
    }

    private fun buildPdf(name: String, pages: Int, draw: (PdfDocument.Page, Int) -> Unit): File {
        val doc = PdfDocument()
        for (p in 1..pages) {
            val page = doc.startPage(PdfDocument.PageInfo.Builder(W, H, p).create())
            draw(page, p)
            doc.finishPage(page)
        }
        val file = File(ctx.cacheDir, "$name.pdf")
        file.outputStream().use { doc.writeTo(it) }
        doc.close()
        return file
    }

    private val extractor by lazy { AprysePdfExtractor(ctx) }

    @Test
    fun cjkTextIsExtractedVerbatim() {
        val ja = "東京都は日本の首都です。人口は約千四百万人。"
        val zh = "北京是中华人民共和国的首都。"
        val ko = "서울은 대한민국의 수도입니다."
        val pdf = buildPdf("cjk", 1) { page, _ ->
            val c = page.canvas
            val p = paint(16f)
            c.drawText(ja, 50f, 100f, p)
            c.drawText(zh, 50f, 140f, p)
            c.drawText(ko, 50f, 180f, p)
            c.drawText("Mixed: Tokyo 東京 2023", 50f, 220f, p)
        }
        extractor.open(pdf.absolutePath).use { doc ->
            val t0 = SystemClock.elapsedRealtime()
            val layout = doc.extractPage(1)
            val ms = SystemClock.elapsedRealtime() - t0
            val text = layout.plainText
            Log.i("SPIKE5", "cjk page (${ms}ms, ${layout.lines.size} lines):\n$text")
            // TextExtractor may split CJK runs into several "words"; compare with whitespace removed.
            val squashed = text.replace(Regex("\\s+"), "")
            assertTrue(text, squashed.contains(ja))
            assertTrue(text, squashed.contains(zh))
            assertTrue(text, squashed.contains(ko.replace(" ", "")))
            assertTrue(text, squashed.contains("Tokyo東京2023"))
            assertTrue(ScriptDetector.detect(ja + zh).isCjk)
            assertTrue("script stats should see CJK", ScriptDetector.stats(text).cjk > 20)
            assertTrue("geometry must be populated", layout.lines.all { l -> l.words.all { it.box.width > 0 && it.box.height > 0 } })
        }
    }

    @Test
    fun multiColumnKeepsColumnsSeparate() {
        val left = (1..12).map { "Left column sentence number $it about apples." }
        val right = (1..12).map { "Right column sentence number $it about oranges." }
        val pdf = buildPdf("columns", 1) { page, _ ->
            val c = page.canvas
            val p = paint(9f)
            left.forEachIndexed { i, s -> c.drawText(s, 40f, 100f + i * 14f, p) }
            right.forEachIndexed { i, s -> c.drawText(s, 310f, 100f + i * 14f, p) }
        }
        extractor.open(pdf.absolutePath).use { doc ->
            val layout = doc.extractPage(1)
            val lines = layout.lines.map { it.text }
            Log.i("SPIKE5", "columns:\n${lines.joinToString("\n")}")
            // No extracted line may merge the two columns.
            assertTrue(lines.joinToString("\n"), lines.none { it.contains("apples") && it.contains("oranges") })
            assertEquals(24, lines.size)
            // Reading order: TextExtractor emits column 1 fully before column 2 (flow order), not row-interleaved.
            val firstRight = lines.indexOfFirst { it.contains("oranges") }
            val lastLeft = lines.indexOfLast { it.contains("apples") }
            val flowOrder = lastLeft < firstRight
            Log.i("SPIKE5", "columns flow-ordered=$flowOrder (lastLeft=$lastLeft firstRight=$firstRight)")
            assertTrue("columns interleaved; chunker would mix columns", flowOrder)
        }
    }

    @Test
    fun tableRowsAreClusteredWithHeader() {
        val header = listOf("Year", "Revenue", "Profit", "Margin")
        val rows = listOf(
            listOf("2021", "3.1M", "0.4M", "12.9%"),
            listOf("2022", "3.7M", "0.6M", "16.2%"),
            listOf("2023", "4.2M", "0.9M", "21.4%"),
        )
        val xs = listOf(60f, 180f, 300f, 420f)
        val pdf = buildPdf("table", 1) { page, _ ->
            val c = page.canvas
            val p = paint(11f)
            c.drawText("Financial summary for the fiscal years shown below.", 60f, 80f, p)
            header.forEachIndexed { i, h -> c.drawText(h, xs[i], 120f, paint(11f, Typeface.DEFAULT_BOLD)) }
            rows.forEachIndexed { r, row -> row.forEachIndexed { i, cell -> c.drawText(cell, xs[i], 140f + r * 16f, p) } }
            c.drawText("Figures are unaudited.", 60f, 220f, p)
        }
        extractor.open(pdf.absolutePath).use { doc ->
            val layout = doc.extractPage(1)
            val content = TableClusterer().analyse(layout)
            val geometry = layout.lines.joinToString("\n") { l -> l.words.joinToString(" ") { w -> "${w.text}@(${w.box.left.toInt()},${w.box.top.toInt()},${w.box.right.toInt()},${w.box.bottom.toInt()})" } }
            Log.i("SPIKE5", "table page geometry:\n$geometry\nsegments:\n${content.segments.joinToString("\n")}")
            val table = content.segments.filterIsInstance<Segment.Table>().singleOrNull()
            assertTrue("expected exactly one table, got ${content.segments}\ngeometry:\n$geometry", table != null)
            assertTrue(table!!.header, table.header.contains("Year") && table.header.contains("Margin"))
            assertEquals(3, table.rows.size)
            assertTrue(table.rows[2], table.rows[2].contains("2023") && table.rows[2].contains("21.4%"))
            val paragraphs = content.segments.filterIsInstance<Segment.Paragraph>()
            assertEquals(2, paragraphs.size)
        }
    }

    @Test
    fun repeatedHeadersAreStrippedAcrossPages() {
        val pdf = buildPdf("hf", 6) { page, p ->
            val c = page.canvas
            c.drawText("ACME Corp Annual Report 2023", 60f, 40f, paint(9f))
            c.drawText("Body paragraph on page $p discussing topic ${"ABCDEF"[p - 1]} in detail.", 60f, 300f, paint(11f))
            c.drawText("Page $p of 6", 260f, 810f, paint(9f))
        }
        extractor.open(pdf.absolutePath).use { doc ->
            val pages = (1..doc.pageCount).map { doc.extractPage(it) }
            val stripped = HeaderFooterStripper().strip(pages)
            val texts = stripped.map { it.plainText }
            Log.i("SPIKE5", "after strip:\n${texts.joinToString("\n---\n")}")
            assertTrue(texts.joinToString(), texts.none { it.contains("ACME") || it.contains("Page ") })
            assertTrue(texts.joinToString(), texts.all { it.contains("Body paragraph") })
        }
    }

    @Test
    fun extractionThroughputOn100Pages() {
        val pdf = buildPdf("long", 100) { page, p ->
            val c = page.canvas
            val paint = paint(10f)
            for (i in 0 until 45) c.drawText("Page $p line $i: The quick brown fox jumps over the lazy dog near the riverbank.", 50f, 60f + i * 16f, paint)
        }
        extractor.open(pdf.absolutePath).use { doc ->
            assertEquals(100, doc.pageCount)
            val t0 = SystemClock.elapsedRealtime()
            var chars = 0
            for (p in 1..doc.pageCount) chars += doc.extractPage(p).charCount
            val ms = SystemClock.elapsedRealtime() - t0
            Log.i("SPIKE5", "100 pages extracted in ${ms}ms (${ms / 100.0}ms/page), $chars chars; PDFNet ${AprysePdfExtractor.version()}")
            assertTrue(chars > 100 * 45 * 60)
            assertTrue("too slow: ${ms}ms for 100 pages", ms < 60_000)
        }
    }

    @Test
    fun imageOnlyPageIsFlaggedForOcr() {
        val bmp = android.graphics.Bitmap.createBitmap(400, 400, android.graphics.Bitmap.Config.ARGB_8888).apply { eraseColor(0xFF888888.toInt()) }
        val pdf = buildPdf("scan", 1) { page, _ -> page.canvas.drawBitmap(bmp, 50f, 50f, null) }
        extractor.open(pdf.absolutePath).use { doc ->
            val layout = doc.extractPage(1)
            assertEquals(0, layout.charCount)
            assertTrue("image XObject not detected", layout.hasImages)
            val rendered = doc.renderPage(1, maxLongEdgePx = 1024)
            assertTrue(rendered.width in 700..1024 && rendered.height in 700..1024)
            rendered.recycle()
        }
    }
}
