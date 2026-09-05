package com.example.pdfgemmarag.inference.index

import android.os.SystemClock
import android.util.Log
import androidx.appsearch.app.EmbeddingVector
import com.example.pdfgemmarag.core.model.DocumentInfo
import com.example.pdfgemmarag.core.model.IndexingProgress
import com.example.pdfgemmarag.core.model.IndexingProgress.Stage
import com.example.pdfgemmarag.inference.chunk.ScriptAwareChunker
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import com.example.pdfgemmarag.inference.ocr.MlKitOcr
import com.example.pdfgemmarag.inference.ocr.Script
import com.example.pdfgemmarag.inference.ocr.ScriptDetector
import com.example.pdfgemmarag.inference.pdf.AprysePdfExtractor
import com.example.pdfgemmarag.inference.pdf.HeaderFooterStripper
import com.example.pdfgemmarag.inference.pdf.PageContent
import com.example.pdfgemmarag.inference.pdf.PageLayout
import com.example.pdfgemmarag.inference.pdf.TableClusterer
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.PdfChunkDocument
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.currentCoroutineContext

/**
 * PDF -> pages -> (OCR) -> layout -> chunks -> embeddings -> AppSearch.
 *
 * Runs inside the :inference foreground service. Cancellation is cooperative through the coroutine
 * context; on cancel or failure any chunks already written for this document are removed so the
 * store never holds a partial index.
 */
class IndexPdfUseCase(
    private val pdf: AprysePdfExtractor,
    private val ocr: MlKitOcr,
    private val embedder: EmbeddingGemmaEmbedder,
    private val store: AppSearchVectorStore,
    private val chunker: ScriptAwareChunker = ScriptAwareChunker(),
    private val stripper: HeaderFooterStripper = HeaderFooterStripper(),
    private val tables: TableClusterer = TableClusterer(),
    /** Suspends while the device is thermally throttled. */
    private val awaitCool: suspend () -> Unit = {},
) {

    suspend fun run(
        docHash: String,
        pdfPath: String,
        displayName: String,
        onProgress: (IndexingProgress) -> Unit,
    ): DocumentInfo {
        var wroteAny = false
        try {
            onProgress(IndexingProgress(docHash, Stage.PREPARING, 0, 1))
            // Re-indexing the same content hash replaces the previous index.
            store.removeDocument(docHash)

            val layouts = ArrayList<PageLayout>()
            val ocrPages = ArrayList<Int>()
            pdf.open(pdfPath).use { doc ->
                val pageCount = doc.pageCount
                // 1. Text extraction
                for (p in 1..pageCount) {
                    currentCoroutineContext().ensureActive(); awaitCool()
                    val layout = doc.extractPage(p)
                    if (layout.charCount < MlKitOcr.MIN_TEXT_CHARS && layout.hasImages) ocrPages += p
                    layouts += layout
                    if (p % 2 == 0 || p == pageCount) onProgress(IndexingProgress(docHash, Stage.EXTRACTING, p, pageCount))
                }
                // 2. OCR for image-only pages; recogniser chosen once per document
                if (ocrPages.isNotEmpty()) {
                    val textSoFar = layouts.filter { it.charCount >= MlKitOcr.MIN_TEXT_CHARS }.joinToString("\n") { it.plainText }.take(20_000)
                    val script: Script = if (textSoFar.length >= 200) {
                        ScriptDetector.detect(textSoFar)
                    } else {
                        val probe = doc.renderPage(ocrPages.first())
                        try { ocr.probeScript(probe) } finally { probe.recycle() }
                    }
                    Log.i(TAG, "OCR ${ocrPages.size}/${pageCount} pages with $script recogniser")
                    ocrPages.forEachIndexed { i, p ->
                        currentCoroutineContext().ensureActive(); awaitCool()
                        val bitmap = doc.renderPage(p)
                        try {
                            val original = layouts[p - 1]
                            layouts[p - 1] = ocr.recognize(bitmap, script, p, original.width, original.height)
                        } finally {
                            bitmap.recycle()
                        }
                        onProgress(IndexingProgress(docHash, Stage.OCR, i + 1, ocrPages.size, detail = script.name))
                    }
                }
            }

            // 3. Layout analysis
            onProgress(IndexingProgress(docHash, Stage.CHUNKING, 0, 1))
            val stripped = stripper.strip(layouts)
            val contents: List<PageContent> = stripped.map { tables.analyse(it) }
            val chunks = chunker.chunk(contents)
            val allText = contents.joinToString("\n") { pc ->
                pc.segments.joinToString("\n") {
                    when (it) {
                        is com.example.pdfgemmarag.inference.pdf.Segment.Paragraph -> it.text
                        is com.example.pdfgemmarag.inference.pdf.Segment.Table -> it.header
                    }
                }
            }.take(50_000)
            val docScript = ScriptDetector.detect(allText)
            if (chunks.isEmpty()) throw IllegalStateException("No text could be extracted from this PDF (even with OCR)")
            Log.i(TAG, "$displayName: ${layouts.size} pages -> ${chunks.size} chunks (${docScript})")

            // 4. Embed + index in batches
            val pageCount = layouts.size
            val batch = ArrayList<PdfChunkDocument>(BATCH)
            var indexed = 0
            val now = System.currentTimeMillis()
            val embedStarted = SystemClock.elapsedRealtime()
            Log.i(TAG, "embedding ${chunks.size} chunks on ${embedder.backend} seq=${embedder.sequenceLength}")
            for (chunk in chunks) {
                currentCoroutineContext().ensureActive(); awaitCool()
                val t0 = SystemClock.elapsedRealtime()
                val vec = embedder.embedDocument(chunk.text)
                val embedMs = SystemClock.elapsedRealtime() - t0
                batch += PdfChunkDocument().apply {
                    namespace = docHash
                    id = "$docHash:${chunk.chunkIndex}"
                    creationTimestampMillis = now
                    text = chunk.text
                    pageNumber = chunk.pageNumber
                    chunkIndex = chunk.chunkIndex
                    isTable = chunk.isTable
                    docName = displayName
                    script = docScript.name
                    this.pageCount = pageCount
                    embedding = EmbeddingVector(vec, EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
                }
                indexed++
                if (batch.size >= BATCH) {
                    val tPut = SystemClock.elapsedRealtime()
                    store.putChunks(batch); wroteAny = true; batch.clear()
                    Log.i(TAG, "AppSearch put batch ending at $indexed/${chunks.size} in ${SystemClock.elapsedRealtime() - tPut} ms")
                }
                if (indexed % 10 == 0 || indexed == chunks.size) {
                    val elapsed = SystemClock.elapsedRealtime() - embedStarted
                    val rate = if (indexed > 0) elapsed.toDouble() / indexed else 0.0
                    val eta = ((chunks.size - indexed) * rate).toLong()
                    Log.i(TAG, "embed $indexed/${chunks.size} last=${embedMs}ms avg=${rate.toInt()}ms/chunk eta=${eta / 1000}s page=${chunk.pageNumber} dim=${vec.size}")
                    onProgress(IndexingProgress(docHash, Stage.EMBEDDING, indexed, chunks.size, detail = "${rate.toInt()} ms/chunk"))
                } else if (indexed % 5 == 0) {
                    onProgress(IndexingProgress(docHash, Stage.EMBEDDING, indexed, chunks.size))
                }
            }
            if (batch.isNotEmpty()) { store.putChunks(batch); wroteAny = true }
            onProgress(IndexingProgress(docHash, Stage.FINALIZING, 1, 1))
            store.flush()
            Log.i(TAG, "indexed $displayName: $pageCount pages, ${chunks.size} chunks in ${SystemClock.elapsedRealtime() - embedStarted} ms embed+put")
            return DocumentInfo(docHash, displayName, pageCount, chunks.size, docScript.name)
        } catch (t: Throwable) {
            if (wroteAny) runCatching { store.removeDocument(docHash) }
            if (t is CancellationException) Log.i(TAG, "indexing cancelled for $displayName") else Log.e(TAG, "indexing failed", t)
            throw t
        }
    }

    companion object {
        private const val TAG = "IndexPdfUseCase"
        const val BATCH = 100
    }
}
