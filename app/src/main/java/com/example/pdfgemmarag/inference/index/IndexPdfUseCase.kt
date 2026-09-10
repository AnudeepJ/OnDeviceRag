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
import com.example.pdfgemmarag.inference.pdf.StructureAnalyzer
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
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
    private val structure: StructureAnalyzer = StructureAnalyzer(TableClusterer()),
    /** Suspends while the device is thermally throttled. */
    private val awaitCool: suspend () -> Unit = {},
) {

    suspend fun run(
        docHash: String,
        pdfPath: String,
        displayName: String,
        onProgress: (IndexingProgress) -> Unit,
    ): DocumentInfo {
        val previousNamespace = runCatching { store.loadManifest(docHash)?.indexNamespace }.getOrNull()
        val buildId = System.currentTimeMillis().toString(36)
        val stagingNamespace = "$docHash:v${DocumentStructureManifest.INDEX_VERSION}:$buildId"
        try {
            onProgress(IndexingProgress(docHash, Stage.PREPARING, 0, 1))

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
            onProgress(IndexingProgress(docHash, Stage.CHUNKING, 0, layouts.size))
            val stripped = stripper.strip(layouts)
            val contents = stripped.mapIndexed { i, layout ->
                currentCoroutineContext().ensureActive()
                structure.analyse(layout).also {
                    if ((i + 1) % 2 == 0 || i + 1 == stripped.size) {
                        onProgress(IndexingProgress(docHash, Stage.CHUNKING, i + 1, stripped.size))
                    }
                }
            }
            val chunks = chunker.reindex(chunker.chunk(docHash, contents)
                .flatMap { chunk ->
                    chunker.fitToTokenWindow(chunk, embedder.sequenceLength - 2) { text ->
                        embedder.tokenCount("", text)
                    }
                })
            val allText = contents.joinToString("\n") { pc ->
                pc.segments.joinToString("\n") {
                    when (it) {
                        is com.example.pdfgemmarag.inference.pdf.Segment.Heading -> it.text
                        is com.example.pdfgemmarag.inference.pdf.Segment.Paragraph -> it.text
                        is com.example.pdfgemmarag.inference.pdf.Segment.ListBlock ->
                            it.items.joinToString("\n") { item -> "${item.label} ${item.text}" }
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
            val centroidSums = HashMap<String, FloatArray>()
            val centroidCounts = HashMap<String, Int>()
            var indexed = 0
            val now = System.currentTimeMillis()
            val embedStarted = SystemClock.elapsedRealtime()
            Log.i(TAG, "embedding ${chunks.size} chunks on ${embedder.backend} seq=${embedder.sequenceLength}")
            for (chunk in chunks) {
                currentCoroutineContext().ensureActive(); awaitCool()
                val t0 = SystemClock.elapsedRealtime()
                // Tables are embedded as header-qualified row facts (plus caption) so column
                // vocabulary from a question matches the row; prose embeds its body under its path.
                val embedText = if (chunk.isTable) chunk.retrievalText.removePrefix(chunk.sectionPath).trim() else chunk.bodyText
                val vec = embedder.embedDocument(chunk.sectionPath, embedText)
                val sum = centroidSums.getOrPut(chunk.sectionId) { FloatArray(vec.size) }
                for (dimension in vec.indices) sum[dimension] += vec[dimension]
                centroidCounts[chunk.sectionId] = (centroidCounts[chunk.sectionId] ?: 0) + 1
                val embedMs = SystemClock.elapsedRealtime() - t0
                batch += PdfChunkDocument().apply {
                    namespace = stagingNamespace
                    id = DocumentStructureManifest.chunkId(chunk.chunkIndex)
                    creationTimestampMillis = now
                    text = chunk.retrievalText
                    bodyText = chunk.bodyText
                    retrievalText = chunk.retrievalText
                    this.docHash = docHash
                    pageNumber = chunk.pageNumber
                    chunkIndex = chunk.chunkIndex
                    isTable = chunk.isTable
                    sectionId = chunk.sectionId
                    sectionTitle = chunk.sectionTitle
                    sectionPath = chunk.sectionPath
                    specificationNumber = chunk.specificationNumber
                    sectionNumber = chunk.sectionNumber
                    identifierAtoms = chunk.identifierAtoms.joinToString(" ")
                    contentKind = chunk.contentKind
                    tableId = chunk.tableId
                    tableNumber = chunk.tableNumber
                    tableCaption = chunk.tableCaption
                    positionInSection = chunk.positionInSection
                    continuesFromChunkIndex = chunk.continuesFromChunkIndex ?: -1
                    continuesToChunkIndex = chunk.continuesToChunkIndex ?: -1
                    indexVersion = DocumentStructureManifest.INDEX_VERSION
                    embeddingSignature = EmbeddingGemmaEmbedder.MODEL_SIGNATURE
                    docName = displayName
                    script = docScript.name
                    this.pageCount = pageCount
                    embedding = EmbeddingVector(vec, EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
                }
                indexed++
                if (batch.size >= BATCH) {
                    val tPut = SystemClock.elapsedRealtime()
                    store.putChunks(batch); batch.clear()
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
            if (batch.isNotEmpty()) store.putChunks(batch)
            onProgress(IndexingProgress(docHash, Stage.FINALIZING, 1, 1))
            store.flush()
            val centroids = centroidSums.mapValues { (sectionId, sum) ->
                val count = centroidCounts.getValue(sectionId).coerceAtLeast(1)
                for (i in sum.indices) sum[i] /= count
                EmbeddingGemmaEmbedder.l2Normalize(sum)
                sum
            }
            val manifest = DocumentStructureManifest.fromChunks(
                documentHash = docHash,
                namespace = stagingNamespace,
                signature = EmbeddingGemmaEmbedder.MODEL_SIGNATURE,
                chunks = chunks,
                tokenCount = { embedder.tokenCount("", it) },
                centroids = centroids,
            )
            // Publication is the commit point. Until this succeeds, every query keeps using the
            // previous complete namespace.
            store.publishManifest(manifest)
            // Every earlier namespace of this document is now stale: the previous complete index,
            // a legacy V1 namespace, or an index whose manifest version can no longer be read.
            runCatching { store.removeStaleNamespaces(docHash, keep = stagingNamespace) }
                .onSuccess { if (it.isNotEmpty()) Log.i(TAG, "removed stale namespaces $it") }
                .onFailure { Log.w(TAG, "old namespace cleanup failed", it) }
            if (previousNamespace != null && previousNamespace != stagingNamespace) {
                runCatching { store.removeNamespace(previousNamespace) }
            }
            Log.i(TAG, "indexed $displayName: $pageCount pages, ${chunks.size} chunks in ${SystemClock.elapsedRealtime() - embedStarted} ms embed+put")
            return DocumentInfo(
                docHash, displayName, pageCount, chunks.size, docScript.name,
                DocumentStructureManifest.INDEX_VERSION, stagingNamespace,
            )
        } catch (t: Throwable) {
            // A batch can partially succeed even when AppSearch reports an aggregate failure.
            runCatching { store.removeNamespace(stagingNamespace) }
            if (t is CancellationException) Log.i(TAG, "indexing cancelled for $displayName") else Log.e(TAG, "indexing failed", t)
            throw t
        }
    }

    companion object {
        private const val TAG = "IndexPdfUseCase"
        const val BATCH = 100
    }
}
