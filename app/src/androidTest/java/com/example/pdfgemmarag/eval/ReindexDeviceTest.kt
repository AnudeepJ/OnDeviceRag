package com.example.pdfgemmarag.eval

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.core.model.DocumentInfo
import com.example.pdfgemmarag.core.model.IndexingProgress
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.inference.service.AiInferenceService
import com.example.pdfgemmarag.inference.service.IAiInferenceService
import com.example.pdfgemmarag.inference.service.IIndexingCallback
import com.example.pdfgemmarag.inference.service.listDocumentsAsync
import com.example.pdfgemmarag.inference.store.DocumentStructureManifestStore
import com.example.pdfgemmarag.ui.RagViewModel
import com.example.pdfgemmarag.ui.data.ChatDatabase
import com.example.pdfgemmarag.ui.data.DocumentEntity
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Re-indexes an already imported PDF through the production service path (used after an index
 * version bump) and brings the UI database row in line, so the app shows the document as READY
 * with the new index version. Select the document with `-e documentHash <hash>`; without it every
 * document whose manifest is older than the current version is re-indexed.
 */
@RunWith(AndroidJUnit4::class)
class ReindexDeviceTest {

    private val ctx: Context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val connected = CountDownLatch(1)
    private var service: IAiInferenceService? = null
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            service = IAiInferenceService.Stub.asInterface(binder)
            connected.countDown()
        }

        override fun onServiceDisconnected(name: ComponentName) {
            service = null
        }
    }

    @Before
    fun bind() {
        assertTrue(ctx.bindService(Intent(ctx, AiInferenceService::class.java), connection, Context.BIND_AUTO_CREATE))
        assertTrue("inference service did not bind", connected.await(20, TimeUnit.SECONDS))
    }

    @After
    fun unbind() {
        runCatching { ctx.unbindService(connection) }
    }

    /**
     * Imports a PDF that was pushed to the device (`-e importPdf /data/local/tmp/x.pdf
     * -e displayName x.pdf`) exactly as the UI does: content-hash it, copy it to app storage and
     * create the document row, then fall through to indexing it.
     */
    private fun importIfRequested(db: ChatDatabase): DocumentInfo? {
        val source = InstrumentationRegistry.getArguments().getString("importPdf")?.ifBlank { null } ?: return null
        val name = InstrumentationRegistry.getArguments().getString("displayName")?.ifBlank { null } ?: File(source).name
        val digest = MessageDigest.getInstance("SHA-256")
        File(source).inputStream().use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) { val n = input.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
        }
        val hash = digest.digest().joinToString("") { "%02x".format(it) }.take(32)
        val target = ModelPaths.pdfFile(ctx, hash)
        if (!target.exists()) {
            target.parentFile?.mkdirs()
            File(source).copyTo(target, overwrite = true)
        }
        runBlocking {
            db.documents().upsert(
                DocumentEntity(hash, name, 0, 0, "", System.currentTimeMillis(), RagViewModel.STATUS_INDEXING, null, target.absolutePath),
            )
        }
        Log.i(TAG, "imported $name as $hash (${target.length()} bytes)")
        return DocumentInfo(hash, name, 0, 0, "", 0, hash)
    }

    @Test
    fun reindexToCurrentVersion() {
        val svc = requireNotNull(service)
        val requested = InstrumentationRegistry.getArguments().getString("documentHash")?.ifBlank { null }
        val force = InstrumentationRegistry.getArguments().getString("force") == "true"
        val db = ChatDatabase.create(ctx)
        val manifests = DocumentStructureManifestStore(ctx)
        val imported = importIfRequested(db)
        val listed = runBlocking { svc.listDocumentsAsync() }
        // A legacy index whose manifest version can no longer be read is invisible to the service;
        // the UI database still knows the document and where its PDF is.
        val rows = runBlocking { db.documents().observeAll().first() }
        val docs = (listed + rows.filter { row -> listed.none { it.docHash == row.hash } }
            .map { DocumentInfo(it.hash, it.name, it.pageCount, it.chunkCount, it.script, it.indexVersion, it.activeIndexNamespace) })
        Log.i(TAG, "documents: service=${listed.map { it.docHash.take(8) }} ui=${rows.map { "${it.name}:${it.hash.take(8)}:v${it.indexVersion}:${it.status}" }}")
        val targets = if (imported != null) listOf(imported) else docs.filter { doc ->
            (requested == null || doc.docHash.equals(requested, true)) &&
                (force || requested != null || doc.indexVersion < RagViewModel.CURRENT_INDEX_VERSION ||
                    runCatching { manifests.load(doc.docHash) }.getOrNull() == null)
        }
        check(targets.isNotEmpty()) { "nothing to re-index (requested=$requested); indexed=${docs.map { it.docHash.take(8) }}" }
        // Chat and indexing must not both hold big models on 6-8 GB devices.
        runCatching { svc.unloadEngine() }

        for (doc in targets) {
            val pdf = ModelPaths.pdfFile(ctx, doc.docHash)
            check(pdf.exists()) { "PDF for ${doc.displayName} missing at $pdf" }
            val done = CountDownLatch(1)
            var result: DocumentInfo? = null
            var failure: String? = null
            val started = SystemClock.elapsedRealtime()
            var lastLog = 0L
            svc.indexDocument(doc.docHash, pdf.absolutePath, doc.displayName, object : IIndexingCallback.Stub() {
                override fun onProgress(progress: IndexingProgress) {
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastLog > 5_000) {
                        lastLog = now
                        Log.i(TAG, "${doc.displayName}: ${progress.stage} ${progress.current}/${progress.total} ${progress.detail} (${(now - started) / 1000}s)")
                    }
                }

                override fun onCompleted(document: DocumentInfo) { result = document; done.countDown() }
                override fun onCancelled(docHash: String) { failure = "cancelled"; done.countDown() }
                override fun onFailed(docHash: String, message: String) { failure = message; done.countDown() }
            })
            assertTrue("indexing ${doc.displayName} timed out", done.await(45, TimeUnit.MINUTES))
            check(failure == null) { "indexing ${doc.displayName} failed: $failure" }
            val info = requireNotNull(result)
            Log.i(TAG, "REINDEXED ${info.displayName}: ${info.pageCount} pages, ${info.chunkCount} chunks, v${info.indexVersion}, ns=${info.activeIndexNamespace} in ${(SystemClock.elapsedRealtime() - started) / 1000}s")
            assertEquals(RagViewModel.CURRENT_INDEX_VERSION, info.indexVersion)
            val manifest = requireNotNull(manifests.load(info.docHash))
            val health = manifest.health()
            Log.i(TAG, "MANIFEST ${info.displayName}: sections=${manifest.sections.size} tables=${manifest.tables.size} topLevel=${health.topLevelCount} " +
                "kinds=${manifest.sections.groupingBy { it.kind.ifBlank { "ROOT" } }.eachCount()} degraded=${health.degraded} reasons=${health.reasons} quartiles=${health.quartileCoverage}")
            manifest.tables.take(12).forEach { table ->
                Log.i(TAG, "  table ${table.tableNumber.ifBlank { "-" }} '${table.caption.take(60)}' p${table.startPage}-${table.endPage} rows=${table.orderedRowChunkIds.size}")
            }
            manifest.topLevelSections().forEach { Log.i(TAG, "  top ${it.kind} ${it.printedNumber} '${it.title}' p${it.startPage}-${it.endPage} chunks=${it.orderedChunkIds.size}") }

            runBlocking {
                db.documents().get(info.docHash)?.let { row ->
                    db.documents().update(
                        row.copy(
                            pageCount = info.pageCount,
                            chunkCount = info.chunkCount,
                            script = info.script,
                            status = RagViewModel.STATUS_READY,
                            error = null,
                            indexedAt = System.currentTimeMillis(),
                            indexVersion = info.indexVersion,
                            activeIndexNamespace = info.activeIndexNamespace,
                        ),
                    )
                }
            }
        }
        db.close()
    }

    private companion object {
        const val TAG = "REINDEX"
    }
}
