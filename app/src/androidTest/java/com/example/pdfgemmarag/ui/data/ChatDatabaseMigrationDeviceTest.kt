package com.example.pdfgemmarag.ui.data

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.RagApplication
import com.example.pdfgemmarag.ui.RagViewModel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.flow.first
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Verifies the non-destructive V1 -> V2 Room columns against the real on-device database. */
@RunWith(AndroidJUnit4::class)
class ChatDatabaseMigrationDeviceTest {
    @Test
    fun legacyReadyDocumentsAreMarkedForReindexWithoutDeletingRows(): Unit = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val db = (context.applicationContext as RagApplication).database
        val before = db.documents().observeAllSnapshot()
        db.documents().markLegacyIndexes(RagViewModel.CURRENT_INDEX_VERSION)
        val after = db.documents().observeAllSnapshot()
        assertTrue("document rows were lost during migration", after.size >= before.size)
        assertTrue(after.none { it.indexVersion < RagViewModel.CURRENT_INDEX_VERSION && it.status == RagViewModel.STATUS_READY })
    }

    private suspend fun DocumentDao.observeAllSnapshot(): List<DocumentEntity> =
        observeAll().first()
}
