package com.example.pdfgemmarag.eval

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import android.util.Log
import com.example.pdfgemmarag.inference.pdf.AprysePdfExtractor
import com.example.pdfgemmarag.inference.pdf.StructureAnalyzer
import com.example.pdfgemmarag.inference.store.DocumentStructureManifestStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Fast fixture assertion against the real indexed PDF; production detection remains generic. */
@RunWith(AndroidJUnit4::class)
class SinglePdfStructureDeviceTest {
    @Test
    fun dumpConfiguredPageGeometry() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val pageNumber = InstrumentationRegistry.getArguments().getString("pageNumber")?.toIntOrNull() ?: return
        val manifest = DocumentStructureManifestStore(context).list().firstOrNull()
            ?: error("No indexed PDF manifest is installed")
        val source = context.filesDir.resolve("docs/${manifest.documentHash}.pdf")
        AprysePdfExtractor(context).open(source.absolutePath).use { document ->
            document.extractPage(pageNumber).lines.sortedWith(
                compareBy<com.example.pdfgemmarag.inference.pdf.LineBox> { it.box.top }.thenBy { it.box.left },
            ).forEachIndexed { index, line ->
                Log.i(
                    "PDF_GEOMETRY",
                    "%03d x=%.1f..%.1f y=%.1f..%.1f '%s'".format(
                        index, line.box.left, line.box.right, line.box.top, line.box.bottom,
                        line.text.replace('\n', ' '),
                    ),
                )
            }
        }
    }

    @Test
    fun concreteMixHasTwoDistinctResolvableSections() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val manifests = DocumentStructureManifestStore(context).list()
        val manifest = manifests.firstOrNull { it.sections.any { section -> section.startPage == 17 } }
            ?: error("The Division 03 test PDF manifest is not installed")
        val matches = manifest.sections.filter {
            StructureAnalyzer.normalizeHeading(it.title) == "concrete mix"
        }
        assertEquals("Expected exactly two CONCRETE MIX sections", 2, matches.size)
        assertEquals(setOf("03300", "03310"), matches.map { it.specificationNumber }.toSet())
        assertEquals(setOf(17, 60), matches.map { it.startPage }.toSet())
        assertTrue(matches.all { it.orderedChunkIds.isNotEmpty() })
    }
}
