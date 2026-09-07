package com.example.pdfgemmarag.spikes

import android.os.SystemClock
import android.util.Log
import androidx.appsearch.app.EmbeddingVector
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.inference.embed.EmbeddingGemmaEmbedder
import com.example.pdfgemmarag.inference.store.AppSearchVectorStore
import com.example.pdfgemmarag.inference.store.PdfChunkDocument
import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.SectionRecord
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.random.Random

/**
 * Spike 3: AppSearch 1.1.0 `LocalStorage` feature support, hybrid (keyword OR semantic) query with
 * the advanced ranking expression, CJK keyword hits, namespace isolation and remove-by-namespace.
 * Uses a private database name so the app's own index is untouched.
 */
@RunWith(AndroidJUnit4::class)
class Spike3AppSearchHybridTest {

    private lateinit var store: AppSearchVectorStore
    private val nsA = "spike3-a-${System.currentTimeMillis()}"
    private val nsB = "spike3-b-${System.currentTimeMillis()}"

    private fun vec(seed: Int): FloatArray {
        val r = Random(seed)
        val v = FloatArray(512) { r.nextFloat() - 0.5f }
        EmbeddingGemmaEmbedder.l2Normalize(v)
        return v
    }

    /** A vector that is close (cosine ~0.9) to [base] but not identical. */
    private fun near(base: FloatArray, seed: Int): FloatArray {
        val n = vec(seed)
        val v = FloatArray(base.size) { base[it] * 0.95f + n[it] * 0.3f }
        EmbeddingGemmaEmbedder.l2Normalize(v)
        return v
    }

    private fun doc(ns: String, i: Int, text: String, v: FloatArray, page: Int = i + 1) = PdfChunkDocument().apply {
        namespace = ns; id = "$ns:$i"; creationTimestampMillis = System.currentTimeMillis()
        this.text = text; pageNumber = page; chunkIndex = i; docName = "spike"; script = "MIXED"; pageCount = 10
        bodyText = text; retrievalText = text; docHash = ns; sectionId = "section-a"
        sectionTitle = "Example"; sectionPath = "Example"; contentKind = "PARAGRAPH"
        indexVersion = DocumentStructureManifest.INDEX_VERSION; embeddingSignature = EmbeddingGemmaEmbedder.MODEL_SIGNATURE
        embedding = EmbeddingVector(v, EmbeddingGemmaEmbedder.MODEL_SIGNATURE)
    }

    @Before
    fun open(): Unit = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        store = AppSearchVectorStore.open(ctx, "spike3_test_db", enforceProcess = false)
    }

    @After
    fun close(): Unit = runBlocking {
        runCatching { store.removeDocument(nsA); store.removeDocument(nsB) }
        store.close()
    }

    @Test
    fun featureSupport() {
        Log.i("SPIKE3", "features:\n${store.features}")
        assertTrue("hybrid search features missing:\n${store.features}", store.features.hybridOk)
    }

    @Test
    fun hybridQueryCjkKeywordsAndVectors(): Unit = runBlocking {
        val base = vec(1)
        val docs = listOf(
            doc(nsA, 0, "東京都は日本の首都です。人口は約千四百万人。", vec(10)),
            doc(nsA, 1, "北京是中华人民共和国的首都。", vec(11)),
            doc(nsA, 2, "Paris is the capital of France.", near(base, 12)),
            doc(nsA, 3, "| Year | Revenue |\n| 2023 | 4.2M |", vec(13)),
            doc(nsA, 4, "서울은 대한민국의 수도입니다.", vec(14)),
            doc(nsA, 5, "Unrelated filler about cooking pasta with garlic.", vec(15)),
        )
        val t0 = SystemClock.elapsedRealtime()
        store.putChunks(docs)
        val putMs = SystemClock.elapsedRealtime() - t0

        suspend fun top(label: String, text: String, q: FloatArray, floor: Double = 0.3): String? {
            val t = SystemClock.elapsedRealtime()
            val hits = store.search(nsA, text, q, topK = 3, similarityFloor = floor)
            Log.i("SPIKE3", "$label -> ${hits.map { it.chunkId.substringAfter(':') + "@" + "%.3f".format(it.score) }} ${SystemClock.elapsedRealtime() - t}ms")
            return hits.firstOrNull()?.chunkId
        }
        // Vector-only: the query vector is near doc 2 and the keyword is nonsense.
        assertEquals("$nsA:2", top("vector-only", "zzzzqqq", base))
        // Keyword-only hits with an orthogonal vector (floor keeps unrelated vectors out).
        assertEquals("$nsA:0", top("ja keyword 首都 東京", "東京", vec(99)))
        assertEquals("$nsA:1", top("zh keyword 北京", "北京", vec(99)))
        assertEquals("$nsA:4", top("ko keyword 서울", "서울", vec(99)))
        assertEquals("$nsA:2", top("en keyword Paris", "Paris", vec(99)))
        assertEquals("$nsA:3", top("table keyword Revenue", "Revenue", vec(99)))
        // Prefix match on a Latin token.
        assertEquals("$nsA:2", top("en prefix capi", "capi", vec(99)))
        // Both signals agree -> hybrid ranking keeps it first.
        assertEquals("$nsA:2", top("hybrid Paris + near-vector", "Paris", base))
        // Shared token "首都" appears in ja and zh docs: both must be hits.
        val shared = store.search(nsA, "首都", vec(99), topK = 5, similarityFloor = 0.3).map { it.chunkId }.toSet()
        assertTrue("首都 should hit ja and zh docs, got $shared", shared.containsAll(setOf("$nsA:0", "$nsA:1")))

        val cit = store.getCitation("$nsA:0")
        assertEquals(1, cit?.pageNumber)
        Log.i("SPIKE3", "put ${docs.size} docs in ${putMs}ms")
    }

    @Test
    fun namespaceIsolationAndRemove(): Unit = runBlocking {
        val shared = vec(7)
        store.putChunks(listOf(doc(nsA, 0, "alpha document text", shared), doc(nsB, 0, "alpha document text", shared)))
        val onlyA = store.search(nsA, "alpha", shared, topK = 5, similarityFloor = 0.0)
        assertEquals(listOf("$nsA:0"), onlyA.map { it.chunkId })
        store.removeDocument(nsA)
        assertTrue(store.search(nsA, "alpha", shared, topK = 5, similarityFloor = 0.0).isEmpty())
        assertEquals(1, store.search(nsB, "alpha", shared, topK = 5, similarityFloor = 0.0).size)
        val docs = store.listDocuments()
        assertTrue(docs.any { it.docHash == nsB } && docs.none { it.docHash == nsA })
    }

    @Test
    fun directFetchUsesExplicitNamespaceAndRestoresManifestOrder(): Unit = runBlocking {
        val docs = listOf(doc(nsA, 0, "first", vec(20)), doc(nsA, 1, "second", vec(21)))
        store.putChunks(docs)
        val manifest = DocumentStructureManifest(
            documentHash = nsA,
            indexNamespace = nsA,
            indexVersion = DocumentStructureManifest.INDEX_VERSION,
            embeddingSignature = EmbeddingGemmaEmbedder.MODEL_SIGNATURE,
            sections = listOf(SectionRecord("section-a", "", "1.1", "Example", "Example", 1, 2, listOf("$nsA:1", "$nsA:0"), 10)),
        )
        val fetched = store.getChunks(manifest, manifest.sections.single().orderedChunkIds)
        assertEquals(listOf("$nsA:1", "$nsA:0"), fetched.map { it.chunkId })
        assertTrue(fetched.all { it.indexNamespace == nsA })
    }

    @Test
    fun listDocumentsReturnsCanonicalHashInsteadOfStagedNamespace(): Unit = runBlocking {
        val canonicalHash = "$nsA-canonical"
        val stagedNamespace = "$canonicalHash:v21:stage"
        try {
            val stored = doc(stagedNamespace, 0, "staged document", vec(31)).apply {
                docHash = canonicalHash
            }
            store.putChunk(stored)
            store.publishManifest(
                DocumentStructureManifest(
                    documentHash = canonicalHash,
                    indexNamespace = stagedNamespace,
                    indexVersion = DocumentStructureManifest.INDEX_VERSION,
                    embeddingSignature = EmbeddingGemmaEmbedder.MODEL_SIGNATURE,
                    sections = listOf(
                        SectionRecord(
                            "section-a", "", "1.1", "Example", "Example", 1, 1,
                            listOf(stored.id), 10,
                        ),
                    ),
                ),
            )

            val listed = store.listDocuments().single { it.displayName == "spike" && it.docHash == canonicalHash }
            assertEquals(canonicalHash, listed.docHash)
            assertEquals(stagedNamespace, listed.activeIndexNamespace)
            assertEquals(DocumentStructureManifest.INDEX_VERSION, listed.indexVersion)
        } finally {
            store.removeDocument(canonicalHash)
        }
    }

    @Test
    fun explicitNamespaceAllowsFactSearchWhenManifestIsCorrupt(): Unit = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val canonicalHash = "$nsA-fallback"
        val stagedNamespace = "$canonicalHash:v21:published"
        val source = doc(stagedNamespace, 0, "fallback fact value alpha", vec(37)).apply {
            docHash = canonicalHash
        }
        val manifestFile = java.io.File(ctx.filesDir, "rag_manifests/$canonicalHash.json")
        try {
            store.putChunk(source)
            manifestFile.parentFile?.mkdirs()
            manifestFile.writeText("not valid json")

            val hits = store.search(
                canonicalHash,
                "fallback fact alpha",
                vec(37),
                topK = 3,
                similarityFloor = 0.0,
                indexNamespace = stagedNamespace,
            )

            assertEquals(listOf(source.id), hits.map { it.chunkId })
        } finally {
            manifestFile.delete()
            store.removeNamespace(stagedNamespace)
        }
    }

    @Test
    fun explicitSpecificationFilterExcludesOtherSpecifications(): Unit = runBlocking {
        val shared = vec(41)
        val first = doc(nsA, 0, "maximum water cement ratios", shared).apply {
            specificationNumber = "03300"
        }
        val second = doc(nsA, 1, "maximum water cement ratios", shared).apply {
            specificationNumber = "03310"
        }
        store.putChunks(listOf(first, second))

        val hits = store.search(
            nsA,
            "maximum water cement ratios in specification 03310",
            shared,
            topK = 4,
            similarityFloor = 0.0,
            specificationNumber = "03310",
        )

        assertEquals(listOf("03310"), hits.map { it.specificationNumber }.distinct())
        assertEquals(listOf(second.id), hits.map { it.chunkId })
    }

    @Test
    fun bulkPutAndSearchLatency(): Unit = runBlocking {
        // ~500-page document at 3 chunks/page.
        val n = 1500
        val docs = (0 until n).map { i -> doc(nsB, i, "chunk $i text about topic ${i % 37} and item ${i % 11}", vec(1000 + i), page = i / 3 + 1) }
        val t0 = SystemClock.elapsedRealtime()
        store.putChunks(docs)
        val putMs = SystemClock.elapsedRealtime() - t0
        val q = near(vec(1000 + 777), 5)
        val t1 = SystemClock.elapsedRealtime()
        val hits = store.search(nsB, "topic", q, topK = 8, similarityFloor = 0.3)
        val searchMs = SystemClock.elapsedRealtime() - t1
        Log.i("SPIKE3", "bulk: put $n docs in ${putMs}ms (${putMs * 1.0 / n}ms/doc); hybrid search over $n vectors ${searchMs}ms; top=${hits.firstOrNull()?.chunkId}")
        assertEquals("$nsB:777", hits.firstOrNull()?.chunkId)
        assertTrue("search too slow: ${searchMs}ms", searchMs < 5000)
    }
}
