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
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.DocumentInfo
import com.example.pdfgemmarag.core.model.EngineStatus
import com.example.pdfgemmarag.core.model.GenerationStats
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.core.model.QaPair
import com.example.pdfgemmarag.inference.service.AiInferenceService
import com.example.pdfgemmarag.inference.service.IAiInferenceService
import com.example.pdfgemmarag.inference.service.IEngineCallback
import com.example.pdfgemmarag.inference.service.IStreamCallback
import com.example.pdfgemmarag.inference.service.getCitationInNamespaceAsync
import com.example.pdfgemmarag.inference.service.listDocumentsAsync
import com.example.pdfgemmarag.inference.store.DocumentStructureManifest
import com.example.pdfgemmarag.inference.store.DocumentStructureManifestStore
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Runs strict source-backed regression checks against the indexed real PDF and installed model. */
@RunWith(AndroidJUnit4::class)
class SinglePdfBaselineDeviceTest {

    private data class Case(
        val id: String,
        val intent: String,
        val question: String,
        val expectedPages: Set<Int>,
        val requireAllPages: Boolean,
        val requiredPhrases: List<String>,
        val forbiddenPhrases: List<String>,
        val forbiddenPages: Set<Int>,
        val expectedRefusal: Boolean,
        val expectedClarification: Boolean,
        val historyFrom: String?,
        /** Every group requires at least one alternative; useful for source-valid notation variants. */
        val requiredPhraseAlternatives: List<List<String>> = emptyList(),
        /** Cited pages must fall into at least this many distinct document quartiles (overview breadth). */
        val requiredPageQuartiles: Int = 0,
    )

    private data class Turn(
        val case: Case,
        val answer: String,
        val pages: List<Int>,
        val error: String?,
        val modelTtftMs: Long,
        val visibleTtftMs: Long,
        val totalMs: Long,
        val tokensPerSecond: Double,
        val backend: String,
        val retrievedChunks: Int,
        val contextTokens: Int,
        val groundingFailure: Boolean,
        val sourceSectionId: String,
        val retrievalPassed: Boolean,
        val answerPassed: Boolean,
        val retrievalReasons: List<String>,
        val answerReasons: List<String>,
        val intent: String = "",
        val answeredBy: String = "",
        val groundingReasons: String = "",
        val prefillTokensPerSecond: Double = 0.0,
        val decodeTokensPerSecond: Double = 0.0,
        val prefillTokens: Int = 0,
    ) {
        /** First failing pipeline stage, so a report names where to look rather than one pass flag. */
        val stage: String
            get() = when {
                !retrievalPassed -> "RETRIEVAL"
                groundingFailure || answerReasons.any { "unverified" in it || "citation" in it } -> "GROUNDING"
                !answerPassed -> "ANSWER"
                else -> "PASS"
            }
    }

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
     * Selects the document by exact content hash (`-e documentHash`), then by display name
     * (`-e documentName`), then by the suite default. Page-count guessing is only a last resort for
     * the legacy focused baseline.
     */
    private fun selectDocument(defaultHash: String? = null, defaultName: String? = null, allowPageCountFallback: Boolean = false): DocumentInfo {
        val svc = requireNotNull(service)
        val arguments = InstrumentationRegistry.getArguments()
        val requestedHash = arguments.getString("documentHash")?.ifBlank { null } ?: defaultHash
        val requestedName = arguments.getString("documentName")?.ifBlank { null } ?: defaultName
        val docs = runBlocking { svc.listDocumentsAsync() }
        val doc = (if (requestedHash != null) {
            docs.firstOrNull { it.docHash.equals(requestedHash, true) }
        } else requestedName?.let { name -> docs.firstOrNull { it.displayName == name } }
            ?: docs.takeIf { allowPageCountFallback && requestedName == null }?.firstOrNull { it.pageCount >= 70 })
            ?: error("Required test PDF (hash=$requestedHash name=$requestedName) is not indexed on this device; indexed=${docs.map { it.displayName + ':' + it.docHash.take(8) }}")
        Log.i(TAG, "selected document '${doc.displayName}' hash=${doc.docHash} pages=${doc.pageCount} index=v${doc.indexVersion} ns=${doc.activeIndexNamespace}")
        return doc
    }

    @Test
    fun captureFocusedBaseline() {
        val svc = requireNotNull(service)
        val arguments = InstrumentationRegistry.getArguments()
        val doc = selectDocument(defaultHash = DIVISION03_HASH, allowPageCountFallback = true)
        arguments.getString("dumpChunkRange")?.let { requested ->
            val bounds = requested.split('-').map(String::toInt)
            val manifest = requireNotNull(DocumentStructureManifestStore(ctx).load(doc.docHash))
            for (index in bounds.first()..bounds.last()) {
                val id = DocumentStructureManifest.chunkId(index)
                val citation = runBlocking { svc.getCitationInNamespaceAsync(manifest.indexNamespace, id) } ?: continue
                Log.i(TAG, "DUMP $id p${citation.pageNumber} kind=${citation.contentKind} section=${citation.sectionId} '${citation.text.replace('\n', ' ')}'")
            }
            return
        }
        runBaseline(
            svc,
            doc,
            arguments.getString("casesAsset") ?: DEFAULT_CASES_ASSET,
            arguments.getString("caseId"),
        )
    }

    /** Runs the generated real-user question bank in a normal instrumentation suite. */
    @Test
    fun captureGeneratedLiveQa() = runGeneratedBaseline(GENERATED_CASES_ASSET)

    /** Runs the strict construction-safety bank only against its exact indexed document. */
    @Test
    fun captureConstructionSafetyQa() = runNamedBaseline(CONSTRUCTION_SAFETY_CASES_ASSET, SAFETY_PDF_HASH, "safety.pdf")

    /** Red regression bank locked from the 209-page safety manual live failures (roadmap M0). */
    @Test
    fun captureSafetyPdfQa() = runNamedBaseline(SAFETY_PDF_CASES_ASSET, SAFETY_PDF_HASH, "safety.pdf")

    /** 50-case laboratory safety manual bank (17 table cases) against its exact document. */
    @Test
    fun captureSafetyManualQa() = runNamedBaseline(SAFETY_MANUAL_CASES_ASSET, SAFETY_MANUAL_HASH, "SafetyManual.pdf")

    /** The legacy "more" asset is retained for focused runs; verify it cannot silently diverge. */
    @Test
    fun generatedAssetDuplicatesRemainIdentical() {
        val primary = loadCases(GENERATED_CASES_ASSET).associateBy { it.id }
        val legacy = loadCases(GENERATED_MORE_CASES_ASSET)
        assertEquals("duplicate ids inside legacy generated bank", legacy.size, legacy.map { it.id }.distinct().size)
        legacy.forEach { case -> assertEquals("divergent duplicate fixture ${case.id}", case, primary[case.id]) }
    }

    private fun runGeneratedBaseline(assetName: String) {
        val svc = requireNotNull(service)
        val doc = selectDocument(defaultHash = DIVISION03_HASH, allowPageCountFallback = true)
        runBaseline(svc, doc, assetName, requestedCase = InstrumentationRegistry.getArguments().getString("caseId")?.ifBlank { null })
    }

    private fun runNamedBaseline(assetName: String, defaultHash: String, defaultDocumentName: String) {
        val svc = requireNotNull(service)
        val doc = selectDocument(defaultHash = defaultHash, defaultName = defaultDocumentName)
        runBaseline(svc, doc, assetName, requestedCase = InstrumentationRegistry.getArguments().getString("caseId")?.ifBlank { null })
    }

    private fun runBaseline(
        svc: IAiInferenceService,
        doc: DocumentInfo,
        assetName: String,
        requestedCase: String?,
    ) {
        ensureEngine(svc)
        val cases = loadCases(assetName).filter { requestedCase == null || it.id == requestedCase }
        check(cases.isNotEmpty()) { "Unknown baseline caseId: $requestedCase" }
        val completed = LinkedHashMap<String, Turn>()
        val reportCases = JSONArray()

        for (case in cases) {
            val prior = case.historyFrom?.let(completed::get)
            val history = if (prior == null) emptyList()
            else listOf(QaPair(prior.case.question, prior.answer, prior.sourceSectionId))
            val turn = ask(svc, doc.docHash, doc.activeIndexNamespace, case, history, doc.pageCount)
            completed[case.id] = turn
            reportCases.put(turn.toJson())
            Log.i(TAG, "${case.id}: stage=${turn.stage} retrieval=${turn.retrievalPassed} answer=${turn.answerPassed} pages=${turn.pages} backend=${turn.backend} " +
                "intent=${turn.intent} by=${turn.answeredBy.ifBlank { "MODEL" }} " +
                "modelTtft=${turn.modelTtftMs} visibleTtft=${turn.visibleTtftMs} total=${turn.totalMs} " +
                "tok_s=${"%.2f".format(Locale.ROOT, turn.tokensPerSecond)} prefill=${turn.prefillTokens}tok@${"%.0f".format(Locale.ROOT, turn.prefillTokensPerSecond)} " +
                "ctx=${turn.contextTokens} groundingFailure=${turn.groundingFailure} grounding='${turn.groundingReasons}' " +
                "retrievalReasons=${turn.retrievalReasons} answerReasons=${turn.answerReasons}")
            Log.i(TAG, "${case.id} answer='${turn.answer.take(1200).replace('\n', ' ')}'")
        }

        val retrievalPassed = completed.values.count { it.retrievalPassed }
        val answerPassed = completed.values.count { it.answerPassed }
        val byStage = completed.values.groupingBy { it.stage }.eachCount()
        val generated = completed.values.filter { it.answeredBy.isBlank() && it.modelTtftMs > 0 }
        val report = JSONObject()
            .put("document", doc.displayName)
            .put("docHash", doc.docHash)
            .put("indexVersion", doc.indexVersion)
            .put("indexNamespace", doc.activeIndexNamespace)
            .put("device", android.os.Build.MODEL)
            .put("androidRelease", android.os.Build.VERSION.RELEASE)
            .put("total", completed.size)
            .put("retrievalPassed", retrievalPassed)
            .put("answerPassed", answerPassed)
            .put("byStage", JSONObject(byStage.mapValues { it.value as Any }))
            .put("generatedTurns", generated.size)
            .put("medianModelTtftMs", generated.map { it.modelTtftMs }.sorted().let { if (it.isEmpty()) -1 else it[it.size / 2] })
            .put("medianDecodeTokensPerSecond", generated.map { it.decodeTokensPerSecond }.filter { it > 0 }.sorted().let { if (it.isEmpty()) 0.0 else it[it.size / 2] })
            .put("medianPrefillTokensPerSecond", generated.map { it.prefillTokensPerSecond }.filter { it > 0 }.sorted().let { if (it.isEmpty()) 0.0 else it[it.size / 2] })
            .put("cases", reportCases)
        val suiteName = assetName.substringBeforeLast('.').replace(Regex("[^a-zA-Z0-9_-]"), "_")
        val output = File(requireNotNull(ctx.getExternalFilesDir(null)), "single_pdf_baseline-$suiteName.json")
        output.writeText(report.toString(2))
        Log.i(TAG, "BASELINE retrieval=$retrievalPassed/${completed.size} answer=$answerPassed/${completed.size} stages=$byStage; report=${output.absolutePath}")
        assertTrue("baseline did not execute any cases", completed.isNotEmpty())
        assertEquals("retrieval regressions; report=$output", completed.size, retrievalPassed)
        assertEquals("answer regressions; report=$output", completed.size, answerPassed)
    }

    @Test
    fun documentOverviewUsesBroadStructuralContext() {
        val svc = requireNotNull(service)
        val doc = selectDocument(defaultHash = DIVISION03_HASH, allowPageCountFallback = true)
        ensureEngine(svc)
        val overview = Case(
            id = "document-overview",
            intent = "DOCUMENT_OVERVIEW",
            question = "Summarize this document",
            expectedPages = emptySet(),
            requireAllPages = false,
            requiredPhrases = emptyList(),
            forbiddenPhrases = emptyList(),
            forbiddenPages = emptySet(),
            expectedRefusal = false,
            expectedClarification = false,
            historyFrom = null,
        )

        val turn = ask(svc, doc.docHash, doc.activeIndexNamespace, overview, emptyList())

        assertTrue("overview failed: ${turn.error}", turn.error == null && turn.answer.isNotBlank())
        assertTrue("overview used only ${turn.retrievedChunks} chunks", turn.retrievedChunks >= 4)
        assertTrue("document overview must not establish one follow-up section", turn.sourceSectionId.isBlank())
    }

    @Test
    fun resolvedSectionMetadataSurvivesMultiCitationAnswerForFollowUp() {
        val svc = requireNotNull(service)
        val doc = selectDocument(defaultHash = DIVISION03_HASH, allowPageCountFallback = true)
        ensureEngine(svc)
        val summary = Case(
            "metadata-summary", "SECTION_SUMMARY",
            "Summarize section 2.05 concrete mix in specification 03300",
            setOf(17), false, emptyList(), emptyList(), emptySet(), false, false, null,
        )
        val first = ask(svc, doc.docHash, doc.activeIndexNamespace, summary, emptyList())
        assertTrue("planner did not publish a resolved source section", first.sourceSectionId.isNotBlank())

        val followUp = Case(
            "metadata-follow-up", "FOLLOW_UP", "What about its slump requirements?",
            setOf(17), false, emptyList(), emptyList(), emptySet(), false, false, null,
        )
        val second = ask(
            svc,
            doc.docHash,
            doc.activeIndexNamespace,
            followUp,
            listOf(QaPair(summary.question, first.answer, first.sourceSectionId)),
        )

        assertTrue("follow-up escaped the resolved section: pages=${second.pages}", 17 in second.pages)
        assertEquals(first.sourceSectionId, second.sourceSectionId)
    }

    private fun ensureEngine(svc: IAiInferenceService) {
        if (svc.engineStatus.state == EngineStatus.State.READY) return
        val model = ModelPaths.installedLlms(ctx).firstOrNull() ?: error("No installed Gemma model")
        val ready = CountDownLatch(1)
        var failure: String? = null
        svc.loadEngine(model.absolutePath, true, object : IEngineCallback.Stub() {
            override fun onProgress(stage: String) {
                Log.i(TAG, "engine: $stage")
            }

            override fun onReady(status: EngineStatus) {
                Log.i(TAG, "engine ready on ${status.backend}")
                ready.countDown()
            }

            override fun onFailed(message: String) {
                failure = message
                ready.countDown()
            }
        })
        assertTrue("engine load timed out", ready.await(180, TimeUnit.SECONDS))
        check(failure == null) { "engine load failed: $failure" }
    }

    private fun ask(
        svc: IAiInferenceService,
        docHash: String,
        activeIndexNamespace: String,
        case: Case,
        history: List<QaPair>,
        pageCount: Int = 0,
    ): Turn {
        val done = CountDownLatch(1)
        val answer = StringBuilder()
        val citations = ArrayList<Citation>()
        var error: String? = null
        var stats: GenerationStats? = null
        val started = SystemClock.elapsedRealtime()
        var visibleFirstToken = -1L
        var generationId = -1L
        generationId = svc.ask(docHash, activeIndexNamespace, case.question, history, object : IStreamCallback.Stub() {
            override fun onRetrieved(id: Long, value: List<Citation>) {
                if (id == generationId || generationId < 0) synchronized(citations) { citations += value }
            }

            override fun onToken(id: Long, token: String) {
                if (id != generationId && generationId >= 0) return
                if (visibleFirstToken < 0) visibleFirstToken = SystemClock.elapsedRealtime()
                synchronized(answer) { answer.append(token) }
            }

            override fun onDone(id: Long, value: GenerationStats) {
                if (id != generationId && generationId >= 0) return
                stats = value
                done.countDown()
            }

            override fun onError(id: Long, message: String) {
                if (id != generationId && generationId >= 0) return
                error = message
                done.countDown()
            }
        })
        if (!done.await(120, TimeUnit.SECONDS)) {
            runCatching { svc.cancelGeneration(generationId) }
            error = "Timed out after 120 seconds"
        }
        val elapsed = SystemClock.elapsedRealtime() - started
        val text = synchronized(answer) { answer.toString() }
        val pages = synchronized(citations) { citations.map { it.pageNumber }.distinct() }
        synchronized(citations) {
            Log.i(
                TAG,
                "${case.id} citation metadata=" + citations.distinctBy { it.chunkId }.take(12).joinToString {
                    "p${it.pageNumber}/c${it.chunkIndex}/spec='${it.specificationNumber}'/section='${it.sectionId.take(8)}'"
                },
            )
        }
        val retrievalReasons = scoreRetrieval(case, pages, pageCount)
        val groundingFailure = stats?.groundingFailure ?: false
        val answerReasons = scoreAnswer(case, text, error, groundingFailure)
        return Turn(
            case = case,
            answer = text,
            pages = pages,
            error = error,
            modelTtftMs = stats?.timeToFirstTokenMs ?: -1,
            visibleTtftMs = if (visibleFirstToken < 0) -1 else visibleFirstToken - started,
            totalMs = stats?.totalMs ?: elapsed,
            tokensPerSecond = stats?.approxTokensPerSecond ?: 0.0,
            backend = stats?.backend ?: "unknown",
            retrievedChunks = stats?.retrievedChunks ?: 0,
            contextTokens = stats?.contextTokensApprox ?: 0,
            groundingFailure = groundingFailure,
            sourceSectionId = stats?.sourceSectionId.orEmpty(),
            retrievalPassed = retrievalReasons.isEmpty(),
            answerPassed = answerReasons.isEmpty(),
            retrievalReasons = retrievalReasons,
            answerReasons = answerReasons,
            intent = stats?.intent.orEmpty(),
            answeredBy = stats?.answeredBy.orEmpty(),
            groundingReasons = stats?.groundingReasons.orEmpty(),
            prefillTokensPerSecond = stats?.prefillTokensPerSecond ?: 0.0,
            decodeTokensPerSecond = stats?.decodeTokensPerSecond ?: 0.0,
            prefillTokens = stats?.prefillTokens ?: 0,
        )
    }

    private fun scoreRetrieval(case: Case, pages: List<Int>, pageCount: Int): List<String> {
        val reasons = ArrayList<String>()
        val actualPages = pages.toSet()
        if (case.requiredPageQuartiles > 0 && pageCount > 0) {
            val quartiles = actualPages.map { page -> ((page - 1) * 4 / pageCount).coerceIn(0, 3) }.toSet()
            if (quartiles.size < case.requiredPageQuartiles) {
                reasons += "cited pages $actualPages span ${quartiles.size} quartile(s); need ${case.requiredPageQuartiles}"
            }
        }
        if (case.expectedPages.isNotEmpty()) {
            val pageOk = if (case.requireAllPages) actualPages.containsAll(case.expectedPages)
            else actualPages.any { it in case.expectedPages }
            if (!pageOk) reasons += "expected pages ${case.expectedPages}, got $actualPages"
        }
        val forbidden = actualPages.intersect(case.forbiddenPages)
        if (forbidden.isNotEmpty()) reasons += "distractor pages $forbidden"
        return reasons
    }

    private fun scoreAnswer(
        case: Case,
        answer: String,
        error: String?,
        groundingFailure: Boolean,
    ): List<String> {
        val reasons = ArrayList<String>()
        if (error != null) reasons += "error: $error"
        val normalized = canonical(answer)
        if (normalized.isBlank()) reasons += "empty answer"
        if (groundingFailure) reasons += "grounding filter rejected generated content"
        if ("[unverified value]" in normalized || INTERNAL_OR_MALFORMED_CITATION.containsMatchIn(normalized)) {
            reasons += "internal or malformed citation/value marker"
        }
        if (RAW_LATEX.containsMatchIn(answer)) reasons += "raw LaTeX control sequence in user-visible answer"
        val missing = case.requiredPhrases.filterNot { canonical(it) in normalized }
        if (missing.isNotEmpty()) reasons += "missing phrases $missing"
        val missingAlternatives = case.requiredPhraseAlternatives.filter { alternatives ->
            alternatives.none { canonical(it) in normalized }
        }
        if (missingAlternatives.isNotEmpty()) reasons += "missing phrase alternatives $missingAlternatives"
        val forbidden = case.forbiddenPhrases.filter { canonical(it) in normalized }
        if (forbidden.isNotEmpty()) reasons += "forbidden phrases $forbidden"
        if (case.expectedRefusal && !looksLikeRefusal(normalized)) reasons += "expected refusal"
        if (!case.expectedRefusal &&
            (case.requiredPhrases.isNotEmpty() || case.requiredPhraseAlternatives.isNotEmpty()) &&
            looksLikeRefusal(normalized)
        ) {
            reasons += "unexpected refusal contradicts a source-backed expected answer"
        }
        if (case.expectedClarification && !looksLikeClarification(normalized)) reasons += "expected clarification or both matching sections"
        if (case.expectedPages.isNotEmpty() && !case.expectedRefusal && !case.expectedClarification &&
            !PAGE_CITATION.containsMatchIn(answer)
        ) reasons += "missing user-visible page citation"
        return reasons
    }

    private fun loadCases(assetName: String? = null): List<Case> {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val path = assetName?.ifBlank { null } ?: "single_pdf_baseline.json"
        val raw = if (path.startsWith("/")) {
            File(path).readText()
        } else {
            testContext.assets.open(path).bufferedReader().use { it.readText() }
        }
        val array = JSONArray(raw)
        return (0 until array.length()).map { index ->
            val item = array.getJSONObject(index)
            Case(
                id = item.getString("id"),
                intent = item.getString("intent"),
                question = item.getString("question"),
                expectedPages = item.optJSONArray("expectedPages")?.ints().orEmpty(),
                requireAllPages = item.optBoolean("requireAllPages"),
                requiredPhrases = item.optJSONArray("requiredPhrases")?.strings().orEmpty(),
                forbiddenPhrases = item.optJSONArray("forbiddenPhrases")?.strings().orEmpty(),
                forbiddenPages = item.optJSONArray("forbiddenPages")?.ints().orEmpty(),
                expectedRefusal = item.optBoolean("expectedRefusal"),
                expectedClarification = item.optBoolean("expectedClarification"),
                historyFrom = item.optString("historyFrom").takeIf { it.isNotEmpty() },
                requiredPhraseAlternatives = item.optJSONArray("requiredPhraseAlternatives")?.stringLists().orEmpty(),
                requiredPageQuartiles = item.optInt("requiredPageQuartiles"),
            )
        }
    }

    private fun Turn.toJson() = JSONObject()
        .put("id", case.id)
        .put("intent", case.intent)
        .put("question", case.question)
        .put("retrievalPassed", retrievalPassed)
        .put("answerPassed", answerPassed)
        .put("stage", stage)
        .put("plannerIntent", intent)
        .put("answeredBy", answeredBy.ifBlank { "MODEL" })
        .put("groundingReasons", groundingReasons)
        .put("prefillTokens", prefillTokens)
        .put("prefillTokensPerSecond", prefillTokensPerSecond)
        .put("decodeTokensPerSecond", decodeTokensPerSecond)
        .put("retrievalReasons", JSONArray(retrievalReasons))
        .put("answerReasons", JSONArray(answerReasons))
        .put("retrievedPages", JSONArray(pages))
        .put("modelTtftMs", modelTtftMs)
        .put("visibleTtftMs", visibleTtftMs)
        .put("totalMs", totalMs)
        .put("tokensPerSecond", tokensPerSecond)
        .put("backend", backend)
        .put("retrievedChunks", retrievedChunks)
        .put("contextTokens", contextTokens)
        .put("groundingFailure", groundingFailure)
        .put("sourceSectionId", sourceSectionId)
        .put("error", error ?: JSONObject.NULL)
        .put("answer", answer)

    private fun JSONArray.ints(): Set<Int> = (0 until length()).map(::getInt).toSet()
    private fun JSONArray.strings(): List<String> = (0 until length()).map(::getString)
    private fun JSONArray.stringLists(): List<List<String>> =
        (0 until length()).map { getJSONArray(it).strings() }.also { groups ->
            require(groups.all { it.isNotEmpty() }) { "requiredPhraseAlternatives cannot contain an empty group" }
        }

    private fun canonical(value: String): String = value.lowercase(Locale.ROOT)
        .replace("\\pm", "+/-")
        .replace("±", "+/-")
        .replace(",", "")
        .replace(Regex("\\s+"), " ")

    private fun looksLikeRefusal(value: String): Boolean =
        "does not contain" in value || "do not contain" in value || "does not cover" in value || "do not cover" in value ||
            "not specified" in value || "no information" in value || "cannot find" in value || "not mention" in value

    private fun looksLikeClarification(value: String): Boolean =
        ("03300" in value && "03310" in value) || "which section" in value ||
            "which specification" in value || "clarify" in value

    companion object {
        private const val TAG = "SINGLE_PDF_BASELINE"
        private const val DEFAULT_CASES_ASSET = "single_pdf_baseline.json"
        private const val GENERATED_CASES_ASSET = "generated_live_qa.json"
        private const val GENERATED_MORE_CASES_ASSET = "generated_live_qa_more.json"
        private const val CONSTRUCTION_SAFETY_CASES_ASSET = "construction_safety_qa.json"
        private const val SAFETY_PDF_CASES_ASSET = "safety_pdf_qa.json"
        private const val SAFETY_MANUAL_CASES_ASSET = "safety_manual_qa.json"
        private const val SAFETY_MANUAL_HASH = "f7a26e28fdfc60153819ded2a2405695"
        /** Content hashes (first 32 hex chars of SHA-256, as computed on import); fixtures are bound to these documents. */
        private const val SAFETY_PDF_HASH = "6630f344c15e8c7588c40da4369d34b5"
        private const val DIVISION03_HASH = "086248dad810dc9368e844d3bcb1cab5"
        private val PAGE_CITATION = Regex("(?i)\\[page\\s+\\d+]")
        private val INTERNAL_OR_MALFORMED_CITATION = Regex("(?i)\\[e(?:\\d+|\\[|$)")
        private val RAW_LATEX = Regex("\\$[^$]*\\\\(?:pm|frac|text|mathrm)[^$]*\\$")
    }
}
