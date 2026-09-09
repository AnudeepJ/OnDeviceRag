package com.example.pdfgemmarag.spikes

import android.content.Context
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.pdfgemmarag.core.model.ModelPaths
import com.example.pdfgemmarag.inference.ocr.MlKitOcr
import com.example.pdfgemmarag.inference.ocr.Script
import com.example.pdfgemmarag.inference.pdf.AprysePdfExtractor
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale

/**
 * Spike 6: measures Gemma Vision against the deterministic extraction paths on real PDF pages.
 *
 * This is deliberately an instrumentation-only experiment. Production indexing continues to use
 * Apryse first and ML Kit for image-only pages. The spike answers a narrower question: whether a
 * multimodal Gemma pass adds enough table/diagram/photo understanding to justify a selective visual
 * enrichment stage, and whether it could safely replace ML Kit for exact OCR.
 *
 * Optional instrumentation arguments:
 * - `visionPdfPath`: app-readable absolute PDF path. Defaults to the largest file in `files/docs`.
 * - `visionPages`: comma-separated one-based pages. Defaults to SafetyManual's table/photo/flowchart
 *   sample (`20,29,42`); override this for every new evaluation document.
 * - `visionMaxEdge`: Apryse render size, default 1536 pixels.
 *
 * Results are logged with tag `VISION_SPIKE` and written to
 * `files/vision_spike_report.json`, which can be collected with `adb shell run-as`.
 */
@RunWith(AndroidJUnit4::class)
class Spike6GemmaVisionDeviceTest {

    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val ctx: Context get() = instrumentation.targetContext

    private data class PageCase(
        val page: Int,
        val kind: String,
        val expectedTerms: List<String>,
        val expectedEdges: List<Edge> = emptyList(),
        val requiredJsonKeys: List<String> = emptyList(),
        val expectedRows: List<List<String>> = emptyList(),
        val expectedObservations: List<List<String>> = emptyList(),
        val prompt: String,
    )

    private data class Edge(val from: String, val condition: String, val to: String)

    @Test(timeout = 30 * 60 * 1_000L)
    fun compareApryseMlKitAndGemmaVisionOnRealPages() {
        val args = InstrumentationRegistry.getArguments()
        val model = ModelPaths.installedLlms(ctx).firstOrNull()
        assumeTrue("Install a multimodal .litertlm model before running Spike 6", model != null)

        val pdf = args.getString("visionPdfPath")?.let(::File)
            ?: File(ctx.filesDir, "docs").listFiles { f -> f.extension.equals("pdf", true) }
                ?.maxByOrNull { it.length() }
        assumeTrue("No app-readable PDF found; pass -e visionPdfPath <path>", pdf?.isFile == true)

        val requestedPages = args.getString("visionPages")
            ?.split(',')
            ?.mapNotNull { it.trim().toIntOrNull() }
            ?.filter { it > 0 }
            ?.distinct()
            .orEmpty()
        val pages = if (requestedPages.isEmpty()) listOf(20, 29, 42) else requestedPages
        val maxEdge = args.getString("visionMaxEdge")?.toIntOrNull()?.coerceIn(768, 2048) ?: 1536
        val cases = pages.map(::pageCase)

        val report = JSONObject()
            .put("model", model!!.name)
            .put("pdf", pdf!!.name)
            .put("pages", JSONArray(pages))
            .put("renderMaxEdge", maxEdge)
            .put("device", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
        val pageResults = JSONArray()
        report.put("results", pageResults)

        val pssBeforeKb = Debug.getPss()
        val cacheDir = File(ctx.cacheDir, "litertlm-vision-spike").apply { mkdirs() }
        val engine = Engine(
            EngineConfig(
                modelPath = model.absolutePath,
                backend = Backend.GPU(),
                visionBackend = Backend.GPU(),
                maxNumTokens = 4096,
                maxNumImages = 1,
                cacheDir = cacheDir.absolutePath,
            ),
        )
        val initStart = SystemClock.elapsedRealtime()
        engine.initialize()
        val initMs = SystemClock.elapsedRealtime() - initStart
        report.put("engineInitMs", initMs).put("pssBeforeKb", pssBeforeKb).put("pssAfterInitKb", Debug.getPss())

        val extractor = AprysePdfExtractor(ctx)
        val ocr = MlKitOcr(ctx)
        try {
            extractor.open(pdf.absolutePath).use { document ->
                val validCases = cases.filter { it.page <= document.pageCount }
                assertTrue("None of the requested pages exist in a ${document.pageCount}-page PDF", validCases.isNotEmpty())

                for (case in validCases) {
                    val original = document.extractPage(case.page)
                    val bitmap = document.renderPage(case.page, maxEdge)
                    try {
                        val ocrStart = SystemClock.elapsedRealtime()
                        val ocrLayout = runBlocking {
                            ocr.recognize(
                                bitmap = bitmap,
                                script = Script.LATIN,
                                pageNumber = case.page,
                                pageWidthPts = original.width,
                                pageHeightPts = original.height,
                            )
                        }
                        val ocrMs = SystemClock.elapsedRealtime() - ocrStart

                        val jpeg = ByteArrayOutputStream().use { bytes ->
                            assertTrue("Could not encode page ${case.page}", bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 92, bytes))
                            bytes.toByteArray()
                        }
                        val conversation = engine.createConversation(
                            ConversationConfig(
                                systemInstruction = Contents.of(
                                    "You analyze document page images conservatively. Report only visible evidence. " +
                                        "Preserve labels, numbers, units, table row associations, and arrow direction exactly. " +
                                    "If something is unreadable, mark it as unreadable instead of guessing. " +
                                        "Return only the JSON object requested by the user, with no Markdown fence or commentary.",
                                ),
                                samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0, seed = 0),
                                maxOutputToken = 512,
                                thinkingConfig = ThinkingConfig(false, 0),
                            ),
                        )
                        val visionStart = SystemClock.elapsedRealtime()
                        val visionText = conversation.use {
                            // Gemma 4's model card recommends image content before its text prompt.
                            it.sendMessage(
                                Contents.of(
                                    Content.ImageBytes(jpeg),
                                    Content.Text(case.prompt),
                                ),
                            ).contents.toString()
                        }
                        val visionMs = SystemClock.elapsedRealtime() - visionStart
                        val structured = extractJsonObject(visionText)

                        val result = JSONObject()
                            .put("page", case.page)
                            .put("kind", case.kind)
                            .put("imageBytes", jpeg.size)
                            .put("apryseChars", original.charCount)
                            .put("apryseExpectedRecall", recall(original.plainText, case.expectedTerms))
                            .put("mlKitChars", ocrLayout.charCount)
                            .put("mlKitMs", ocrMs)
                            .put("mlKitExpectedRecall", recall(ocrLayout.plainText, case.expectedTerms))
                            .put("gemmaVisionMs", visionMs)
                            .put("gemmaExpectedRecall", recall(visionText, case.expectedTerms))
                            .put("gemmaStructuredJson", structured != null)
                            .put("gemmaSchemaRecall", schemaRecall(structured, case.requiredJsonKeys))
                            .put("gemmaEdgeRecall", edgeRecall(structured, case.expectedEdges))
                            .put("gemmaRowRecall", itemRecall(structured, "rows", case.expectedRows))
                            .put("gemmaObservationRecall", itemRecall(structured, "observations", case.expectedObservations))
                            .put("expectedTerms", JSONArray(case.expectedTerms))
                            .put("expectedEdges", JSONArray(case.expectedEdges.map { edgeToJson(it) }))
                            .put("mlKitText", ocrLayout.plainText)
                            .put("gemmaVisionText", visionText)
                        pageResults.put(result)
                        Log.i(
                            TAG,
                            "page=${case.page} kind=${case.kind} MLKit=${ocrMs}ms Gemma=${visionMs}ms " +
                                "recall(A/M/G)=${result.getDouble("apryseExpectedRecall")}/" +
                                "${result.getDouble("mlKitExpectedRecall")}/${result.getDouble("gemmaExpectedRecall")}\n" +
                                "Gemma: $visionText",
                        )
                        assertTrue("Gemma returned no output for page ${case.page}", visionText.isNotBlank())
                    } finally {
                        bitmap.recycle()
                    }
                }
            }
        } finally {
            ocr.close()
            engine.close()
            report.put("pssBeforeCloseKb", Debug.getPss())
                .put("thermalStatusAtEnd", ctx.getSystemService(PowerManager::class.java).currentThermalStatus)
            File(ctx.filesDir, REPORT_FILE).writeText(report.toString(2))
            Log.i(TAG, "report=${File(ctx.filesDir, REPORT_FILE).absolutePath}\n${report.toString(2)}")
        }
    }

    private fun pageCase(page: Int): PageCase = when (page) {
        20 -> PageCase(
            page,
            "image-assisted table",
            listOf("Thin Nitrile", "Thick Nitrile", "Heat-Resistant", "Cryogenic", "2-4 mil"),
            requiredJsonKeys = listOf("rows", "visualAdditions", "uncertain"),
            expectedRows = listOf(
                listOf("Thin Nitrile", "laboratory", "<2 mil"),
                listOf("ThickNitrile", "mechanics", "2-4 mil"),
                listOf("Heat-Resistant", "high-temperature", "varies"),
                listOf("Cryogenic", "cold", "varies"),
            ),
            prompt = "Transcribe the glove table, keeping each glove type, pictured use, use-case text, and thickness in the correct row. " +
                "Return {\"rows\":[{\"gloveType\":\"\",\"picturedUse\":\"\",\"useCase\":\"\",\"thickness\":\"\"}]," +
                "\"visualAdditions\":[\"\"],\"uncertain\":[\"\"]}.",
        )
        29 -> PageCase(
            page,
            "photo montage",
            listOf("electrical", "cord", "socket", "water"),
            requiredJsonKeys = listOf("observations", "uncertain"),
            expectedObservations = listOf(
                listOf("top left", "damaged", "cable"),
                listOf("top center", "tape", "wire"),
                listOf("top right", "multiple", "plug"),
                listOf("bottom left", "cord", "floor"),
                listOf("bottom center", "screwdriver", "socket"),
                listOf("bottom right", "water", "electrical"),
            ),
            prompt = "Describe each safety problem shown in the six photographs from left to right, top row then bottom row. " +
                "Return {\"observations\":[{\"position\":\"\",\"visibleEvidence\":\"\",\"inferredHazard\":\"\"}]," +
                "\"uncertain\":[\"\"]}. Keep visible evidence separate from inference.",
        )
        42 -> PageCase(
            page,
            "flowchart",
            listOf("Raise alarm", "Can you handle it", "Anyone injured", "Evacuate", "follow-up"),
            requiredJsonKeys = listOf("nodes", "edges", "unreadable"),
            expectedEdges = listOf(
                Edge("Emergency/Hazard sighted", "always", "Raise alarm"),
                Edge("Raise alarm", "always", "Can you handle it"),
                Edge("Can you handle it", "yes", "Anyone injured"),
                Edge("Can you handle it", "no", "Evacuate area immediately"),
                Edge("Anyone injured", "yes", "Seek medical assistance and help the injured"),
                Edge("Anyone injured", "no", "Handle the emergency"),
                Edge("Handle the emergency", "always", "Emergency handled successfully"),
                Edge("Emergency handled successfully", "yes", "Inform lab-in-charge and faculty in-charge"),
                Edge("Emergency handled successfully", "no", "Evacuate area immediately"),
                Edge("Seek medical assistance and help the injured", "always", "Inform lab-in-charge and faculty in-charge"),
                Edge("Evacuate area immediately", "always", "Inform lab-in-charge and faculty in-charge"),
                Edge("Inform lab-in-charge and faculty in-charge", "always", "Perform follow-up action"),
            ),
            prompt = "Transcribe this emergency flowchart without converting branches into a linear list. " +
                "Return {\"nodes\":[\"\"],\"edges\":[{\"from\":\"\",\"condition\":\"yes|no|always\",\"to\":\"\"}]," +
                "\"unreadable\":[\"\"]}. Emit one edge for every arrow and preserve every Yes/No label exactly.",
        )
        else -> PageCase(
            page,
            "custom page",
            emptyList(),
            requiredJsonKeys = listOf("transcription", "visualStructure", "interpretation", "uncertain"),
            prompt = "Transcribe the important visible text and describe diagrams, tables, images and their relationships. " +
                "Return {\"transcription\":\"\",\"visualStructure\":\"\",\"interpretation\":\"\",\"uncertain\":[\"\"]}.",
        )
    }

    private fun extractJsonObject(text: String): JSONObject? {
        val start = text.indexOf('{')
        val end = text.lastIndexOf('}')
        if (start < 0 || end <= start) return null
        return runCatching { JSONObject(text.substring(start, end + 1)) }.getOrNull()
    }

    private fun edgeRecall(json: JSONObject?, expected: List<Edge>): Double {
        if (expected.isEmpty()) return 0.0
        val array = json?.optJSONArray("edges") ?: return 0.0
        val actual = (0 until array.length()).mapNotNull { index ->
            val edge = array.optJSONObject(index) ?: return@mapNotNull null
            Edge(
                from = normalize(edge.optString("from")),
                condition = normalize(edge.optString("condition")),
                to = normalize(edge.optString("to")),
            )
        }
        val matches = expected.count { wanted ->
            val from = normalize(wanted.from)
            val condition = normalize(wanted.condition)
            val to = normalize(wanted.to)
            actual.any { found ->
                equivalent(found.from, from) && found.condition == condition && equivalent(found.to, to)
            }
        }
        return matches.toDouble() / expected.size
    }

    private fun schemaRecall(json: JSONObject?, requiredKeys: List<String>): Double {
        if (requiredKeys.isEmpty()) return 0.0
        if (json == null) return 0.0
        return requiredKeys.count(json::has).toDouble() / requiredKeys.size
    }

    /** Each expected item's anchors must occur in one JSON array element, preserving association. */
    private fun itemRecall(json: JSONObject?, arrayName: String, expectedItems: List<List<String>>): Double {
        if (expectedItems.isEmpty()) return 0.0
        val array = json?.optJSONArray(arrayName) ?: return 0.0
        val actual = (0 until array.length()).map { index -> normalize(array.opt(index).toString()) }
        val hits = expectedItems.count { anchors ->
            actual.any { item -> anchors.all { anchor -> item.contains(normalize(anchor)) } }
        }
        return hits.toDouble() / expectedItems.size
    }

    private fun edgeToJson(edge: Edge): JSONObject = JSONObject()
        .put("from", edge.from)
        .put("condition", edge.condition)
        .put("to", edge.to)

    private fun equivalent(left: String, right: String): Boolean =
        left == right || (left.length >= 8 && right.contains(left)) || (right.length >= 8 && left.contains(right))

    private fun normalize(value: String): String =
        value.lowercase(Locale.ROOT).replace(Regex("[^\\p{L}\\p{N}]+"), " ").trim()

    private fun recall(text: String, expected: List<String>): Double {
        if (expected.isEmpty()) return 0.0
        val normalized = normalize(text)
        val hits = expected.count { term ->
            val needle = normalize(term)
            normalized.contains(needle)
        }
        return hits.toDouble() / expected.size
    }

    companion object {
        private const val TAG = "VISION_SPIKE"
        private const val REPORT_FILE = "vision_spike_report.json"
    }
}
