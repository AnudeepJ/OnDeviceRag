package com.example.pdfgemmarag.inference.embed

import android.content.Context
import android.util.Log
import com.example.pdfgemmarag.core.model.GpuMarker
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.io.Closeable
import java.io.File
import kotlin.math.sqrt

/**
 * EmbeddingGemma-300M on LiteRT [CompiledModel].
 *
 * Input: token ids padded to the model's fixed sequence length (`seq512` variant). Output: 768-d
 * pooled embedding, truncated (Matryoshka) to [outputDim] and L2-normalised so cosine == dot.
 * Chunks use the `title: none | text: ` prefix, queries `task: search result | query: `; using the
 * wrong prefix costs several points of retrieval quality.
 */
class EmbeddingGemmaEmbedder(
    context: Context,
    modelFile: File,
    tokenizerFile: File,
    preferGpu: Boolean,
    val outputDim: Int = 512,
) : Closeable {

    val tokenizer: SentencePieceTokenizer = SentencePieceTokenizer.load(tokenizerFile)
    private val environment: Environment = Environment.create(context)
    private val model: CompiledModel
    val backend: String
    val sequenceLength: Int
    private val inputIsInt64: Boolean
    private val inputCount: Int
    private val inputs: List<TensorBuffer>
    private val outputs: List<TensorBuffer>
    private val lock = Any()

    init {
        var m: CompiledModel? = null
        var used = "CPU"
        if (preferGpu) {
            try {
                m = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(Accelerator.GPU), environment)
                used = "GPU"
            } catch (t: Throwable) {
                Log.w(TAG, "GPU compile failed, falling back to CPU: ${t.message}")
                GpuMarker.writeEmbedder(context, "compile failed: ${t.message}")
            }
        }
        if (m == null) {
            val opts = CompiledModel.Options(Accelerator.CPU).apply {
                cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
            }
            m = CompiledModel.create(modelFile.absolutePath, opts, environment)
        }
        model = m
        backend = used
        // Input tensor introspection needs the signature input name; the litert-community export uses
        // "input_ids". If the name differs we fall back to int32 x seq512 (the documented layout).
        val inType: TensorType? = INPUT_NAMES.firstNotNullOfOrNull { n -> runCatching { model.getInputTensorType(n) }.getOrNull() }
        inputIsInt64 = inType?.elementType == TensorType.ElementType.INT64
        sequenceLength = inType?.layout?.dimensions?.lastOrNull()?.takeIf { it > 0 }
            ?: seqLenFromFileName(modelFile.name) ?: DEFAULT_SEQ_LEN
        inputs = model.createInputBuffers()
        outputs = model.createOutputBuffers()
        inputCount = inputs.size
        Log.i(TAG, "EmbeddingGemma ready: backend=$backend seq=$sequenceLength inputs=$inputCount int64=$inputIsInt64 " +
            "vocab=${tokenizer.vocabSize} (${tokenizer.normalizerNote})")
    }

    fun embedDocument(text: String): FloatArray = embed(DOC_PREFIX + text)

    fun embedQuery(text: String): FloatArray = embed(QUERY_PREFIX + text)

    /** Token count of the prefixed chunk; used to verify chunks fit the window. */
    fun tokenCount(text: String): Int = tokenizer.encode(DOC_PREFIX + text).size + 2

    private fun embed(text: String): FloatArray = synchronized(lock) {
        val ids = tokenizer.encodeForModel(text, sequenceLength, addBos = true, addEos = true)
        val padded = IntArray(sequenceLength) { tokenizer.padId.coerceAtLeast(0) }
        System.arraycopy(ids, 0, padded, 0, ids.size)
        if (inputIsInt64) inputs[0].writeLong(LongArray(sequenceLength) { padded[it].toLong() }) else inputs[0].writeInt(padded)
        if (inputCount > 1) {
            // Second input is the attention mask on multi-input exports.
            val mask = IntArray(sequenceLength) { if (it < ids.size) 1 else 0 }
            if (inputIsInt64) inputs[1].writeLong(LongArray(sequenceLength) { mask[it].toLong() }) else inputs[1].writeInt(mask)
        }
        model.run(inputs, outputs)
        val raw = outputs[0].readFloat()
        val dim = minOf(outputDim, raw.size)
        val out = FloatArray(dim)
        System.arraycopy(raw, 0, out, 0, dim)
        l2Normalize(out)
        out
    }

    override fun close() {
        try { inputs.forEach { it.close() }; outputs.forEach { it.close() } } catch (_: Throwable) {}
        try { model.close() } catch (_: Throwable) {}
        try { environment.close() } catch (_: Throwable) {}
    }

    companion object {
        private const val TAG = "EmbeddingGemma"
        const val DOC_PREFIX = "title: none | text: "
        const val QUERY_PREFIX = "task: search result | query: "
        const val DEFAULT_SEQ_LEN = 512
        const val MODEL_SIGNATURE = "embeddinggemma-300m-seq512-512d"
        private val INPUT_NAMES = listOf("input_ids", "input_word_ids", "serving_default_input_ids:0", "args_0")

        fun seqLenFromFileName(name: String): Int? =
            Regex("seq(\\d+)").find(name)?.groupValues?.get(1)?.toIntOrNull()

        fun l2Normalize(v: FloatArray) {
            var sum = 0.0
            for (x in v) sum += x * x
            val norm = sqrt(sum).toFloat()
            if (norm > 1e-6f) for (i in v.indices) v[i] /= norm
        }

        fun cosine(a: FloatArray, b: FloatArray): Float {
            var dot = 0f
            for (i in a.indices) dot += a[i] * b[i]
            return dot
        }
    }
}
