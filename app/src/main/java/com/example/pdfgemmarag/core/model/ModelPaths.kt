package com.example.pdfgemmarag.core.model

import android.content.Context
import java.io.File

/** Canonical on-disk locations shared by both processes (all under app-internal storage). */
object ModelPaths {
    const val EMBEDDING_MODEL_FILE = "embeddinggemma-300m-seq512.tflite"
    const val EMBEDDING_TOKENIZER_FILE = "embeddinggemma-tokenizer.model"
    const val LLM_EXTENSION = ".litertlm"

    fun modelsDir(context: Context): File = File(context.filesDir, "models").apply { mkdirs() }
    fun docsDir(context: Context): File = File(context.filesDir, "docs").apply { mkdirs() }
    fun stagingDir(context: Context): File = File(context.filesDir, "staging").apply { mkdirs() }

    fun embeddingModel(context: Context): File = File(modelsDir(context), EMBEDDING_MODEL_FILE)
    fun embeddingTokenizer(context: Context): File = File(modelsDir(context), EMBEDDING_TOKENIZER_FILE)
    fun pdfFile(context: Context, docHash: String): File = File(docsDir(context), "$docHash.pdf")

    fun installedLlms(context: Context): List<File> =
        modelsDir(context).listFiles { f -> f.isFile && f.name.endsWith(LLM_EXTENSION) }?.sortedBy { it.name } ?: emptyList()

    fun embeddingReady(context: Context): Boolean =
        embeddingModel(context).let { it.exists() && it.length() > 1_000_000 } &&
            embeddingTokenizer(context).let { it.exists() && it.length() > 100_000 }
}
