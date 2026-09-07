package com.example.pdfgemmarag.inference.store

import android.content.Context
import android.util.Log
import com.example.pdfgemmarag.inference.chunk.Chunk
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

data class SectionRecord(
    val sectionId: String,
    val specificationNumber: String,
    val sectionNumber: String,
    val title: String,
    val path: String,
    val startPage: Int,
    val endPage: Int,
    val orderedChunkIds: List<String>,
    val tokenCount: Int,
    val centroid: FloatArray? = null,
)

data class DocumentStructureManifest(
    val documentHash: String,
    val indexNamespace: String,
    val indexVersion: Int,
    val embeddingSignature: String,
    val sections: List<SectionRecord>,
) {
    val chunksInOrder: List<String> get() = sections.flatMap { it.orderedChunkIds }.distinct()

    fun validate() {
        require(documentHash.isNotBlank()) { "manifest document hash is blank" }
        require(indexNamespace.isNotBlank()) { "manifest namespace is blank" }
        require(indexVersion == INDEX_VERSION) { "unsupported manifest version $indexVersion" }
        val ids = chunksInOrder
        require(ids.size == sections.sumOf { it.orderedChunkIds.size }) { "manifest contains duplicate chunk IDs" }
        require(sections.map { it.sectionId }.distinct().size == sections.size) { "manifest contains duplicate section IDs" }
        sections.forEach { require(it.startPage in 1..it.endPage) { "invalid page span for ${it.sectionId}" } }
    }

    companion object {
        const val INDEX_VERSION = 21

        fun fromChunks(
            documentHash: String,
            namespace: String,
            signature: String,
            chunks: List<Chunk>,
            tokenCount: (String) -> Int,
            centroids: Map<String, FloatArray> = emptyMap(),
        ): DocumentStructureManifest {
            val sections = chunks.groupBy { it.sectionId }.values.map { group ->
                val first = group.first()
                SectionRecord(
                    sectionId = first.sectionId,
                    specificationNumber = first.specificationNumber,
                    sectionNumber = first.sectionNumber,
                    title = first.sectionTitle,
                    path = first.sectionPath,
                    startPage = group.minOf { it.pageNumber },
                    endPage = group.maxOf { it.pageNumber },
                    orderedChunkIds = group.sortedBy { it.chunkIndex }.map { chunkId(it.chunkIndex) },
                    tokenCount = group.sumOf { tokenCount(it.retrievalText) },
                    centroid = centroids[first.sectionId],
                )
            }.sortedWith(compareBy<SectionRecord> { it.startPage }.thenBy { it.orderedChunkIds.firstOrNull() })
            return DocumentStructureManifest(documentHash, namespace, INDEX_VERSION, signature, sections).also { it.validate() }
        }

        fun chunkId(index: Int): String = "c" + index.toString().padStart(7, '0')
    }
}

class ManifestIntegrityError(message: String, cause: Throwable? = null) : IllegalStateException(message, cause)

/** Small, atomically published per-document manifest. */
class DocumentStructureManifestStore(context: Context) {
    private val directory = File(context.filesDir, "rag_manifests").also { it.mkdirs() }

    fun load(documentHash: String): DocumentStructureManifest? {
        val file = file(documentHash)
        if (!file.exists()) {
            Log.w(
                TAG,
                "manifest missing hash='$documentHash' length=${documentHash.length} path=${file.absolutePath} " +
                    "available=${directory.list().orEmpty().joinToString()}",
            )
            return null
        }
        return try {
            decode(JSONObject(file.readText())).also { manifest ->
                manifest.validate()
                if (manifest.documentHash != documentHash) throw ManifestIntegrityError("manifest hash mismatch")
            }
        } catch (t: Throwable) {
            if (t is ManifestIntegrityError) throw t
            throw ManifestIntegrityError("Could not read structure manifest", t)
        }
    }

    fun publish(manifest: DocumentStructureManifest) {
        manifest.validate()
        val target = file(manifest.documentHash)
        val staging = File(directory, target.name + ".tmp")
        FileOutputStream(staging).use { output ->
            output.write(encode(manifest).toString().toByteArray())
            output.fd.sync()
        }
        if (!staging.renameTo(target)) {
            staging.delete()
            throw ManifestIntegrityError("Could not atomically publish structure manifest")
        }
    }

    fun delete(documentHash: String) {
        file(documentHash).delete()
        File(directory, "$documentHash.json.tmp").delete()
    }

    fun list(): List<DocumentStructureManifest> = directory.listFiles()
        .orEmpty()
        .filter { it.extension == "json" }
        .mapNotNull { runCatching { load(it.nameWithoutExtension) }.getOrNull() }

    private fun file(hash: String) = File(directory, "$hash.json")

    private companion object { const val TAG = "StructureManifest" }

    private fun encode(manifest: DocumentStructureManifest) = JSONObject().apply {
        put("documentHash", manifest.documentHash)
        put("indexNamespace", manifest.indexNamespace)
        put("indexVersion", manifest.indexVersion)
        put("embeddingSignature", manifest.embeddingSignature)
        put("sections", JSONArray().apply {
            manifest.sections.forEach { section ->
                put(JSONObject().apply {
                    put("sectionId", section.sectionId)
                    put("specificationNumber", section.specificationNumber)
                    put("sectionNumber", section.sectionNumber)
                    put("title", section.title)
                    put("path", section.path)
                    put("startPage", section.startPage)
                    put("endPage", section.endPage)
                    put("orderedChunkIds", JSONArray(section.orderedChunkIds))
                    put("tokenCount", section.tokenCount)
                    section.centroid?.let { values -> put("centroid", JSONArray(values.toList())) }
                })
            }
        })
    }

    private fun decode(json: JSONObject): DocumentStructureManifest {
        val sectionsJson = json.getJSONArray("sections")
        val sections = (0 until sectionsJson.length()).map { i ->
            val section = sectionsJson.getJSONObject(i)
            val ids = section.getJSONArray("orderedChunkIds")
            val centroidJson = section.optJSONArray("centroid")
            SectionRecord(
                sectionId = section.getString("sectionId"),
                specificationNumber = section.optString("specificationNumber"),
                sectionNumber = section.optString("sectionNumber"),
                title = section.optString("title"),
                path = section.optString("path"),
                startPage = section.getInt("startPage"),
                endPage = section.getInt("endPage"),
                orderedChunkIds = (0 until ids.length()).map { ids.getString(it) },
                tokenCount = section.optInt("tokenCount"),
                centroid = centroidJson?.let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } },
            )
        }
        return DocumentStructureManifest(
            documentHash = json.getString("documentHash"),
            indexNamespace = json.getString("indexNamespace"),
            indexVersion = json.getInt("indexVersion"),
            embeddingSignature = json.getString("embeddingSignature"),
            sections = sections,
        )
    }
}
