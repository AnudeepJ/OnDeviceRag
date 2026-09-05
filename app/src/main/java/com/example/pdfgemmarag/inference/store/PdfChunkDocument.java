package com.example.pdfgemmarag.inference.store;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.appsearch.annotation.Document;
import androidx.appsearch.app.AppSearchSchema;
import androidx.appsearch.app.EmbeddingVector;
import androidx.appsearch.app.ExperimentalAppSearchApi;

/**
 * One chunk of one PDF in AppSearch LocalStorage.
 *
 * Written in Java so {@code androidx.appsearch:appsearch-compiler} runs as a plain javac annotation
 * processor (kapt cannot be used with AGP's built-in Kotlin). The namespace is the PDF's content
 * hash, which makes per-document search filtering and whole-document deletion a namespace operation.
 */
@OptIn(markerClass = ExperimentalAppSearchApi.class)
@Document(name = PdfChunkDocument.SCHEMA_TYPE)
public class PdfChunkDocument {

    public static final String SCHEMA_TYPE = "PdfChunkDocument";

    /** Content hash of the source PDF. */
    @Document.Namespace
    @NonNull
    public String namespace;

    /** {@code <docHash>:<chunkIndex>} – stable, and what the UI receives as a citation id. */
    @Document.Id
    @NonNull
    public String id;

    @Document.CreationTimestampMillis
    public long creationTimestampMillis;

    /** Chunk text; prefix-indexed with the plain tokenizer so CJK bigrams and Latin words both hit. */
    @Document.StringProperty(
            indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_PREFIXES,
            tokenizerType = AppSearchSchema.StringPropertyConfig.TOKENIZER_TYPE_PLAIN)
    @NonNull
    public String text = "";

    @Document.LongProperty
    public int pageNumber;

    @Document.LongProperty
    public int chunkIndex;

    @Document.BooleanProperty
    public boolean isTable;

    /** Display name of the document, stored (not indexed) for listing. */
    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_NONE)
    @NonNull
    public String docName = "";

    /** Dominant script of the document ("LATIN", "CHINESE", ...). */
    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_NONE)
    @NonNull
    public String script = "";

    /** Total pages of the source PDF (duplicated per chunk; cheap, avoids a second schema). */
    @Document.LongProperty
    public int pageCount;

    /** 512-d, L2-normalised EmbeddingGemma vector, stored 8-bit quantised. */
    @Document.EmbeddingProperty(
            indexingType = AppSearchSchema.EmbeddingPropertyConfig.INDEXING_TYPE_SIMILARITY,
            quantizationType = AppSearchSchema.EmbeddingPropertyConfig.QUANTIZATION_TYPE_8_BIT)
    public EmbeddingVector embedding;
}
