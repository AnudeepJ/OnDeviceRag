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
 * versioned index namespace. The stable PDF content hash is stored separately as {@link #docHash}.
 */
@OptIn(markerClass = ExperimentalAppSearchApi.class)
@Document(name = PdfChunkDocument.SCHEMA_TYPE)
public class PdfChunkDocument {

    public static final String SCHEMA_TYPE = "PdfChunkDocument";

    /** Active or staging namespace for one complete index build. */
    @Document.Namespace
    @NonNull
    public String namespace;

    /** Stable inside the explicit namespace; citations must carry both values. */
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

    /** Original source text displayed in the citation sheet. */
    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_NONE)
    @NonNull
    public String bodyText = "";

    /** Section path + source text used for lexical and semantic retrieval. */
    @Document.StringProperty(
            indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_PREFIXES,
            tokenizerType = AppSearchSchema.StringPropertyConfig.TOKENIZER_TYPE_PLAIN)
    @NonNull
    public String retrievalText = "";

    /** Stable content hash of the source document. */
    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_NONE)
    @NonNull
    public String docHash = "";

    @Document.StringProperty(
            indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_EXACT_TERMS,
            tokenizerType = AppSearchSchema.StringPropertyConfig.TOKENIZER_TYPE_VERBATIM)
    @NonNull
    public String sectionId = "";

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_PREFIXES)
    @NonNull
    public String sectionTitle = "";

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_NONE)
    @NonNull
    public String sectionPath = "";

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_PREFIXES)
    @NonNull
    public String specificationNumber = "";

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_PREFIXES)
    @NonNull
    public String sectionNumber = "";

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_PREFIXES)
    @NonNull
    public String identifierAtoms = "";

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_NONE)
    @NonNull
    public String contentKind = "PARAGRAPH";

    @Document.LongProperty
    public int positionInSection;

    @Document.LongProperty
    public int continuesFromChunkIndex = -1;

    @Document.LongProperty
    public int continuesToChunkIndex = -1;

    @Document.LongProperty
    public int indexVersion;

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_NONE)
    @NonNull
    public String embeddingSignature = "";

    @Document.LongProperty
    public int pageNumber;

    @Document.LongProperty
    public int chunkIndex;

    @Document.BooleanProperty
    public boolean isTable;

    /** Stable table identity; exact/verbatim so an explicit table fetch does not collide with a section id. */
    @Document.StringProperty(
            indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_EXACT_TERMS,
            tokenizerType = AppSearchSchema.StringPropertyConfig.TOKENIZER_TYPE_VERBATIM)
    @NonNull
    public String tableId = "";

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_PREFIXES)
    @NonNull
    public String tableNumber = "";

    @Document.StringProperty(indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_PREFIXES)
    @NonNull
    public String tableCaption = "";

    /** Logical list identity and source completeness metadata. */
    @Document.StringProperty(
            indexingType = AppSearchSchema.StringPropertyConfig.INDEXING_TYPE_EXACT_TERMS,
            tokenizerType = AppSearchSchema.StringPropertyConfig.TOKENIZER_TYPE_VERBATIM)
    @NonNull
    public String listId = "";

    @Document.LongProperty
    public int listItemStart;

    @Document.LongProperty
    public int listItemCount;

    @Document.LongProperty
    public int listTotalItems;

    @Document.BooleanProperty
    public boolean listComplete;

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
