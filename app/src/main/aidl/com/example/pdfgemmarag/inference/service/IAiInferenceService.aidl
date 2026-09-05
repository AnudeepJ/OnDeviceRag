package com.example.pdfgemmarag.inference.service;

import com.example.pdfgemmarag.core.model.Citation;
import com.example.pdfgemmarag.core.model.EngineStatus;
import com.example.pdfgemmarag.core.model.DocumentInfo;
import com.example.pdfgemmarag.core.model.QaPair;
import com.example.pdfgemmarag.inference.service.IStreamCallback;
import com.example.pdfgemmarag.inference.service.IIndexingCallback;
import com.example.pdfgemmarag.inference.service.IEngineCallback;
import com.example.pdfgemmarag.inference.service.IInstallCallback;

interface IAiInferenceService {
    int ping();

    // ---- Engine lifecycle ----
    void loadEngine(String modelPath, boolean allowGpu, IEngineCallback callback);
    void unloadEngine();
    EngineStatus getEngineStatus();

    // ---- Generation ----
    /** Starts a RAG turn; returns the generationId echoed in every callback. */
    long ask(String docHash, String question, in List<QaPair> history, IStreamCallback callback);
    void cancelGeneration(long generationId);

    // ---- Ingestion ----
    void indexDocument(String docHash, String pdfPath, String displayName, IIndexingCallback callback);
    void cancelIndexing();
    boolean isIndexing();
    void deleteDocument(String docHash);
    List<DocumentInfo> listDocuments();

    // ---- Citations ----
    Citation getCitation(String chunkId);

    // ---- Model install (SHA-256 verify + copy into filesDir/models) ----
    void installModel(String sourcePath, String targetFileName, String expectedSha256, long expectedSize, boolean deleteSource, IInstallCallback callback);

    // ---- Diagnostics (feature support, backend, versions) ----
    String getDiagnostics();

    /** Runs the Phase 0 spike checks on-device (tokenizer, embedder, AppSearch hybrid) and returns a report. */
    String runSelfTest();
}
