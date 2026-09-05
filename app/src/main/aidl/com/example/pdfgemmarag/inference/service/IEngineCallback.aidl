package com.example.pdfgemmarag.inference.service;

import com.example.pdfgemmarag.core.model.EngineStatus;

oneway interface IEngineCallback {
    /** Fired repeatedly during initialisation so the UI watchdog can reset its timer. */
    void onProgress(String stage);
    void onReady(in EngineStatus status);
    void onFailed(String message);
}
