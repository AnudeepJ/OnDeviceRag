package com.example.pdfgemmarag.inference.service;

import com.example.pdfgemmarag.core.model.Citation;
import com.example.pdfgemmarag.core.model.GenerationStats;

/**
 * Token stream for one generation. Every call carries the generationId so the UI can drop
 * stragglers that arrive after cancelProcess() (which is asynchronous in LiteRT-LM).
 */
oneway interface IStreamCallback {
    void onRetrieved(long generationId, in List<Citation> citations);
    void onToken(long generationId, String token);
    void onDone(long generationId, in GenerationStats stats);
    void onError(long generationId, String message);
}
