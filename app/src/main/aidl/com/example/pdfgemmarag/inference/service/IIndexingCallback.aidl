package com.example.pdfgemmarag.inference.service;

import com.example.pdfgemmarag.core.model.IndexingProgress;
import com.example.pdfgemmarag.core.model.DocumentInfo;

oneway interface IIndexingCallback {
    void onProgress(in IndexingProgress progress);
    void onCompleted(in DocumentInfo document);
    void onCancelled(String docHash);
    void onFailed(String docHash, String message);
}
