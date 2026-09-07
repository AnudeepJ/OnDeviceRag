package com.example.pdfgemmarag.inference.service;

import com.example.pdfgemmarag.core.model.DocumentInfo;

oneway interface IDocumentsCallback {
    void onResult(in List<DocumentInfo> documents);
    void onError(String message);
}
