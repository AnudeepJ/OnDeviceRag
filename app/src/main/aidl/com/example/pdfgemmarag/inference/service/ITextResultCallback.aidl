package com.example.pdfgemmarag.inference.service;

oneway interface ITextResultCallback {
    void onResult(String value);
    void onError(String message);
}
