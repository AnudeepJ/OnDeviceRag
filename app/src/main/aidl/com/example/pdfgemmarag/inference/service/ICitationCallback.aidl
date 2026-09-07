package com.example.pdfgemmarag.inference.service;

import com.example.pdfgemmarag.core.model.Citation;

oneway interface ICitationCallback {
    void onResult(in Citation citation);
    void onNotFound();
    void onError(String message);
}
