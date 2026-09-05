package com.example.pdfgemmarag.inference.service;

oneway interface IInstallCallback {
    void onProgress(long bytesCopied, long totalBytes);
    void onInstalled(String targetPath, String sha256);
    void onFailed(String message);
}
