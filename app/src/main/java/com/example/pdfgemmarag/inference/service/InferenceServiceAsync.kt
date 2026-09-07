package com.example.pdfgemmarag.inference.service

import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.DocumentInfo
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeout
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Coroutine adapters for result-bearing AIDL calls that execute asynchronously in the service. */
suspend fun IAiInferenceService.listDocumentsAsync(): List<DocumentInfo> =
    withTimeout(QUICK_CALL_TIMEOUT_MS) {
        suspendCancellableCoroutine { continuation ->
            try {
                listDocuments(object : IDocumentsCallback.Stub() {
                    override fun onResult(documents: List<DocumentInfo>) {
                        if (continuation.isActive) continuation.resume(documents)
                    }

                    override fun onError(message: String) {
                        if (continuation.isActive) continuation.resumeWithException(ServiceCallException(message))
                    }
                })
            } catch (error: Throwable) {
                if (continuation.isActive) continuation.resumeWithException(error)
            }
        }
    }

suspend fun IAiInferenceService.getCitationAsync(chunkId: String): Citation? =
    withTimeout(QUICK_CALL_TIMEOUT_MS) { awaitCitation { callback -> getCitation(chunkId, callback) } }

suspend fun IAiInferenceService.getCitationInNamespaceAsync(
    indexNamespace: String,
    chunkId: String,
): Citation? = withTimeout(QUICK_CALL_TIMEOUT_MS) {
    awaitCitation { callback -> getCitationInNamespace(indexNamespace, chunkId, callback) }
}

suspend fun IAiInferenceService.diagnosticsAsync(): String =
    withTimeout(QUICK_CALL_TIMEOUT_MS) { awaitText { callback -> getDiagnostics(callback) } }

suspend fun IAiInferenceService.runSelfTestAsync(): String =
    withTimeout(LONG_CALL_TIMEOUT_MS) { awaitText { callback -> runSelfTest(callback) } }

suspend fun IAiInferenceService.probeRetrievalAsync(docHash: String): String =
    withTimeout(LONG_CALL_TIMEOUT_MS) { awaitText { callback -> probeRetrieval(docHash, callback) } }

private suspend fun awaitCitation(call: (ICitationCallback) -> Unit): Citation? =
    suspendCancellableCoroutine { continuation ->
        val callback = object : ICitationCallback.Stub() {
            override fun onResult(citation: Citation) {
                if (continuation.isActive) continuation.resume(citation)
            }

            override fun onNotFound() {
                if (continuation.isActive) continuation.resume(null)
            }

            override fun onError(message: String) {
                if (continuation.isActive) continuation.resumeWithException(ServiceCallException(message))
            }
        }
        try {
            call(callback)
        } catch (error: Throwable) {
            if (continuation.isActive) continuation.resumeWithException(error)
        }
    }

private suspend fun awaitText(call: (ITextResultCallback) -> Unit): String =
    suspendCancellableCoroutine { continuation ->
        val callback = object : ITextResultCallback.Stub() {
            override fun onResult(value: String) {
                if (continuation.isActive) continuation.resume(value)
            }

            override fun onError(message: String) {
                if (continuation.isActive) continuation.resumeWithException(ServiceCallException(message))
            }
        }
        try {
            call(callback)
        } catch (error: Throwable) {
            if (continuation.isActive) continuation.resumeWithException(error)
        }
    }

private class ServiceCallException(message: String) : IllegalStateException(message)

private const val QUICK_CALL_TIMEOUT_MS = 30_000L
private const val LONG_CALL_TIMEOUT_MS = 180_000L
