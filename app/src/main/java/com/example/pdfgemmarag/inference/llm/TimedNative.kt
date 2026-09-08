package com.example.pdfgemmarag.inference.llm

import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/**
 * Runs a native LiteRT call on a dedicated thread so a hang cannot block [GemmaEngine]'s closer.
 * Returns false if [timeoutSec] elapses. JNI may ignore the worker interrupt, so a false result is
 * fatal for the owning native runtime: callers must quarantine it and recycle the process instead
 * of invoking another method on the same conversation or engine.
 */
internal object TimedNative {
    fun run(timeoutSec: Long, label: String, block: () -> Unit): Boolean {
        require(timeoutSec > 0) { "timeoutSec must be positive ($label)" }
        val exec = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "gemma-timed-$label").apply { isDaemon = true }
        }
        return try {
            exec.submit(block).get(timeoutSec, TimeUnit.SECONDS)
            true
        } catch (_: TimeoutException) {
            false
        } catch (error: ExecutionException) {
            throw error.cause ?: error
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
            false
        } finally {
            exec.shutdownNow()
        }
    }
}
