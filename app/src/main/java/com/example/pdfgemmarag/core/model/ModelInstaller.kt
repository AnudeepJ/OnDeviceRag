package com.example.pdfgemmarag.core.model

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/**
 * One streaming pass that SHA-256-verifies a downloaded/picked file while copying it into
 * `filesDir/models/`, then atomically renames it into place. Works for 3.6 GB files on 8 GB devices
 * because nothing is held in memory beyond a 1 MB buffer.
 */
object ModelInstaller {

    class VerificationException(message: String) : Exception(message)

    /**
     * @param expectedSha256 lower-case hex; empty string skips the hash check (local picker path).
     * @param expectedSize expected byte length, or -1 to skip.
     * @param onProgress bytes copied so far.
     */
    fun verifyAndInstall(
        source: File,
        target: File,
        expectedSha256: String,
        expectedSize: Long,
        deleteSource: Boolean,
        onProgress: (Long) -> Unit = {},
    ): String {
        if (!source.isFile || !source.canRead()) throw VerificationException("source file is missing or unreadable: ${source.name}")
        if (source.canonicalFile == target.canonicalFile) throw VerificationException("source and target must be different files")
        if (expectedSize > 0 && source.length() != expectedSize) {
            throw VerificationException("size mismatch: expected $expectedSize, got ${source.length()}")
        }
        target.parentFile?.mkdirs()
        val tmp = File(target.parentFile, target.name + ".installing")
        // A killed process must not make a stale partial file look like an installed model.
        if (tmp.exists() && !tmp.delete()) {
            throw VerificationException("could not clear stale install file: ${tmp.name}")
        }
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1 shl 20)
        var copied = 0L
        var committed = false
        try {
            FileInputStream(source).use { input ->
                FileOutputStream(tmp).use { out ->
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        digest.update(buffer, 0, n)
                        out.write(buffer, 0, n)
                        copied += n
                        onProgress(copied)
                    }
                    out.fd.sync()
                }
            }
            if (copied != source.length()) {
                throw VerificationException("copy was incomplete: expected ${source.length()}, got $copied")
            }
            val actual = digest.digest().joinToString("") { "%02x".format(it) }
            if (expectedSha256.isNotBlank() && !actual.equals(expectedSha256, ignoreCase = true)) {
                throw VerificationException("SHA-256 mismatch: expected $expectedSha256, got $actual")
            }
            try {
                Files.move(
                    tmp.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
            committed = true
            if (deleteSource && !source.delete()) {
                // Installation succeeded. A source cleanup failure is non-fatal and can be retried.
            }
            return actual
        } finally {
            if (!committed) tmp.delete()
        }
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1 shl 20)
        FileInputStream(file).use { input ->
            while (true) { val n = input.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
