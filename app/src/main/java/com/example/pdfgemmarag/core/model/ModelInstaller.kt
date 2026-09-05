package com.example.pdfgemmarag.core.model

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
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
        val tmp = File(target.parentFile, target.name + ".part")
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1 shl 20)
        var copied = 0L
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
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        if (expectedSha256.isNotBlank() && !actual.equals(expectedSha256, ignoreCase = true)) {
            tmp.delete()
            throw VerificationException("SHA-256 mismatch: expected $expectedSha256, got $actual")
        }
        if (target.exists()) target.delete()
        if (!tmp.renameTo(target)) {
            tmp.delete()
            throw VerificationException("could not move ${tmp.name} into place")
        }
        if (deleteSource) source.delete()
        return actual
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
