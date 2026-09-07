package com.example.pdfgemmarag.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.security.MessageDigest

class ModelInstallerTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun sha(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    @Test
    fun `verifies hash copies and deletes source`() {
        val payload = ByteArray(3 * 1024 * 1024 + 17) { (it % 251).toByte() }
        val src = tmp.newFile("dl.bin").apply { writeBytes(payload) }
        val target = tmp.root.resolve("models/model.litertlm")
        var progressCalls = 0
        val got = ModelInstaller.verifyAndInstall(src, target, sha(payload), payload.size.toLong(), deleteSource = true) { progressCalls++ }
        assertEquals(sha(payload), got)
        assertTrue(target.exists()); assertEquals(payload.size.toLong(), target.length())
        assertFalse(src.exists())
        assertTrue(progressCalls >= 3)
        assertFalse(target.parentFile!!.resolve("model.litertlm.installing").exists())
    }

    @Test
    fun `hash mismatch leaves no target and keeps source`() {
        val src = tmp.newFile("dl.bin").apply { writeBytes(ByteArray(1024) { 1 }) }
        val target = tmp.root.resolve("models/model.litertlm")
        try {
            ModelInstaller.verifyAndInstall(src, target, "00".repeat(32), -1, deleteSource = true)
            fail("expected VerificationException")
        } catch (e: ModelInstaller.VerificationException) {
            assertTrue(e.message!!.contains("SHA-256"))
        }
        assertFalse(target.exists()); assertTrue(src.exists())
    }

    @Test
    fun `size mismatch is rejected before copying`() {
        val src = tmp.newFile("dl.bin").apply { writeBytes(ByteArray(10)) }
        try {
            ModelInstaller.verifyAndInstall(src, tmp.root.resolve("m"), "", 11, deleteSource = false)
            fail()
        } catch (e: ModelInstaller.VerificationException) { assertTrue(e.message!!.contains("size")) }
    }

    @Test
    fun `verification failure preserves an existing installed model and removes stale partial`() {
        val src = tmp.newFile("replacement.bin").apply { writeBytes(ByteArray(1024) { 2 }) }
        val directory = tmp.newFolder("installed")
        val target = directory.resolve("model.litertlm").apply { writeText("known-good") }
        directory.resolve("model.litertlm.installing").writeText("stale")

        try {
            ModelInstaller.verifyAndInstall(src, target, "00".repeat(32), -1, deleteSource = true)
            fail("expected VerificationException")
        } catch (_: ModelInstaller.VerificationException) {
            // Expected.
        }

        assertEquals("known-good", target.readText())
        assertTrue(src.exists())
        assertFalse(directory.resolve("model.litertlm.installing").exists())
    }
}
