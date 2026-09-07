package com.example.pdfgemmarag.inference.pdf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test

class TextNormalizerTest {

    @Test
    fun `kangxi radicals and radicals supplement fold to unified ideographs`() {
        // Seen on device: 日 as U+2F47 (Kangxi), 民 as U+2EA0 (Radicals Supplement), 人 as U+2F08.
        assertEquals("日本の首都", TextNormalizer.normalize("\u2F47本の\u2FB8都"))
        assertEquals("中华人民共和国", TextNormalizer.normalize("中华\u2F08\u2EA0共和国"))
    }

    @Test
    fun `supplementary unified ideographs are not truncated to UTF-16`() {
        // U+2E87 maps to U+20628. Truncating it to a Char produced U+0628 (Arabic BEH).
        assertEquals("\uD841\uDE28", TextNormalizer.normalize("\u2E87"))
    }

    @Test
    fun `fullwidth ascii and ligatures are folded`() {
        assertEquals("ABC 123 fi fl", TextNormalizer.normalize("ＡＢＣ　１２３ ﬁ ﬂ"))
    }

    @Test
    fun `ascii input is returned as-is`() {
        val s = "plain ascii text 123"
        assertSame(s, TextNormalizer.normalize(s))
    }

    @Test
    fun `korean and ordinary cjk are untouched`() {
        assertEquals("서울은 대한민국의 수도입니다.", TextNormalizer.normalize("서울은 대한민국의 수도입니다."))
        assertEquals("東京都は日本の首都です。", TextNormalizer.normalize("東京都は日本の首都です。"))
    }
}
