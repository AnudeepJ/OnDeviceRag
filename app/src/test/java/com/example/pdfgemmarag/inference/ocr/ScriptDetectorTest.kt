package com.example.pdfgemmarag.inference.ocr

import org.junit.Assert.assertEquals
import org.junit.Test

class ScriptDetectorTest {
    @Test fun latin() = assertEquals(Script.LATIN, ScriptDetector.detect("The quick brown fox jumps over the lazy dog."))
    @Test fun japanese() = assertEquals(Script.JAPANESE, ScriptDetector.detect("東京都は日本の首都です。"))
    @Test fun chinese() = assertEquals(Script.CHINESE, ScriptDetector.detect("机器学习是人工智能的一个分支。"))
    @Test fun korean() = assertEquals(Script.KOREAN, ScriptDetector.detect("안녕하세요, 만나서 반갑습니다."))
    @Test fun `mostly english with a few cjk chars stays latin`() =
        assertEquals(Script.LATIN, ScriptDetector.detect("The company (株式会社) reported strong results across all regions this year."))
    @Test fun `empty and digits are latin`() {
        assertEquals(Script.LATIN, ScriptDetector.detect(""))
        assertEquals(Script.LATIN, ScriptDetector.detect("12345 67890"))
    }
}
