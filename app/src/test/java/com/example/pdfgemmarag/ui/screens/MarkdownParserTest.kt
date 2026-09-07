package com.example.pdfgemmarag.ui.screens

import org.junit.Assert.assertEquals
import org.junit.Test

class MarkdownParserTest {
    @Test
    fun `parses common model response blocks`() {
        assertEquals(
            listOf(
                MarkdownBlock.Heading(2, "Summary"),
                MarkdownBlock.Bullet("First point"),
                MarkdownBlock.Numbered("2", "Second point"),
                MarkdownBlock.Quote("Source note"),
            ),
            MarkdownParser.blocks("## Summary\n- First point\n2. Second point\n> Source note"),
        )
    }

    @Test
    fun `parses fenced code without showing fence markers`() {
        assertEquals(
            listOf(MarkdownBlock.Code("val answer = 42")),
            MarkdownParser.blocks("```kotlin\nval answer = 42\n```"),
        )
    }
}
