package com.example.pdfgemmarag.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle

/**
 * A deliberately small, safe Markdown renderer for model output.
 *
 * It does not interpret HTML and has no external dependency.  Keeping the parsed representation
 * separate also makes it usable for a partially streamed response: an unfinished marker simply
 * remains visible as ordinary text until its closing marker arrives.
 */
@Composable
internal fun MarkdownText(markdown: String, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxWidth()) {
        MarkdownParser.blocks(markdown).forEach { block ->
            when (block) {
                is MarkdownBlock.Blank -> Text("")
                is MarkdownBlock.Heading -> Text(
                    inlineMarkdown(block.text),
                    style = if (block.level <= 2) MaterialTheme.typography.titleMedium else MaterialTheme.typography.titleSmall,
                )
                is MarkdownBlock.Bullet -> Text(
                    buildAnnotatedString { append("• "); append(inlineMarkdown(block.text)) },
                    style = MaterialTheme.typography.bodyMedium,
                )
                is MarkdownBlock.Numbered -> Text(
                    buildAnnotatedString { append("${block.number}. "); append(inlineMarkdown(block.text)) },
                    style = MaterialTheme.typography.bodyMedium,
                )
                is MarkdownBlock.Quote -> Text(
                    buildAnnotatedString { withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { append("│ "); append(inlineMarkdown(block.text)) } },
                    style = MaterialTheme.typography.bodyMedium,
                )
                is MarkdownBlock.Code -> Text(block.text, style = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace))
                is MarkdownBlock.Paragraph -> Text(inlineMarkdown(block.text), style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

private fun inlineMarkdown(text: String): AnnotatedString = buildAnnotatedString {
    var cursor = 0
    while (cursor < text.length) {
        val token = listOf("**", "__", "`", "*", "_")
            .map { it to text.indexOf(it, cursor) }
            .filter { it.second >= 0 }
            .minByOrNull { it.second }
        if (token == null) {
            append(text.substring(cursor))
            break
        }
        val (marker, start) = token
        append(text.substring(cursor, start))
        val end = text.indexOf(marker, start + marker.length)
        if (end < 0) {
            append(marker)
            cursor = start + marker.length
            continue
        }
        val content = text.substring(start + marker.length, end)
        val style = when (marker) {
            "**", "__" -> SpanStyle(fontWeight = FontWeight.Bold)
            "`" -> SpanStyle(fontFamily = FontFamily.Monospace)
            else -> SpanStyle(fontStyle = FontStyle.Italic)
        }
        withStyle(style) { append(content) }
        cursor = end + marker.length
    }
}

internal sealed interface MarkdownBlock {
    data object Blank : MarkdownBlock
    data class Heading(val level: Int, val text: String) : MarkdownBlock
    data class Bullet(val text: String) : MarkdownBlock
    data class Numbered(val number: String, val text: String) : MarkdownBlock
    data class Quote(val text: String) : MarkdownBlock
    data class Code(val text: String) : MarkdownBlock
    data class Paragraph(val text: String) : MarkdownBlock
}

internal object MarkdownParser {
    private val heading = Regex("^(#{1,6})\\s+(.+)$")
    private val numbered = Regex("^(\\d+)[.)]\\s+(.+)$")

    fun blocks(markdown: String): List<MarkdownBlock> {
        var inCodeFence = false
        return markdown.lines().mapNotNull { line ->
            if (line.trimStart().startsWith("```")) {
                inCodeFence = !inCodeFence
                null
            } else if (inCodeFence) MarkdownBlock.Code(line)
            else parseLine(line)
        }
    }

    private fun parseLine(line: String): MarkdownBlock = when {
        line.isBlank() -> MarkdownBlock.Blank
        heading.matches(line) -> heading.matchEntire(line)!!.destructured.let { (hashes, text) -> MarkdownBlock.Heading(hashes.length, text) }
        line.startsWith("- ") || line.startsWith("* ") || line.startsWith("+ ") -> MarkdownBlock.Bullet(line.drop(2))
        numbered.matches(line) -> numbered.matchEntire(line)!!.destructured.let { (number, text) -> MarkdownBlock.Numbered(number, text) }
        line.startsWith("> ") -> MarkdownBlock.Quote(line.drop(2))
        else -> MarkdownBlock.Paragraph(line)
    }
}
