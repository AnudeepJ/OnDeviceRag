package com.example.pdfgemmarag.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AssistChip
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.core.model.EngineStatus
import com.example.pdfgemmarag.ui.RagViewModel
import com.example.pdfgemmarag.ui.data.MessageEntity

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: RagViewModel, docHash: String, title: String, onBack: () -> Unit) {
    val ui by viewModel.ui.collectAsStateWithLifecycle()
    val chat by viewModel.chat.collectAsStateWithLifecycle()
    val messages by viewModel.messages(docHash).collectAsStateWithLifecycle(initialValue = emptyList())
    var input by remember { mutableStateOf("") }
    var citationToShow by remember { mutableStateOf<String?>(null) }
    val listState = rememberLazyListState()

    LaunchedEffect(docHash) { viewModel.openChat(docHash) }
    // Scroll on new messages or when generation starts — not on every token (that stalls the UI).
    LaunchedEffect(messages.size, chat.generating) {
        val count = messages.size + if (chat.generating) 1 else 0
        if (count > 0) listState.scrollToItem(count - 1)
    }
    var elapsedSec by remember { mutableStateOf(0) }
    LaunchedEffect(chat.generating, chat.generationId) {
        elapsedSec = 0
        if (!chat.generating) return@LaunchedEffect
        while (true) {
            kotlinx.coroutines.delay(1000)
            elapsedSec += 1
        }
    }

    val canAsk = ui.engine.state == EngineStatus.State.READY && !chat.generating &&
        !(viewModel.gate.chatBlockedWhileIndexing && ui.indexing != null)

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Column { Text(title, maxLines = 1, style = MaterialTheme.typography.titleMedium); Text(engineLine(ui), style = MaterialTheme.typography.bodySmall) } },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().imePadding()) {
            if (ui.engine.state != EngineStatus.State.READY) {
                Surface(color = MaterialTheme.colorScheme.errorContainer, modifier = Modifier.fillMaxWidth()) {
                    Text(
                        when (ui.engine.state) {
                            EngineStatus.State.LOADING -> "Loading model… ${ui.engineMessage}"
                            EngineStatus.State.FAILED -> "Model failed: ${ui.engine.message}"
                            else -> "No model loaded. Open Models and load a Gemma model."
                        },
                        Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
            LazyColumn(state = listState, modifier = Modifier.weight(1f).fillMaxWidth(), contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(messages, key = { it.id }) { m -> MessageBubble(m, onCitation = { citationToShow = it }) }
                if (chat.generating) {
                    item(key = "streaming") {
                        StreamingBubble(chat, elapsedSec, onCitation = { citationToShow = it })
                    }
                }
                chat.error?.let { err -> item(key = "error") { Text(err, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) } }
                chat.lastStats?.let { s ->
                    item(key = "stats") {
                        Text(
                            "ttft ${s.timeToFirstTokenMs} ms · ${"%.1f".format(s.approxTokensPerSecond)} tok/s · ${s.retrievedChunks} chunks · ~${s.contextTokensApprox} ctx tokens · ${s.backend}",
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                }
            }
            Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(
                    value = input, onValueChange = { input = it }, modifier = Modifier.weight(1f),
                    placeholder = { Text(if (docHash.isEmpty()) "Ask Gemma anything" else "Ask about this document") }, maxLines = 4, enabled = !chat.generating,
                )
                Spacer(Modifier.width(8.dp))
                if (chat.generating) {
                    TextButton(onClick = { viewModel.stopGeneration() }) {
                        Icon(Icons.Default.Stop, contentDescription = null)
                        Spacer(Modifier.width(4.dp))
                        Text("Stop")
                    }
                } else {
                    IconButton(onClick = { viewModel.ask(input); input = "" }, enabled = canAsk && input.isNotBlank()) {
                        Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Send")
                    }
                }
            }
        }
    }

    citationToShow?.let { id -> CitationSheet(chunkId = id, viewModel = viewModel, onDismiss = { citationToShow = null }) }
}

private fun engineLine(ui: RagViewModel.UiState): String = when (ui.engine.state) {
    EngineStatus.State.READY -> "${ui.engine.modelName} · ${ui.engine.backend}"
    EngineStatus.State.LOADING -> "loading…"
    else -> "no model"
}

@Composable
private fun MessageBubble(m: MessageEntity, onCitation: (String) -> Unit) {
    val isUser = m.role == "user"
    Column(Modifier.fillMaxWidth(), horizontalAlignment = if (isUser) Alignment.End else Alignment.Start) {
        Box(
            Modifier.widthIn(max = 340.dp).clip(RoundedCornerShape(16.dp))
                .background(if (isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant)
                .padding(12.dp),
        ) { Text(m.text, style = MaterialTheme.typography.bodyMedium) }
        val ids = m.citationIds.split(',').filter { it.isNotBlank() }
        if (ids.isNotEmpty()) CitationChips(ids.map { it to it.substringAfterLast(':') }, onCitation)
        if (!isUser && m.tokensPerSecond > 0) Text("${"%.1f".format(m.tokensPerSecond)} tok/s · ${m.backend}${if (m.cancelled) " · stopped" else ""}", style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun StreamingBubble(chat: RagViewModel.ChatState, elapsedSec: Int, onCitation: (String) -> Unit) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.Start) {
        Box(Modifier.widthIn(max = 340.dp).clip(RoundedCornerShape(16.dp)).background(MaterialTheme.colorScheme.surfaceVariant).padding(12.dp)) {
            Text(
                when {
                    chat.streamingText.isNotEmpty() -> chat.streamingText + "▍"
                    chat.streamingCitations.isEmpty() -> "Searching document…"
                    else -> "Generating answer… ${elapsedSec}s (first token can take 30–90s on GPU)"
                },
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        if (chat.streamingCitations.isNotEmpty()) CitationChips(chat.streamingCitations.map { it.chunkId to "p.${it.pageNumber}" }, onCitation)
    }
}

@Composable
private fun CitationChips(items: List<Pair<String, String>>, onCitation: (String) -> Unit) {
    Spacer(Modifier.height(4.dp))
    LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        items(items, key = { it.first }) { (id, label) ->
            AssistChip(onClick = { onCitation(id) }, label = { Text(if (label.startsWith("p.")) label else "#$label") })
        }
    }
}

/** Kept for API symmetry with the plan: citations arrive as ids and are resolved lazily. */
@Suppress("unused")
private fun Citation.label() = "Page $pageNumber"
