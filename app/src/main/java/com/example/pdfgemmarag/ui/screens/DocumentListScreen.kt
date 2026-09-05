package com.example.pdfgemmarag.ui.screens

import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.pdfgemmarag.core.model.EngineStatus
import com.example.pdfgemmarag.ui.RagViewModel
import com.example.pdfgemmarag.ui.data.DocumentEntity

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DocumentListScreen(
    viewModel: RagViewModel,
    onOpenChat: (DocumentEntity) -> Unit,
    onOpenPlainChat: () -> Unit,
    onOpenModels: () -> Unit,
    onOpenDiagnostics: () -> Unit,
) {
    val ui by viewModel.ui.collectAsStateWithLifecycle()
    val docs by viewModel.documents.collectAsStateWithLifecycle()
    val context = LocalContext.current

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            var name = "document.pdf"
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) name = c.getString(0) ?: name
            }
            viewModel.importPdf(uri, name)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Documents") },
                actions = {
                    IconButton(onClick = onOpenModels) { Icon(Icons.Default.Memory, contentDescription = "Models") }
                    IconButton(onClick = onOpenDiagnostics) { Icon(Icons.Default.BugReport, contentDescription = "Diagnostics") }
                },
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { if (ui.indexing == null && !ui.busy) picker.launch(arrayOf("application/pdf")) },
                icon = { Icon(Icons.Default.Add, contentDescription = null) },
                text = { Text("Add PDF") },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            EngineBanner(ui, onOpenModels)
            if (ui.engine.state == EngineStatus.State.READY) {
                Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), horizontalArrangement = Arrangement.End) {
                    TextButton(onClick = onOpenPlainChat) {
                        Icon(Icons.AutoMirrored.Filled.Chat, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Chat without a document")
                    }
                }
            }
            ui.indexing?.let { p ->
                IndexingScreen(
                    docName = ui.indexingDocName,
                    progress = p,
                    onCancel = { viewModel.cancelIndexing() },
                )
            }
            if (docs.isEmpty()) {
                Column(Modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("No documents yet", style = MaterialTheme.typography.titleLarge)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        if (ui.embeddingReady) "Add a PDF to index it on-device. Nothing leaves the phone."
                        else "Install the embedding model from the Models screen, then add a PDF.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            } else {
                LazyColumn(Modifier.fillMaxSize(), contentPadding = androidx.compose.foundation.layout.PaddingValues(bottom = 96.dp)) {
                    items(docs, key = { it.hash }) { doc ->
                        DocumentRow(
                            doc = doc,
                            onClick = { if (doc.status == RagViewModel.STATUS_READY) onOpenChat(doc) },
                            onDelete = { viewModel.deleteDocument(doc) },
                            onReindex = { viewModel.reindex(doc) },
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun EngineBanner(ui: RagViewModel.UiState, onOpenModels: () -> Unit) {
    val (label, hint) = when (ui.engine.state) {
        EngineStatus.State.READY -> "Model ready · ${ui.engine.backend}" to ui.engine.modelName
        EngineStatus.State.LOADING -> "Loading model…" to ui.engineMessage
        EngineStatus.State.FAILED -> "Model failed" to ui.engine.message
        EngineStatus.State.UNLOADED -> "No model loaded" to "Tap to choose a Gemma model"
    }
    Row(
        Modifier.fillMaxWidth().clickable { onOpenModels() }.padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        AssistChip(onClick = onOpenModels, label = { Text(label) })
        Spacer(Modifier.width(12.dp))
        Text(hint, style = MaterialTheme.typography.bodySmall, maxLines = 2)
    }
    if (ui.engine.state == EngineStatus.State.LOADING) LinearProgressIndicator(Modifier.fillMaxWidth().padding(horizontal = 16.dp))
}

@Composable
private fun DocumentRow(doc: DocumentEntity, onClick: () -> Unit, onDelete: () -> Unit, onReindex: () -> Unit) {
    Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp).clickable { onClick() }) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(doc.name, style = MaterialTheme.typography.titleMedium, maxLines = 2)
                Spacer(Modifier.height(4.dp))
                Text(
                    when (doc.status) {
                        RagViewModel.STATUS_READY -> "${doc.pageCount} pages · ${doc.chunkCount} chunks · ${doc.script.lowercase()}"
                        RagViewModel.STATUS_INDEXING -> "Indexing…"
                        else -> "Failed: ${doc.error ?: "unknown"}"
                    },
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (doc.status == RagViewModel.STATUS_FAILED) IconButton(onClick = onReindex) { Icon(Icons.Default.Refresh, contentDescription = "Re-index") }
            IconButton(onClick = onDelete) { Icon(Icons.Default.Delete, contentDescription = "Delete") }
        }
    }
}
