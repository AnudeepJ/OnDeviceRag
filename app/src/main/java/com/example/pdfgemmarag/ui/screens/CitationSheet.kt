package com.example.pdfgemmarag.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.pdfgemmarag.core.model.Citation
import com.example.pdfgemmarag.ui.RagViewModel

/** Fetches the chunk text by id from :inference on demand, so token streams stay small. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CitationSheet(chunkId: String, viewModel: RagViewModel, onDismiss: () -> Unit) {
    var citation by remember(chunkId) { mutableStateOf<Citation?>(null) }
    var failed by remember(chunkId) { mutableStateOf(false) }
    LaunchedEffect(chunkId) {
        citation = viewModel.citation(chunkId)
        failed = citation == null
    }
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(Modifier.fillMaxWidth().padding(20.dp).verticalScroll(rememberScrollState())) {
            when {
                citation != null -> {
                    Text("Page ${citation!!.pageNumber} · chunk ${citation!!.chunkIndex}", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(12.dp))
                    Text(citation!!.text, style = MaterialTheme.typography.bodyMedium)
                }
                failed -> Text("This passage is no longer in the index.", style = MaterialTheme.typography.bodyMedium)
                else -> CircularProgressIndicator()
            }
            Spacer(Modifier.height(32.dp))
        }
    }
}
