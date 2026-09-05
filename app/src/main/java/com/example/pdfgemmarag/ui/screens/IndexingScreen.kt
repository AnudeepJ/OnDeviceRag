package com.example.pdfgemmarag.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.pdfgemmarag.core.model.IndexingProgress

/** Live ingestion progress card shown on the document list while a PDF is being indexed. */
@Composable
fun IndexingScreen(
    docName: String?,
    progress: IndexingProgress,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
        Column(Modifier.padding(16.dp)) {
            Text("Indexing ${docName.orEmpty()}", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(4.dp))
            Text(
                buildString {
                    append(progress.stage.name.lowercase().replaceFirstChar { it.uppercase() })
                    if (progress.total > 0) append(" ${progress.current}/${progress.total}")
                    if (progress.detail.isNotEmpty()) append(" · ${progress.detail}")
                    if (progress.paused) append(" · paused (device too hot)")
                },
                style = MaterialTheme.typography.bodySmall,
            )
            Spacer(Modifier.height(8.dp))
            if (progress.total > 0) LinearProgressIndicator(progress = { progress.fraction }, modifier = Modifier.fillMaxWidth())
            else LinearProgressIndicator(Modifier.fillMaxWidth())
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                TextButton(onClick = onCancel) { Text("Cancel") }
            }
        }
    }
}
