package com.example.pdfgemmarag.ui.screens

import android.app.DownloadManager
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import com.example.pdfgemmarag.ui.DeviceGate
import com.example.pdfgemmarag.ui.RagViewModel
import com.example.pdfgemmarag.ui.download.CatalogEntry
import com.example.pdfgemmarag.ui.download.ModelCatalog

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelManagerScreen(viewModel: RagViewModel, onBack: () -> Unit) {
    val ui by viewModel.ui.collectAsStateWithLifecycle()
    val downloads by viewModel.downloads.states.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val gate = viewModel.gate

    val filePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            var name = "model.bin"
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c -> if (c.moveToFirst()) name = c.getString(0) ?: name }
            viewModel.importLocalModel(uri, name)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Models") },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } },
            )
        },
    ) { padding ->
        LazyColumn(Modifier.padding(padding).fillMaxSize(), contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Device", style = MaterialTheme.typography.titleMedium)
                        Text(gate.describe(), style = MaterialTheme.typography.bodySmall)
                        when (gate.tier) {
                            DeviceGate.Tier.UNSUPPORTED -> Text("Less than 6 GB RAM: Gemma 4 cannot run reliably on this device.", color = MaterialTheme.colorScheme.error)
                            DeviceGate.Tier.LOW -> Text("6-8 GB RAM: E2B only; chat is paused while a document is indexing.", style = MaterialTheme.typography.bodySmall)
                            else -> Unit
                        }
                        Spacer(Modifier.height(8.dp))
                        Text("Engine: ${ui.engine.state} ${ui.engine.backend} ${ui.engineMessage}", style = MaterialTheme.typography.bodySmall)
                        if (ui.gpuDisabled) {
                            Text("GPU disabled: ${ui.gpuDisabledReason ?: "previous failure"}", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                            TextButton(onClick = { viewModel.retryGpu() }) { Text("Retry GPU") }
                        }
                        Text(
                            if (ui.embeddingReady) "Embedding model: installed" else "Embedding model: missing (required for indexing)",
                            style = MaterialTheme.typography.bodySmall,
                            color = if (ui.embeddingReady) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }

            item { Text("Installed Gemma models", style = MaterialTheme.typography.titleMedium) }
            if (ui.installedModels.isEmpty()) item { Text("None yet. Download below or import a local .litertlm file.", style = MaterialTheme.typography.bodySmall) }
            items(ui.installedModels, key = { it.absolutePath }) { file ->
                val loaded = ui.engine.modelPath == file.absolutePath && ui.engine.state == EngineStatus.State.READY
                val loading = ui.engine.modelPath == file.absolutePath && ui.engine.state == EngineStatus.State.LOADING
                Card(Modifier.fillMaxWidth()) {
                    Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(file.name, style = MaterialTheme.typography.bodyLarge)
                            Text("${file.length() shr 20} MB${if (loaded) " · loaded on ${ui.engine.backend}" else ""}", style = MaterialTheme.typography.bodySmall)
                        }
                        if (loaded) OutlinedButton(onClick = { viewModel.unloadEngine() }) { Text("Unload") }
                        else Button(onClick = { viewModel.loadEngine(file.absolutePath) }, enabled = !loading && gate.canLoadLlm) { Text(if (loading) "Loading…" else "Load") }
                        IconButton(onClick = { viewModel.deleteModel(file) }) { Icon(Icons.Default.Delete, contentDescription = "Delete") }
                    }
                }
            }
            item {
                OutlinedButton(onClick = { filePicker.launch(arrayOf("*/*")) }, modifier = Modifier.fillMaxWidth()) {
                    Text("Import local file (.litertlm / .tflite / tokenizer .model)")
                }
                ui.install?.let { (name, copied, total) ->
                    Spacer(Modifier.height(8.dp))
                    Text("Verifying and installing $name: ${copied shr 20}/${total shr 20} MB", style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(progress = { if (total > 0) (copied.toFloat() / total).coerceIn(0f, 1f) else 0f }, modifier = Modifier.fillMaxWidth())
                }
            }

            item { Text("Download catalog", style = MaterialTheme.typography.titleMedium) }
            items(ModelCatalog.entries.filter { it.minRamGb <= gate.totalRamGb + 0.5 }, key = { it.id }) { entry ->
                CatalogRow(entry, downloads[entry.id], installed = viewModel.downloads.installed(entry), onDownload = { viewModel.download(entry) }, onCancel = { viewModel.cancelDownload(entry) })
            }
            item {
                Text(
                    "Gemma weights are gated on Hugging Face and must be mirrored on your own CDN (MODEL_CDN_BASE_URL + SHA256_* Gradle properties). " +
                        "Until then, import files picked from device storage.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun CatalogRow(entry: CatalogEntry, state: com.example.pdfgemmarag.ui.download.ModelDownloadManager.DownloadState?, installed: Boolean, onDownload: () -> Unit, onCancel: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(entry.displayName, style = MaterialTheme.typography.bodyLarge)
                    Text(entry.description, style = MaterialTheme.typography.bodySmall)
                }
                when {
                    installed -> Text("Installed", style = MaterialTheme.typography.labelLarge)
                    state?.running == true -> TextButton(onClick = onCancel) { Text("Cancel") }
                    else -> Button(onClick = onDownload, enabled = entry.downloadable) { Text("Download") }
                }
            }
            if (!installed && state != null && state.running) {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(progress = { state.fraction }, modifier = Modifier.fillMaxWidth())
                Text("${state.bytesSoFar shr 20} / ${state.totalBytes shr 20} MB", style = MaterialTheme.typography.bodySmall)
            } else if (!installed && state?.status == DownloadManager.STATUS_FAILED) {
                Text("Download failed (${downloadFailure(state.reason)})", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            } else if (!installed && state?.status == DownloadManager.STATUS_SUCCESSFUL) {
                Text("Downloaded; verifying and installing…", style = MaterialTheme.typography.bodySmall)
            } else if (!installed && !entry.downloadable) {
                Text("Unavailable: configure the CDN URL and SHA-256 first.", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

private fun downloadFailure(reason: Int): String = when (reason) {
    DownloadManager.ERROR_CANNOT_RESUME -> "cannot resume"
    DownloadManager.ERROR_DEVICE_NOT_FOUND -> "storage unavailable"
    DownloadManager.ERROR_FILE_ALREADY_EXISTS -> "file already exists"
    DownloadManager.ERROR_FILE_ERROR -> "file error"
    DownloadManager.ERROR_HTTP_DATA_ERROR -> "HTTP data error"
    DownloadManager.ERROR_INSUFFICIENT_SPACE -> "insufficient space"
    DownloadManager.ERROR_TOO_MANY_REDIRECTS -> "too many redirects"
    DownloadManager.ERROR_UNHANDLED_HTTP_CODE -> "HTTP error"
    else -> "reason $reason"
}
