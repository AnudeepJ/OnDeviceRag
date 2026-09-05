package com.example.pdfgemmarag.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.pdfgemmarag.ui.RagViewModel

/** Phase 0 spike output on a real device: versions, AppSearch feature flags, backend, self-test. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DiagnosticsScreen(viewModel: RagViewModel, onBack: () -> Unit) {
    val ui by viewModel.ui.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { viewModel.loadDiagnostics() }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Diagnostics") },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back") } },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { viewModel.loadDiagnostics() }) { Text("Refresh") }
                Button(onClick = { viewModel.runSelfTest() }, enabled = ui.selfTest != "Running…") { Text("Run spike self-test") }
            }
            Spacer(Modifier.height(8.dp))
            Button(onClick = { viewModel.probeRetrieval() }, enabled = ui.probe != "Probing live index…") { Text("Probe live index") }
            Spacer(Modifier.height(12.dp))
            Text(ui.diagnostics ?: "Loading…", fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
            ui.probe?.let {
                Spacer(Modifier.height(16.dp))
                Text("Live index probe", style = MaterialTheme.typography.titleMedium)
                Text(it, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
            }
            ui.selfTest?.let {
                Spacer(Modifier.height(16.dp))
                Text("Self-test", style = MaterialTheme.typography.titleMedium)
                Text(it, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
