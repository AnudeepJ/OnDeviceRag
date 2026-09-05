package com.example.pdfgemmarag.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.remember
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation3.runtime.NavKey
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.ui.NavDisplay
import com.example.pdfgemmarag.ui.screens.ChatScreen
import com.example.pdfgemmarag.ui.screens.DiagnosticsScreen
import com.example.pdfgemmarag.ui.screens.DocumentListScreen
import com.example.pdfgemmarag.ui.screens.ModelManagerScreen
import kotlinx.serialization.Serializable

@Serializable data object DocumentsKey : NavKey
@Serializable data object ModelsKey : NavKey
@Serializable data object DiagnosticsKey : NavKey
@Serializable data class ChatKey(val docHash: String, val title: String) : NavKey

@Composable
fun RagNavigation(viewModel: RagViewModel) {
    val backStack = rememberNavBackStack(DocumentsKey)
    val ui by viewModel.ui.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(ui.notice) {
        val n = ui.notice ?: return@LaunchedEffect
        snackbar.showSnackbar(n)
        viewModel.clearNotice()
    }

    Scaffold(snackbarHost = { SnackbarHost(snackbar) }) { padding ->
        NavDisplay(
            backStack = backStack,
            onBack = { if (backStack.size > 1) backStack.removeLastOrNull() },
            modifier = Modifier.padding(padding),
            entryProvider = entryProvider {
                entry<DocumentsKey> {
                    DocumentListScreen(
                        viewModel = viewModel,
                        onOpenChat = { doc -> viewModel.openChat(doc.hash); backStack.add(ChatKey(doc.hash, doc.name)) },
                        onOpenPlainChat = { viewModel.openChat(""); backStack.add(ChatKey("", "Chat (no document)")) },
                        onOpenModels = { backStack.add(ModelsKey) },
                        onOpenDiagnostics = { backStack.add(DiagnosticsKey) },
                    )
                }
                entry<ModelsKey> { ModelManagerScreen(viewModel = viewModel, onBack = { if (backStack.size > 1) backStack.removeLastOrNull() }) }
                entry<DiagnosticsKey> { DiagnosticsScreen(viewModel = viewModel, onBack = { if (backStack.size > 1) backStack.removeLastOrNull() }) }
                entry<ChatKey> { key ->
                    ChatScreen(viewModel = viewModel, docHash = key.docHash, title = key.title, onBack = { if (backStack.size > 1) backStack.removeLastOrNull() })
                }
            },
        )
    }
}
