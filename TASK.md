# Implementation Tasks: On-Device Android RAG

- `[x]` **Phase 0: Spikes (High-Risk Verification)** — results in `docs/SPIKES.md`
  - `[x]` Spike 1: Verify `litertlm` + `litert:2.x` packaging and runtime coexistence (`libLiteRt.so` pickFirsts). Green (`Spike1NativeCoexistenceTest`).
  - `[x]` Spike 2: EmbeddingGemma on-device: SentencePiece parity on 50 CJK/Latin strings vs Python (green, `SentencePieceParityTest` 50/50), latency CPU/GPU (device-only; `SelfTest`, needs gated `.tflite`).
  - `[x]` Spike 3: AppSearch 1.1.0 feature-support log (`Features.isFeatureSupported`) + hybrid query (CJK keyword hits). Green (`Spike3AppSearchHybridTest`).
  - `[x]` Spike 4: Gemma GPU init fallback: exception path + marker verified across processes (`Spike4GpuFallbackTest`); GPU init timing on target phones is device-only.
  - `[x]` Spike 5: Apryse `TextExtractor` output profiling on CJK, multi-column, and tabular PDFs. Green after `TextNormalizer` and `RowReflow` fixes (`Spike5ApryseExtractionTest`).

- `[x]` **Phase 1: Skeleton (Project & IPC Foundation)**
  - `[x]` Initialize Android project with Compose & Material 3 (`minSdk 29`, `targetSdk 35`).
  - `[x]` Configure `settings.gradle.kts` with Apryse repository.
  - `[x]` Configure `build.gradle.kts` (AppSearch 1.1.0, LiteRT-LM, LiteRT 2.x, Apryse, ML Kit 16.0.1, Room).
  - `[x]` Configure `AndroidManifest.xml` (Foreground service dataSync, OpenCL, VNDK, permissions).
  - `[x]` Implement `core/process/ProcessGuard.kt` (`Application.onCreate` process branching).
  - `[x]` Define AIDL interfaces (`IAiInferenceService.aidl`, `IStreamCallback.aidl`, `IIndexingCallback.aidl`, `IEngineCallback.aidl`, `IInstallCallback.aidl`, parcelables).
  - `[x]` Setup Room Database (`ui/data/ChatDatabase.kt` for documents and messages).
  - `[x]` Implement Model Manager (`ui/download/ModelDownloadManager.kt`, `ModelCatalog.kt`) with streaming SHA-256 verify & copy.

- `[x]` **Phase 2: Generation (LLM Engine & Chat UI)**
  - `[x]` Implement `inference/llm/GemmaEngine.kt` (init, backend selection, per-turn `Conversation`, close).
  - `[x]` Implement `:ui` `EngineWatchdog.kt` (120s / 30s timeout, atomic `gpu_disabled.marker`, service restart).
  - `[x]` Implement streaming response flow with `generationId` validation and cancellation support.
  - `[x]` Build `ui/screens/ChatScreen.kt` with basic chat generation without retrieval (`PlainChatUseCase`).

- `[x]` **Phase 3: Ingestion (PDF Parsing, Chunking & Indexing)**
  - `[x]` Implement `inference/pdf/AprysePdfExtractor.kt` with bounding boxes and `HeaderFooterStripper.kt`.
  - `[x]` Implement `TableClusterer.kt` (X-band clustering with table-header injection).
  - `[x]` Implement `inference/ocr/MlKitOcr.kt` and `ScriptDetector.kt` (per-document script routing, 100% scanned fallback).
  - `[x]` Implement `inference/chunk/ScriptAwareChunker.kt` (CJK 300-330 chars, European 1000-1200 chars, 10-15% overlap).
  - `[x]` Implement `inference/embed/EmbeddingGemmaEmbedder.kt` with `SentencePieceTokenizer.kt` (512d Matryoshka + L2 normalization).
  - `[x]` Implement `inference/store/PdfChunkDocument.java` and `AppSearchVectorStore.kt` (batch putAsync x100, flush x500, removeByNamespace).
  - `[x]` Implement `inference/index/IndexPdfUseCase.kt` with cancellation (partial index purge) and `IndexingScreen.kt`.

- `[x]` **Phase 4: Retrieval + RAG (End-to-End Pipeline)**
  - `[x]` Implement Hybrid Search in `AppSearchVectorStore.kt` (advanced query syntax, keyword + cosine ranking).
  - `[x]` Implement `inference/llm/ContextAssembler.kt` (4k token budget, citation tags, QA seed).
  - `[x]` Implement `inference/chat/AnswerQuestionUseCase.kt`.
  - `[x]` Wire citation chips in `ChatScreen.kt` to `CitationSheet.kt` fetching via AIDL `chunkId`.

- `[x]` **Phase 5: Hardening (Resiliency, Performance & Evaluation)**
  - `[x]` Implement device/RAM gating (`<6GB` refuse, `6-8GB` E2B, `>=12GB` E4B) and storage pre-flight (~2x model size).
  - `[x]` Implement thermal monitoring (`PowerManager.OnThermalStatusChangedListener`) with UI badge.
  - `[x]` Implement LMK recovery in `:ui` (*"Reloading model"*).
  - `[x]` Set up CI 16 KB page-size alignment checks (`scripts/check_16kb_alignment.sh`, `.github/workflows/ci.yml`).
  - `[x]` Build Retrieval Evaluation Harness (~30 Q/A per language, `page-hit@5` logging).
