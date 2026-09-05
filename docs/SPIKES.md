# Phase 0 spike results

Environment: AGP 9.0.1, Kotlin 2.3.20, `litertlm-android` 0.16.1, `litert` 2.2.0, AppSearch 1.1.0,
Apryse PDFNet 12.1.0 (demo mode, no licence key configured), API 36 x86_64 emulator (4 GB RAM,
SwiftShader, no GPU). Rows marked **device** still need a run on the two weakest target phones;
the on-device path for those is `Diagnostics -> Run spike self-test` (`SelfTest.kt`).

How to reproduce:

```sh
./gradlew :app:testDebugUnitTest                # JVM: tokenizer parity, layout, chunker, assembler
./gradlew :app:connectedDebugAndroidTest        # emulator/device: Spike1/3/4/5 tests under spikes/
adb logcat -s SPIKE1 SPIKE3 SPIKE4 SPIKE5       # timings and geometry dumps
```

| Spike | Decision under test | Result | Evidence |
|---|---|---|---|
| 1 | `litertlm` + `litert` packaging and runtime coexistence | **Green** | `Spike1NativeCoexistenceTest`: `libLiteRt`, `liblitertlm_jni`, `liblitert_jni` all load in one process; `Engine` and `CompiledModel` classes initialise; `Environment.create` succeeds. APK carries exactly one `libLiteRt.so` per ABI (`pickFirsts`), 16 KB aligned (`scripts/check_16kb_alignment.sh`). Libraries are mmapped from `base.apk` (`useLegacyPackaging=false`), so `/proc/self/maps` shows the APK path, not `.so` names. |
| 2a | Pure-Kotlin SentencePiece parity with Python `sentencepiece` | **Green** | `SentencePieceParityTest`: 50/50 strings (Latin, de/fr/es, zh, ja, ko, mixed, tables, control-token literals, whitespace edge cases) produce identical ids against the real Gemma `tokenizer.model` (sha256 `1299c11d…`, ungated copy at `unsloth/embeddinggemma-300m`). Fixture: `app/src/test/resources/tokenizer_fixtures.json`, regenerate with `scripts/gen_tokenizer_fixtures.py`. Found and fixed: user-defined symbols (`<start_of_turn>`) must be matched verbatim before BPE; `<bos>` literal text must *not* map to the control id (matches reference). |
| 2b | EmbeddingGemma latency CPU/GPU, cosine sanity | **Device** | Needs the gated `embeddinggemma-300M_seq512_mixed-precision.tflite`. `SelfTest.embedderChecks` prints init time, ms/chunk, dim and `cosine(q, relevant) > cosine(q, unrelated)` for en and ja; the same 50-string parity check runs on device from `assets/tokenizer_fixtures.json`. |
| 3 | AppSearch 1.1.0 `LocalStorage` hybrid search | **Green** | `Spike3AppSearchHybridTest` on API 36: all required features supported (`SCHEMA_EMBEDDING_PROPERTY_CONFIG`, `SEARCH_SPEC_SEARCH_STRING_PARAMETERS`, `SEARCH_SPEC_ADVANCED_RANKING_EXPRESSION`, `LIST_FILTER_QUERY_LANGUAGE`, plus `SCHEMA_EMBEDDING_QUANTIZATION`, `NUMERIC_SEARCH`, `VERBATIM_SEARCH`). Query `getSearchStringParameter(0) OR semanticSearch(getEmbeddingParameter(0), 0.3, 1)` with ranking `sum(matchedSemanticScores) + 0.05*relevanceScore` returns the right chunk for vector-only, ja `東京`, zh `北京`, ko `서울`, en `Paris`, prefix `capi`, table token `Revenue`; `首都` hits both ja and zh chunks. Namespace isolation and `removeByNamespace` verified. 1 500 chunks (≈500 pages): put 4.7 s (3.1 ms/chunk incl. flushes), hybrid search 12–18 ms. `LocalStorage` is bundled, so support does not depend on the OS `AppSearch` version. |
| 4a | GPU -> CPU fallback contract across processes | **Green** | `Spike4GpuFallbackTest` binds the real `:inference` service and loads a corrupt `.litertlm` with `allowGpu=true`: LiteRT-LM throws `LiteRtLmJniException: INVALID_ARGUMENT: Invalid magic number` (clean exception, no crash); the service writes `gpu_disabled.marker`, reports "retrying on CPU", fails again, and `onFailed` reaches the UI process. Second load honours the marker (first stage "Creating engine (CPU)"); clearing the marker re-enables the GPU attempt. |
| 4b | Gemma 4 E2B GPU init time and driver stability | **Device** | Needs a real GPU and the gated model. Measure cold (shader compile) and warm init on the two weakest targets; the watchdog timeouts (120 s / 30 s in `EngineWatchdog`) must exceed the observed cold/warm times with margin. Force a crash (e.g. kill `:inference` mid-init) and confirm the watchdog writes the marker and the next load is CPU. |
| 5 | Apryse `TextExtractor` on CJK, multi-column, tabular pages | **Green with two fixes** | `Spike5ApryseExtractionTest` (PDFs generated on device with `android.graphics.pdf`, embedded Noto CJK). CJK: text and per-word geometry extracted; 75 ms/page on a 45-line page (100 pages in 7.5 s on the emulator). Multi-column: 24 lines, no line merges the two columns, flow order (column 1 fully before column 2). Image-only page: `hasImages=true`, 0 chars, renders at the requested DPI (OCR routing input). Repeated header/footer/page numbers stripped across 6 pages. **Finding 1:** ToUnicode returned CJK *radicals* for common ideographs (`日` as U+2F47 Kangxi, `民` as U+2EA0 Radicals Supplement) -> `TextNormalizer` (NFKC + `EquivalentUnifiedIdeograph.txt` map) applied to every extracted word. **Finding 2:** on tables, TextExtractor emitted every cell as its own one-word line in column order (other generators yield one vertical "line" per column) -> `RowReflow` rebuilds true rows from geometry before `TableClusterer`, which then yields `| Year | Revenue | Profit | Margin |` plus three rows. Two-column body text is unaffected (unit-tested). |

## Decisions confirmed

- Keep `litertlm-android` + `litert` with `pickFirsts("**/libLiteRt.so")`.
- Keep the pure-Kotlin SentencePiece tokenizer (parity 50/50); no DJL dependency.
- Keep AppSearch `LocalStorage` as the hybrid store; brute-force cosine over 1 500 quantised vectors is 12–18 ms, so ANN is not needed at 500 pages.
- Keep Apryse `TextExtractor` with the two post-processing steps above; no need for a different reading-order strategy for multi-column text.
- The marker-file fallback works for clean GPU exceptions; process death is the watchdog's job (unchanged).

## Open items (real hardware)

- 2b and 4b above. Paste the `Run spike self-test` output and `SPIKE4` init timings per device into this file.
- Korean: Skia-generated PDFs dropped inter-word spaces (`서울은대한민국의수도입니다.`); Icing segments Hangul with ICU so keyword search still works, but verify on real Korean PDFs (Word/Hancom output keeps space glyphs).
