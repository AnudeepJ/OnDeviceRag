# Gemma Vision Spike

## Decision being tested

Apryse remains the primary PDF text and geometry extractor. Production indexing continues to use
ML Kit only for image-only or nearly image-only pages. This spike measures whether Gemma Vision is
useful as a selective visual enrichment backend for tables, diagrams and photographs, and whether
its exact OCR is strong enough to replace ML Kit for any subset of those pages.

The spike does not write Gemma output to AppSearch, manifests, chunks or citations.

## Current implementation

`Spike6GemmaVisionDeviceTest`:

- loads the already-installed multimodal `.litertlm` model with language and vision GPU backends;
- renders selected PDF pages through Apryse;
- runs ML Kit OCR and Gemma Vision on the same pixels;
- preserves image-before-text ordering required by Gemma 4;
- records term recall, JSON-schema adherence, same-row association, position-specific observations
  and directed flowchart-edge accuracy;
- records engine initialization, per-page OCR and vision latency, PSS and thermal status;
- writes the full evidence to `files/vision_spike_report.json`.

The default Safety Manual cases are:

| Page | Visual structure | What the case tests |
|---:|---|---|
| 20 | Illustrated glove table | Exact row association and units |
| 29 | Six-photo montage | Visual hazard interpretation beyond OCR |
| 42 | Emergency flowchart | Labels, arrows and Yes/No branch topology |

Pages are instrumentation parameters so the harness is not coupled to this PDF.

## Safe device execution

Do not use `connectedDebugAndroidTest` on a development device when the only model copy is under
the app's private `files/models` directory. Android Gradle Plugin may uninstall the test and target
packages during cleanup, deleting their private data.

Build APKs, install them with `adb install -r -t`, then invoke instrumentation directly:

```text
./gradlew :app:assembleDebug :app:assembleDebugAndroidTest
adb -s DEVICE install -r -t app/build/outputs/apk/debug/app-debug.apk
adb -s DEVICE install -r -t app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s DEVICE shell am instrument -w -r \
  -e class com.example.pdfgemmarag.spikes.Spike6GemmaVisionDeviceTest \
  -e visionPages 20,29,42 \
  com.example.pdfgemmarag.test/androidx.test.runner.AndroidJUnitRunner
adb -s DEVICE exec-out run-as com.example.pdfgemmarag \
  cat files/vision_spike_report.json > vision_spike_report.json
```

Optional arguments:

- `visionPdfPath`: app-readable absolute PDF path;
- `visionPages`: comma-separated one-based pages;
- `visionMaxEdge`: render long edge from 768 to 2048 pixels (default 1536).

## Evaluation gates

Gemma Vision is eligible for selective enrichment only if real-device evidence shows:

1. It adds material diagram, photograph or table relationships unavailable in extracted text.
2. Its latency, memory and thermal cost are acceptable when limited to a small number of pages.
3. Output is explicitly tagged as visual inference and remains separate from authoritative text.
4. Exact values are corroborated against Apryse or ML Kit tokens before they are answer evidence.
5. Native failure, timeout or memory pressure cleanly skips enrichment without failing indexing.

Removing ML Kit requires a higher bar: Gemma must match exact transcription and numeric recall and
must provide sufficiently stable geometry for table association and citation highlighting. A good
semantic description by itself is not evidence that ML Kit can be removed.

## Nothing Phone result - 2026-09-09

Device: Nothing A001, Gemma 4 E2B, LiteRT-LM 0.16.1, GPU language and vision backends, 1536-pixel
page render, one image per fresh conversation. Thermal status ended at `LIGHT` (`1`) in both runs.

| Page | Case | ML Kit | Gemma Vision | Result |
|---:|---|---:|---:|---|
| 20 | Illustrated glove table | 516 ms, 80% term recall | 36,252 ms, 100% term recall | 100% same-row recall, but only 33% requested-schema adherence |
| 29 | Six-photo electrical montage | 367 ms, 50% term recall | 45,748 ms, 25% term recall | 0% position-specific evidence recall; descriptions were generic |
| 42 | Emergency flowchart | 246 ms, 100% term recall | 49,088 ms, 100% term recall | Only 7 of 12 directed edges were correct (58.3%) |

Warm engine initialization was 4.2-4.7 seconds. PSS immediately after initialization was about
2.19 GB. The first cold run initialized in 36.7 seconds. ML Kit's text does not itself express table
or arrow topology, but its fast deterministic words and geometry remain suitable evidence for the
existing structure analyzers.

The flowchart result demonstrates why text recall is insufficient for visual QA. Gemma emitted
valid JSON containing every expected label, but it:

- attached `no` to the unconditional arrow from `Raise alarm!`;
- routed `Can you handle it? -> no` to medical assistance instead of evacuation;
- added conditions to an unconditional arrow into `Emergency handled successfully?`;
- invented `Seek medical assistance -> Handle the emergency`;
- invented `Perform follow-up action -> Evacuate area immediately`.

### Spike decision

Do not replace ML Kit with Gemma 4 E2B for page OCR. Do not index Gemma's inferred flowchart edges or
photo hazards as authoritative facts.

Retain a future option for tightly selected table enrichment when deterministic table reconstruction
has low confidence. Any Gemma-produced rows must be validated against Apryse/ML Kit tokens and units,
tagged as `VISUAL_INFERENCE`, cached after indexing, and skipped on timeout, memory pressure or thermal
pressure. At measured latency, vision generation must never be placed in the normal question path.
