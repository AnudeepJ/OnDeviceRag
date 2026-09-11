# `generic_hard` device evidence — 2026-09-11

This directory preserves the reports used to accept road-ahead points 1–5. The device-generated JSON files are byte-for-byte copies of reports originally pulled into ignored `build/` directories. `SHA256SUMS` records every archived file's content hash, and `manifest.json` classifies each artifact.

## Verdict supported by this evidence

- Nothing Phone (Android report generation v29): construction safety **12/12**, safety PDF **23/23**, and SafetyManual **50/50** for both retrieval and answer checks.
- Pixel 10 (v28): focused baseline **19/19** and generated live QA **21/21** for both retrieval and answer checks.
- Every accepted single-PDF run records `thermalOverride: false`. The Nothing v29 runs record thermal status `1` at start and end; the Pixel runs record `1/1` and `0/0`.
- The corrected 210-query multilingual benchmark favors hybrid retrieval. `hybrid_balanced` reached **97.62% Hit@5** and **0.9738 MRR**, compared with dense-only at **93.81% Hit@5 / 0.9313 MRR** and lexical-only at **68.10% Hit@5 / 0.6476 MRR**.
- GPU embedding produced zero vectors on both tested devices. The retained backend reports show the CPU fallback restoring non-zero, normalized vectors.

## Directory contents

| Directory | Role | Contents |
|---|---|---|
| `accepted/nothing-v29/` | Authoritative acceptance | Final three-PDF Nothing reports after the v29 retention migration and reindex |
| `accepted/pixel-v28/` | Authoritative acceptance | Final focused and generated QA reports on Pixel 10 |
| `multilingual/` | Accepted benchmark | Corrected strategy-isolated, real-embedding comparison over 210 queries |
| `embedding/` | Diagnostic evidence | Tensor contract, pre-fix GPU failure, and post-fix CPU fallback reports on both devices |
| `host/` | Host verification | Aggregate Gradle result and 37 JUnit XML reports with only the local hostname redacted |
| `regressions/` | Supporting failure evidence | Reports that exposed the document-retention and query-routing defects later fixed |
| `invalid/` | Audit only | A superseded mixed-mode retrieval comparison and a deliberate corpus-mismatch run |

`invalid/` must not be used to select retrieval weights or claim product quality. It is retained to make the evaluation history auditable.

## Benchmark scope

The multilingual report uses `acme-210-v1`: 105 synthetic fact chunks and 210 same-language questions across German, English, French, Italian, Japanese, Korean, and Chinese. It exercises the real EmbeddingGemma document/query vectors and an isolated AppSearch namespace. Each retrieval strategy has distinct candidate generation, is warmed before measurement, and is measured in three rotated passes.

This establishes multilingual retrieval behavior under a controlled corpus. It does not establish cross-language retrieval, scanned-PDF OCR quality, or performance on a broad production corpus. Those remain later validation work. The report was produced with v28; it remains applicable to the accepted v29 state because v29 changed PDF chunk-retention migration behavior, while the benchmark builds its own isolated synthetic corpus.

## Reproduce the checks

Host gate:

```bash
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew testDevDebugUnitTest assembleDevDebug compileDevDebugAndroidTestKotlin
```

The archived host result contains 292 tests, zero failures/errors, and three skipped tests. The JUnit reports preserve test names, timestamps, durations, and outcomes; their machine-local `hostname` attribute is replaced with `redacted-local-host`. The APK binaries are reproducible outputs and are not stored in Git.

Corrected multilingual comparison on a connected Android device:

```bash
adb shell am instrument -w \
  -e class com.example.pdfgemmarag.eval.RetrievalEvalDeviceTest#compareFrozenCandidateStrategies \
  com.example.pdfgemmarag.test/androidx.test.runner.AndroidJUnitRunner
```

Single-PDF acceptance is driven by `SinglePdfBaselineDeviceTest` and the fixture assets under `app/src/androidTest/assets/`. The indexed source PDF must match the selected fixture; `invalid/corpus-mismatch-home-pdf-vs-division-fixture.json` demonstrates why that identity check matters.

## Related implementation and review material

- `docs/ROAD_AHEAD_1_TO_5_ACCEPTANCE.md` — accepted contracts and final test matrix
- `docs/CURRENT_CHANGES_REVIEW_AND_NEXT_PLAN.md` — current review and next work
- `docs/APPSEARCH_AND_MULTILINGUAL_RECOMMENDATIONS.md` — benchmark interpretation and proposals
- `app/src/androidTest/java/com/example/pdfgemmarag/eval/RetrievalEvalDeviceTest.kt` — retrieval benchmark harness
- `app/src/androidTest/java/com/example/pdfgemmarag/eval/SinglePdfBaselineDeviceTest.kt` — live single-PDF harness
- `app/src/androidTest/assets/generated_live_qa.json` and `safety_manual_qa.json` — QA fixtures changed during this work
