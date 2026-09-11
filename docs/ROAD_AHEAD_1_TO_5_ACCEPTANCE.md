# Generic-hard road ahead: points 1–5 acceptance

Status: **accepted for the current fully on-device, single-PDF scope on index v29**. The evidence contracts, English QA portfolio, and measured retrieval comparison are complete. Aconex library extraction remains point 6 and should be planned separately.

## 1. Correctness closure — accepted

The production path enforces exact document identity, section ancestry, list identity and completion, table/cell provenance, physical-page coverage, bounded structural expansion, and conservative deterministic-answer routes. Permanent tests cover the original defects and the additional failures found on physical devices.

The final v29 fixes also cover:

- validation of the selected 512-dimensional embedding output and automatic GPU-to-CPU fallback when the GPU returns zero or non-finite vectors;
- true dense-only and lexical-only benchmark isolation rather than running the combined AppSearch query with a zeroed rerank weight;
- subject-scoped exact-count lists, so “the five steps of HIRA” cannot select or truncate another five/eight-item list;
- letter-labelled rows when a named visual table is extracted as a logical list;
- retention of short lowercase source continuations that follow a heading, preventing source text such as `impractical.` from being dropped;
- sentence-level standard selection, including a subject constraint for questions such as the plywood standard/class lookup.

## 2. Final build, reindex and device runs — accepted

All three PDFs on Nothing A001 were reindexed to v29 through the production indexing service. The final v29 device portfolio ran with the normal thermal gate and `thermalOverride=false` in every report.

| Device / PDF / suite | Index | Retrieval | Answers | Result |
|---|---:|---:|---:|---|
| Nothing / `safety.pdf` / strict construction | v29 | 12/12 | 12/12 | Pass |
| Nothing / `safety.pdf` / broader construction | v29 | 23/23 | 23/23 | Pass |
| Nothing / `SafetyManual.pdf` / laboratory manual | v29 | 50/50 | 50/50 | Pass |
| Pixel / Division 03 / focused | v28 | 19/19 | 19/19 | Pass |
| Pixel / Division 03 / generated | v28 | 21/21 | 21/21 | Pass; no timeout in the final run |

The Pixel was disconnected at the user's request before the v29 build. The v29 changes are extraction retention and deterministic list/table routing; they passed host regressions and the complete Nothing portfolio. The final Pixel v28 generated run completed in 159.844 seconds after the plywood route fix. Earlier v27 soak evidence still records a 120-second model turn timeout and a later Binder death, so repeated-run service recovery remains a release-hardening item.

Nothing's third indexed file is `home.pdf`, not the frozen Division 03 PDF. Division evidence therefore comes from Pixel; the evaluator correctly rejects cross-fixture substitutions.

## 3. Evidence contracts — accepted

Index v29 persists and validates:

- explicit parent section IDs and parent-only descendant traversal;
- stable list IDs, item ordinals/counts, completion state, and source references;
- canonical table coordinates, row identities, header paths, full values, and fragment provenance before embedding splits;
- a physical-page record for every PDF page;
- page-stratified overview sampling and bounded section coverage;
- immutable raw-candidate, expanded-candidate, selected-context, answer-evidence, and final-citation snapshots;
- completeness and truncation status without inferring success from candidate count;
- document hash, extraction/index identity, model signature, device, thermal state, and debug override state in reports;
- source continuation retention before the minimum-size chunk filter.

## 4. Stage-aware English portfolio — accepted

The evaluator scores retrieval independently of final citations and preserves enough identity to distinguish a correct page from a correct list or table row. The final failures were diagnosed at their actual stages: HIRA was an ambiguous list route, Table 1.3 was a flattened list-row route, and the PPE condition was an extraction/chunk-retention loss.

The host gate passes `:app:testDebugUnitTest`, `:app:compileDebugAndroidTestKotlin`, `:app:assembleDebug`, and `:app:assembleDebugAndroidTest`. The suite contains **292 tests: 289 passed and 3 skipped**, with zero failures or errors.

## 5. Measured retrieval and runtime work — accepted for the current default

The corrected five-strategy benchmark contains 210 questions across English, Chinese, French, German, Italian, Japanese, and Korean. It ran on Nothing with thermal status 3 at start and end and `thermalOverride=false`. A direct in-memory cosine oracle exactly matched AppSearch dense ranking, validating the AppSearch vector query after the backend fix.

| Strategy | Hit@1 | Hit@5 | MRR | Median | p95 |
|---|---:|---:|---:|---:|---:|
| Dense only | 92.86% | 93.81% | 0.9313 | 7 ms | 11 ms |
| Lexical only | 61.90% | 68.10% | 0.6476 | 6 ms | 17 ms |
| Hybrid, keyword 0.05 | 96.19% | 96.67% | 0.9643 | 9 ms | 11 ms |
| Hybrid, keyword 0.10 | 97.14% | 97.62% | 0.9738 | 9 ms | 10 ms |
| Hybrid, keyword 0.20 | 97.14% | 97.62% | 0.9726 | 8 ms | 10 ms |

For the 0.10 hybrid, hit@5 was 100% for German, English, French, and Italian; 96.67% for Korean; and 93.33% for Japanese and Chinese. These results supersede the earlier low CJK numbers, which were produced by zero GPU embeddings. They support the seven measured languages on this synthetic frozen corpus, not a universal multilingual claim.

The production keyword weight remains **0.05**. A two-question gain on one synthetic corpus is insufficient to change the default without held-out document and language evidence.

## Thermal override record

A debug-only thermal marker was temporarily used during the earlier v28 Nothing migration after USB charging held Android at thermal status 3. It was removed immediately after migration and verified absent. The v29 migrations and all final v29 QA suites ran under the normal gate; their reports record `thermalOverride=false`. The final on-device verification also confirmed that the marker does not exist.

## Evidence locations

The durable evidence archive is `docs/evidence/generic-hard-2026-09-11/`. Its `README.md` explains scope and reproduction, `manifest.json` classifies every report, and `SHA256SUMS` protects the raw copies. It includes the final Nothing v29 and Pixel v28 reports, corrected multilingual comparison, embedding diagnostics, regression evidence, invalid mixed-mode comparison, and deliberate corpus-mismatch report.

## Point 6 entry conditions

Points 1–5 are complete, so point 6 design can begin. Before treating the extracted Aconex library as release-ready, add bounded Binder rebind/retry and repeat the generated Division bank as a multi-run Pixel soak. Keep the public API free of Apryse/LiteRT types and preserve host-owned authorization, account/project/revision identity, viewer navigation, cancellation, and capability reporting.
