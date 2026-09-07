# Single-PDF V2.1 Validation — Pixel 10

## Test environment

- Device: Google Pixel 10, USB connected
- Inference: Gemma E2B on GPU; processing entirely on device
- Test PDF: Technical Specifications Division 03, 79 physical pages
- Index: V21 manifest, 716 chunks, 166 structural sections
- Date: 2026-09-06
- Suite: 15 real questions executed sequentially through `AiInferenceService`
- Full-suite runtime: 255.477 seconds
- Android runtime crashes: none

The repeatable cases are stored in
`app/src/androidTest/assets/single_pdf_baseline.json`. The test writes its latest JSON report to the
app's external files directory as `single_pdf_baseline.json`.

## Result

**Retrieval: 15/15. Answer checks: 15/15.**

The previous V1 baseline was 5/15 acceptable answers. V2.1 now passes the duplicated-section,
ordered-summary, continuation, table, numeric-grounding, refusal, and follow-up cases.

| Coverage | Verified behavior |
|---|---|
| Ambiguous title | `Summarize concrete mix` identifies specification 03300/page 17 and specification 03310/page 60 and asks which one the user means. |
| 03300 Concrete Mix | Page 17 summary and facts include 0.50, 2400/4000 psi, both slump ranges, air, and admixture constraints. |
| 03310 Concrete Mix | Pages 60–62 summary includes testing-laboratory requirements, 0.45/0.40/0.55 ratios, and 4–6% air. |
| Curing | Ordered page 44–45 summary, membrane-curing requirements, mass-concrete period, and page 45–46 freezing follow-up pass. |
| Tables | Page 62 slump relationships and page 63 Class D / 5,000 psi / 658 lb per cubic yard remain associated. |
| Grounding | Truncated values, compound IDs, units, specification IDs, and citations are checked incrementally while output streams. |
| Unsupported question | The handrail-color question safely refuses and exposes no irrelevant citations. |

## Real-device latency

For the 15-case full run:

| Metric | Median | Worst observed |
|---|---:|---:|
| Model time to first token | 4.195 s | 8.390 s |
| User-visible first text | 3.339 s | 8.404 s |
| Completion | 15.109 s | 34.172 s |

Deterministic ambiguity resolution completed in 315 ms. Targeted post-suite checks also verified:

- specification 03300 / section 2.05 / ratio 0.50 with no masked identifier;
- the page 62 slump table with only its relevant row group;
- the page 63 Class D row with 5,000 psi and 658 pounds per cubic yard.

## Architectural conclusion

The single-PDF foundation is ready for broader document testing. Production behavior is generic:
there are no hard-coded answers, page numbers, section numbers, or document-specific retrieval
rules. PDF-specific facts live only in the repeatable evaluation fixture. The implementation uses
structural manifests, direct ordered section fetch, hybrid retrieval, conservative neighbor
expansion, table-aware evidence leads, and a manifest-scoped compatibility fallback for older
AppSearch indexes.

## Nothing Phone final validation — 2026-09-07

### Exact environment and flow

- Device: Nothing A001, Android 16 / API 36, `arm64-v8a`, 7.2 GB usable RAM, Mali GPU.
- Build fingerprint: `Nothing/GalagaIND/Galaga:16/BP2A.250605.031.A3/2608121729:user/release-keys`.
- Source base: `997ff3d` plus the V2.1 working-tree changes under review.
- Debug APK: 312,683,440 bytes, SHA-256
  `c20ed56500df2ece5d37f978b41078af98496c3f7f161d53308164a74a760aba`.
- Test PDF: Technical Specifications Division 03, 79 physical pages.
- Retained on-device index: V21, 560 chunks, 164 structural sections. The APK was installed with
  `adb install -r -t`, so this run exercised the existing index rather than re-indexing.
- Flow: bind the real `AiInferenceService`, load the installed E2B model on GPU, then run all cases
  sequentially against the one indexed PDF. Follow-up cases pass the prior answer and source-section
  metadata through the same Binder interface used by the app.
- Run count: 19 real questions in one 197.739-second instrumentation run; 12 used model generation
  and 7 used deterministic grounded resolution.

### Result

**Retrieval: 19/19. Answer checks: 19/19. Grounding failures: 0. Crashes/restarts: 0.**

Every turn reported the GPU backend. Generated turns averaged 5.913 seconds to the first model token,
14.896 seconds to completion, and 10.46 approximate tokens/second. Deterministic turns averaged
1.010 seconds. Across all 19 turns, median completion was 8.003 seconds and the worst completion was
31.509 seconds; long section summaries are therefore still the main latency limitation.

Script 1 passed 4/4 with real conversation history:

| Turn | Verified answer | Completion |
|---|---|---:|
| Piers from plumb | `1/2" in 10'` | 1.145 s |
| Column cross-section | `+1/2", -1/4"` (equivalent `±1/2", -1/4"` source form) | 1.121 s |
| Curb top level | `3/16" in 10'` | 1.110 s |
| Testing laboratory | Section `01450` | 1.092 s |

Manual review also confirmed that the 03300 curing summary no longer turns heading `3.26` into a
measurement, the 03310 summary always preserves the exact `0.45`/`0.40`/`0.55` ratio block, and the
slump answer no longer exposes raw diagnostic context. The ambiguous `Summarize concrete mix` query
returns both valid candidate sections instead of guessing.

The final thermal snapshot reported Android thermal status 1 (light), skin 42.291 C, SoC/GPU sensor
50.248 C, battery 39 C, and no active Mali cooling-device throttling. This was a deliberately
sustained sequence following several focused runs, not an isolated cold-start benchmark. The vendor's
cached 71.774 C CPU/GPU entries are stale/static; the HAL current-temperature entries are the useful
measurements.

Evidence artifacts for this run:

- `/private/tmp/codex-chat-suite-nothing-20260907/final-19of19-single_pdf_baseline.json`
- `/private/tmp/codex-chat-suite-nothing-20260907/final-19of19-logcat.txt`

The E2B inference process previously measured about 2.89 GB PSS (2.68 GB RSS and 273 MB swap PSS)
after generation on this device. The low-memory gate applies to debug and release builds: on 6–8 GB
devices, chat is unloaded/blocked while indexing so the LLM and embedding/indexing workloads are not
resident together.
