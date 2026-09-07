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

## Nothing Phone follow-up — 2026-09-07

The same indexed PDF and pinned E2B/EmbeddingGemma artifacts were validated on a Nothing A001
(7.2 GB usable RAM, Android API 36, Mali GPU). Indexing produced the same 79 pages, 716 chunks, and
166-section V21 manifest. The full index took about 9 minutes 57 seconds; no thermal pause was
triggered, although the device reached light thermal status near completion.

Real UI testing found that the earlier immediate summary preview exposed raw chunk fragments before
the generated answer, duplicated facts, and made visible latency look faster than the actual answer.
That preview is now restricted to exact table/fact evidence. Generated section summaries start with
model output, use a five-bullet/80-word target, and have 192 output tokens of headroom. Retesting
`Summarize specification 03300 section 2.05 concrete mix` produced five complete page-17 bullets
(cement, both strengths, both slump ranges, and water/cement ratio), with no raw preview and no
cut-off sentence. Retrieval took 161 ms; model/visible TTFT was 8.268 seconds on the warm GPU.
`Summarize concrete mix` still returned the two-section clarification immediately.

The E2B inference process used about 2.89 GB PSS (2.68 GB RSS and 273 MB swap PSS) after generation.
The low-memory device gate therefore now applies to debug and release builds: on 6–8 GB devices,
chat is unloaded/blocked while indexing so the LLM and embedding/indexing workloads are not resident
together.
