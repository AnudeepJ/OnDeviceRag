# Current changes: final review and next plan

Reviewed 11 September 2026 on `generic_hard`. This supersedes the earlier point 1–5 closure notes.

## Final verdict

The approach is sound for the current fully on-device, single-PDF application, and points 1–5 are complete on index v29. The branch now treats scope, completeness, structure, embedding validity, and evaluator stages as enforceable contracts.

The final Nothing portfolio is fully green on v29: 12/12 strict construction, 23/23 broader construction, and 50/50 SafetyManual for both retrieval and answers. Pixel completed the v28 Division focused bank at 19/19 and generated bank at 21/21 before it was disconnected. The final v29 changes passed host regressions and the complete Nothing portfolio.

## What changed in the final closure

1. **Embedding correctness is checked at runtime.** The installed EmbeddingGemma model exposes one 768-float pooled output, from which the configured 512 dimensions are selected. Both physical devices returned all-zero vectors through the GPU backend. Startup now probes the actual output and permanently falls back to CPU for that installation when the vector is zero or non-finite. CPU vectors were non-zero and semantically ordered on both devices.
2. **Strategy measurements are isolated.** Dense-only omits keyword candidates and lexical-only omits semantic candidates and embedding parameters. A direct cosine oracle matches AppSearch dense results exactly.
3. **Stored evidence is complete.** Canonical table/list identity remains intact, and v29 retains lowercase short continuations after headings. This recovered the source word `impractical.` that v28 dropped from SafetyManual page 14.
4. **Ambiguous deterministic routes are bounded.** Exact-count questions reject longer complete lists, “of X” scopes a list to X, named table categories can resolve letter-labelled list rows, and standard questions select individual subject-matching evidence sentences.
5. **The evaluator identifies the failing stage.** The final device defects were independently attributed to list routing, table-row routing, and chunk retention, then covered by permanent host tests and full device reruns.

## Final device evidence

| Device | Document / bank | Index | Final observation |
|---|---|---:|---|
| Nothing A001 | `safety.pdf`, strict | v29 | 12/12 retrieval and answers |
| Nothing A001 | `safety.pdf`, broader | v29 | 23/23 retrieval and answers |
| Nothing A001 | `SafetyManual.pdf` | v29 | 50/50 retrieval and answers |
| Pixel 10 | Division 03 focused | v28 | 19/19 retrieval and answers |
| Pixel 10 | Division 03 generated | v28 | 21/21 retrieval and answers; 159.844 seconds |

All final Nothing reports record thermal status 1 at start and end and `thermalOverride=false`. The v29 migrations also completed without the temporary marker. The marker is absent from app-private storage.

The raw device reports, diagnostic runs, regression evidence, machine-readable manifest, checksums, and reproduction notes are tracked under `docs/evidence/generic-hard-2026-09-11/`.

## Retrieval decision

The corrected 210-question comparison produced 96.19% hit@1 and 96.67% hit@5 for the production 0.05 hybrid. Weights 0.10 and 0.20 both reached 97.14% hit@1 and 97.62% hit@5. The default remains 0.05 because the difference is two questions on a single synthetic corpus and has not been reproduced on held-out documents.

The corrected per-language hit@5 for the 0.10 hybrid is 100% for German, English, French, and Italian, 96.67% for Korean, and 93.33% for Japanese and Chinese. This supports only the seven measured languages and frozen corpus. It does not establish universal multilingual behavior.

## Remaining risks

- Earlier Pixel v27 soak runs recorded a 120-second model turn timeout and a later `DeadObjectException` after service recycle. The final v28 21-case run completed without either failure, but repeated-run recovery is not yet proven.
- The embedder currently uses CPU on both tested devices because their GPU delegate produced invalid zero vectors. This is correct but may affect indexing throughput. Any future GPU re-enable needs a backend-specific correctness and performance gate.
- Pixel did not receive the v29 build because it was disconnected at the user's request. The v29 behavior is covered by host tests and full Nothing runs; run a Pixel v29 smoke/soak when it is next available.
- The multilingual portfolio is synthetic and covers seven languages. Add real PDFs and each language/script before expanding support claims.

## Next plan

1. Start the point 6 Aconex extraction design around stable interfaces: host-owned authorization, account/project/document/revision identity, viewer navigation, cancellation/lifecycle, capability reporting, and isolated extraction/index/inference adapters.
2. Add bounded service rebind/retry, then run the generated Division bank repeatedly on Pixel v29 and retain every timeout/service restart.
3. Add held-out real multilingual PDFs before reconsidering the 0.05 ranking default.
4. Keep the AppSearch proposal as a decision record. Implement only items that demonstrate a reproducible workload, supported API, quality result, and rollback condition.

Do not bundle speculative AppSearch features, a default ranking-weight change, or speculative decoding into the Aconex extraction.
