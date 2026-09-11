# AppSearch and Multilingual Proposal Review

Status: Reviewed proposal as of 11 September 2026. This document is not an implementation plan or an acceptance source. No proposal below is approved merely because points 1–5 are complete.

The ideas below were reviewed against the current `generic_hard` implementation and its frozen device portfolio. Only changes with a clear correctness contract and measured benefit should enter the roadmap.

## Decision record

| Proposal | Verdict | Reason and required evidence |
|---|---|---|
| AppSearch property weights | **Experiment only** | The current Kotlin reranker uses body, section, table, identifier, and semantic signals together. Native weighting may improve caption queries but can regress ordinary body matches. Compare it on the frozen multilingual and three-PDF portfolios before changing the default. |
| Header-only first-stage projection followed by body fetch | **Reject as written** | The current reranker needs body and retrieval text to compute coverage and identity. Projecting only headers removes ranking evidence and adds a second AppSearch fetch. Reconsider only with a lightweight ranking representation that preserves every signal, plus allocation and latency traces. |
| Numeric page-range filtering | **Defer** | Page scoping already exists in the persisted schema, but the proposal does not demonstrate a failing workload or verify the required query features on supported devices. Add a scoped-query benchmark and feature check first. |
| Increase flush interval to 1,500 | **Defer and benchmark** | The stated baseline is wrong. `IndexPdfUseCase` sends 100-document calls and `putChunks` flushes at the end of every call, so effective persistence is currently every 100 chunks. A publication-only flush may be safe because indexing uses a staging namespace, but its crash behavior and device-time benefit must be measured before adoption. |
| Upcoming ANN, native RRF, zero-copy embeddings | **Watchlist only** | These are speculative capabilities, not commitments for the current AndroidX/API floor. Adopt only after the exact API is available, feature-detected, and measured on supported devices. |
| Statistical document stopwords at a fixed 65% threshold | **Reject as a default** | High-frequency domain terms such as `shall`, `safety`, or `contractor` can carry user intent. A universal fixed threshold can erase useful sparse evidence. Test adaptive stopwords as a retrieval strategy on held-out languages and documents; retain the existing path until it wins without per-language regressions. |
| Dense retrieval is universally language-independent | **Reject as an acceptance claim** | Multilingual model capability does not guarantee equal recall by language, script, document style, or OCR quality. The current frozen portfolio covers English, Chinese, French, German, Italian, Japanese, and Korean; it does not justify claims about Spanish, Indic, Arabic, Cyrillic, or Greek. Add each claimed language to the portfolio first. |
| Sticky previous-page context for follow-ups | **Reject as written** | Unconditional page stickiness can inject unrelated evidence. The Type B/Type C failure was fixed by preserving adjacent conditional-value chunks and their provenance, which addresses the observed evidence boundary directly. |
| Fixed 15 ms delay at moderate thermal state | **Reject** | A fixed delay has no demonstrated relation to temperature recovery or throughput. Keep the existing thermal gate, run performance tests on a cool device, and record thermal state and any override. |
| New compact citation IPC type | **Defer** | IPC already strips citation text. Add a second Parcelable and lazy-fetch path only if Binder size or latency measurements show a real problem. |

## Accepted contracts from this review

The useful direction is expressed as testable contracts rather than predicted performance:

1. Persist explicit section parents, list identity and completion, canonical table cells with header paths and source provenance, and every physical page.
2. Keep retrieval, structural expansion, selected context, answer evidence, and final citations separately observable.
3. Compare ranking changes on a frozen portfolio and report aggregate and per-language quality with latency.
4. Keep thermal and model-throughput overrides limited to debug builds and record their use in reports.
5. Preserve staging publication and version the index whenever persisted evidence semantics change.

## Evidence required before reconsidering deferred work

A proposal may move into implementation only when it includes:

- a reproducible failing or expensive workload;
- a supported-device feature check;
- before/after quality by document and language;
- median and tail latency, allocation or storage evidence relevant to the claim;
- index, app build, embedding model, device and thermal identity;
- a rollback condition when quality regresses.

No percentage latency, memory, or indexing improvement in the earlier proposal was backed by a benchmark, so those estimates are intentionally removed.

## Measured disposition after points 1–5 closure

The first five-strategy run was invalid because “dense-only” and “lexical-only” still submitted the combined AppSearch query and only zeroed a later rerank weight. The harness now isolates candidate generation as well as reranking. That invalid artifact is retained at `docs/evidence/generic-hard-2026-09-11/invalid/retrieval-strategy-comparison-mixed-modes.json` and must not be used for decisions.

The corrected 210-question physical-device run on Nothing produced:

| Strategy | Hit@1 | Hit@5 | MRR | Median | p95 |
|---|---:|---:|---:|---:|---:|
| Dense only | 92.86% | 93.81% | 0.9313 | 7 ms | 11 ms |
| Lexical only | 61.90% | 68.10% | 0.6476 | 6 ms | 17 ms |
| Hybrid 0.05 | 96.19% | 96.67% | 0.9643 | 9 ms | 11 ms |
| Hybrid 0.10 | 97.14% | 97.62% | 0.9738 | 9 ms | 10 ms |
| Hybrid 0.20 | 97.14% | 97.62% | 0.9726 | 8 ms | 10 ms |

The prior low dense/CJK result came from a real backend defect: the GPU delegate returned all-zero embeddings on both tested devices. A direct cosine oracle matched the bad AppSearch dense order, proving that AppSearch was ranking the vectors it received correctly. The embedder now probes its selected 512-dimensional output, records a GPU-disable marker for invalid output, and recompiles on CPU. CPU vectors were non-zero and semantically ordered on Nothing and Pixel.

With valid embeddings, the 0.10 hybrid reached 100% hit@5 for German, English, French, and Italian, 96.67% for Korean, and 93.33% for Japanese and Chinese. This supersedes the old CJK figures, but still covers only seven languages on a synthetic frozen corpus.

The production default remains 0.05. The 0.10/0.20 gain is two questions in one corpus, which is insufficient for a default change. Reconsider only after a held-out real-document portfolio shows a repeatable gain without a language or document regression.
