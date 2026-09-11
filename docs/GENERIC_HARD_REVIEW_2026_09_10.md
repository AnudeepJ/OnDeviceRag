# generic_hard review — 2026-09-10

Review target: `61016ea`, compared with `main` at `a7bc443`. The branch contains two commits, 39 changed files, 3,715 insertions and 204 deletions. The working tree was clean at review start. This review makes no production changes.

## Assessment

Keep the architectural direction, but do not treat V2.2 as acceptance-complete or extract it into the Aconex library yet. The branch materially improves representation and source-derived answers. Four independently reproduced correctness defects show that its new deterministic answers still lack sufficient identity and completeness guarantees.

The linked task, **MainTask** (`01a07015-d445-7792-9ff9-ddbfb99f180c`), establishes the intended progression: fully on-device, single-PDF answers; Apryse extraction with ML Kit fallback; fast source-derived answers where possible; evidence-grounded generation otherwise; eventually an Aconex library using the host's existing viewer. It also establishes that generalization beyond the CSI corpus is the immediate problem, and that the vision spike did not justify replacing OCR with Gemma Vision.

## Confirmed correctness findings

### P1 — A definition can change a numeric requirement

Location: `AnswerQuestionUseCase.kt:673-675`, `buildDefinitionLead` at `605-628`.

The definition regex terminates at the first period, semicolon or colon, including the decimal point inside a number. With source `A widget is a component measuring 1.5 metres in length.`, asking `What is a widget?` produces:

```text
A widget is a component measuring 1. [Page 1]
```

This is returned as decisive and bypasses generated-answer grounding. It changes the source value and drops its unit. A semicolon can similarly terminate a definition before additional criteria.

Fix: preserve source sentence boundaries without splitting decimals, identifiers or abbreviations; distinguish a complete defining span from an introduction requiring further criteria. Only return a decisive definition after validating those boundaries and completeness. Add decimal, abbreviation, semicolon, long-definition and cross-page criteria regressions.

### P1 — Explicit table identity can lose to an unrelated caption

Locations: `AnswerQuestionUseCase.kt:1181-1200` (`mergeResolvedTable`) and `:763-766` (`buildRowKeyLead`).

The planner resolves a table ID, but direct fetch merely boosts its rows and merges unrelated hybrid hits. The answerer receives no enforced table scope. It chooses by caption/header word overlap and does not use that score boost to resolve scope.

Reproduction: Table 5.1, caption `Capacities`, contains Class D → Limit 10 and has score 3.0. Table 5.2, caption `Widget load limit`, contains Class D → Limit 99 and has score 1.0. `In Table 5.1 what is the widget load limit for class D?` returns:

```text
D — Limit: 99 [Page 1]
```

Fix: pass resolved table identity into answer resolution as a hard constraint. Apply the constraint before every deterministic lead, including the earlier list and flattened-table paths, and carry it into generated context. Preserve ambiguity when a printed number names multiple tables; do not silently broaden an unresolved explicit scope. Test competing tables with identical row labels, captions with stronger lexical overlap, and table/prose distractors.

### P1 — Invented citation IDs can become valid citations

Location: `GroundingStreamFilter.kt:81-83`.

The compatibility rule intended for `[E5a]` removes any trailing character, rather than only a permitted letter suffix. When only E1 exists, `[E10]` becomes E1 and emits `[Page 1]` without setting `hadGroundingFailure`.

Fix: recognize an explicit alphabetic suffix grammar and require an exact existing base ID. Reject nonexistent numeric IDs. Also decide how partial markers are scored: the new `CITATION_PARTIAL` reason currently does not set the failure flag, so an invented member of a combined citation can be hidden from pass/fail scoring.

### P1 — An unfinished list is marked complete

Location: `AnswerQuestionUseCase.kt:893-915`.

The list walker stops when the next chunk is absent from the retrieved candidate map, or after four chunks. It still returns `decisive = true` unless the user supplied a recognized numeric count that exceeds the retrieved items. It does not prove the list ended.

Reproduction: a setup introduction followed by one LIST chunk with `continuesToChunkIndex = 3`, while chunk 3 is absent, answers `List all widget setup steps` with only `1. Start the widget.` as a decisive answer.

Fix: resolve list evidence through the manifest until a verified end, expected cardinality or explicit budget limit. Treat missing continuation and truncation as partial evidence. Never present a bounded prefix as the complete list. Longer term, persist list identity and item ordinals rather than inferring list ownership solely from adjacent chunk numbers.

### P2 — The benchmark does not enforce its requested content hash

Location: `SinglePdfBaselineDeviceTest.kt:130-135`.

If a requested/default hash is absent, selection falls through to display name. A different revision named `safety.pdf` can therefore run against the original answer key. The new exact-hash contract is not enforced.

Fix: when a hash is provided, fail if it is missing. Use name selection only when no hash is specified. Validate index/extraction identity as well, and include build commit, model signature, extraction fingerprint and thermal/override state in run reports.

## What the branch achieves

| Area | Implemented change | Assessment |
|---|---|---|
| Grounding | Typed value lexer, compact units, fractions/ratios, identifiers, rejection reasons | Useful improvement; streaming validation remains a separate scanner and needs adversarial equivalence tests. |
| Outline | Heading kind and printed number, chapter/appendix parsing, adjacent titles, manifest health | Solves important observed cases; not yet the explicit parent/child outline proposed in the design. |
| Planning | Structural pointers, table resolution, definition/navigation/procedure shapes | Good separation of answer shape and retrieval intent; explicit scope needs enforcement downstream. |
| Retrieval | Fourfold candidate overfetch, prefix expansion, candidate-local IDF reranking, definition promotion and list-introduction neighbors | Reasonable bounded improvements; recall outside the candidate window remains unresolved. |
| Tables | Wrapped-cell heuristics, captions and IDs, row-qualified embedding text, table manifest, row/cell leads | Valuable foundation for simple grids; not yet the full table model. |
| Index lifecycle | v23 namespace, table fields across persistence/citations, stale namespace sweep after publication | Maintains the important publish-before-cleanup order. Requires reindexing and migration validation. |
| Runtime | Per-turn prefill/decode measurements, speculative-decoding flag, thermal test override | Instrumentation is implemented; performance acceptance and experimental feature evaluation remain open. |
| Evaluation | New regression fixture and named suites, import/reindex runner, coarse failure stages | Better repeatability; stage isolation and strict corpus identity remain incomplete. |

## Gaps between implementation and the design

**Outline structure is still implicit.** `SectionRecord` has no parent ID; descendants are inferred from path strings. Parser levels remain fixed: SECTION/CHAPTER at 1, PART at 2. A conventional Part → Chapter hierarchy cannot be represented correctly by that ordering. Repeated identical heading paths can also mix descendant scopes. Make hierarchy explicit and test both specification-style Section → Part and book-style Part → Chapter structures.

**Manifest health is a diagnostic, not the proposed indexing gate.** Its coverage check uses the last represented page rather than independent PDF page count, and it is principally used by overview sampling. Degraded overview fallback still samples titled sections; it cannot recover pages with no recognized headings. Validate source-page coverage against the actual PDF and retain page-level fallback candidates independent of outline quality.

**Long summaries can omit the end before selection starts.** Subtree retrieval takes the first 80 chunk IDs. The selector cannot preserve breadth from discarded chunks. Allocate a bounded sample across descendants/page ranges, and propagate truncation explicitly.

**Tables are still Markdown-based.** `Segment.Table` stores one header string and row strings. There are no canonical cell coordinates, row/header spans, multi-level header paths, confidence or cell source anchors. `TableIdentity.id` includes page and local ordinal, so a continued table on another page gets another identity. This limits multi-page, merged-header and conditional-value questions. The branch implements an initial table milestone, not the complete model in `STRUCTURED_PDF_RAG_V2_2_PLAN.md`.

**Deterministic answers need a shared result contract.** `AnswerQuestionUseCase` now combines orchestration, matching and formatting in over 1,300 lines. The lead order can decide the answer before later handlers inspect the same question. Extract focused answerers returning scope, source spans, completeness, ambiguity and truncation status, then let one policy decide whether generation can be skipped. Copying text is insufficient if the wrong row was selected or the span was cut.

**The evaluator cannot yet identify the first failing pipeline stage.** `RETRIEVAL` is inferred from pages delivered through the citation callback, not a separate raw-retrieval trace. Extraction, retrieved candidates, selected context and final cited evidence are not measured independently. A correct page containing a wrong row can pass retrieval; missing extraction can be labeled answer failure. Introduce explicit stage snapshots and gold spans/row identities rather than relying only on page and phrase matching.

**Generic does not yet mean multilingual.** Intent/heading patterns and prefix suffixes remain English-oriented. Corpus frequency can help term weighting, but cannot supply multilingual intent detection or fix tokenization. Protect identifiers, units and discriminative terms; validate each supported language rather than claiming universal behavior from a frequency threshold.

## Device evidence and milestone status

Fresh Pixel runs used exact content hashes for all three PDFs and confirmed index v23 before executing 125 prompts:

| Run | Retrieval | Answer | Result |
|---|---:|---:|---|
| Division 03 focused | 19/19 | 19/19 | Passed. |
| Division 03 generated | 20/21 | 21/21 | The answer was correct from page 47; the fixture expects the duplicate requirement on page 10. |
| Construction safety red bank, after fix | 21/23 | 21/23 | Type B and Type C now pass; PPE policy and emergency-category recall remain. |
| Construction safety strict | 12/12 | 10/12 | Two retrieved answers omitted required details. |
| Laboratory safety manual | 48/50 | 38/50 | Dense-table and completeness failures remain the largest quality gap. |

The Type B failure was traced to layout extraction splitting its heading (chunk 721) from its value block (chunk 722), while Type A and Type C occupied adjacent pairs. `buildConditionalValueLead` now binds a unique typed heading only to its immediate same-page successor and retains the existing property/condition checks. Type B (`1:1`, 20 feet/6Mt) and the Type C follow-up (`1 1/2:1`) both passed targeted Pixel reruns and the subsequent full red bank via `CONDITIONAL_VALUE_LEAD`, with page 105 citations and no grounding failure.

The roadmap's statement that milestones 0–5 are implemented should be split into **implemented**, **validated**, and **accepted**. Milestone 5's constrained-decoding evaluation and cool-device performance gates are not demonstrated. Milestones 1–4 still have correctness and representation gaps. Adding the golden portfolio is appropriate now as a regression gate, while those gaps are fixed.

## Review of the AppSearch/multilingual recommendations

Treat `APPSEARCH_AND_MULTILINGUAL_RECOMMENDATIONS.md` as proposals requiring correction and measurement, not an approved implementation sequence:

- It describes candidate factor 8 / 64 results; current code uses factor 4 with default topK 12, yielding 48 unscoped candidates.
- Property weighting is a plausible experiment, but measure its interaction with the current combined semantic/relevance ranking and overlapping indexed text fields before choosing weights.
- Dropping bodies before reranking changes the algorithm: current reranking depends on body terms. A header-only pass can discard the decisive evidence before bodies are fetched.
- The proposed same-page follow-up retention must not become unconditional sticky context. Preserve subject and scope, then measure candidate and selected-context recall with neighboring distractors.
- The quoted allocation/latency/indexing improvements are not measurements established by this branch. Benchmark them before accepting them as expected gains.
- Validate suggested AppSearch features against the shipped dependency before scheduling implementation. The document does not establish availability or measured benefit.

## Recommended road ahead

1. **Correctness closure.** Fix the four reproduced production defects and strict hash selection. Promote the temporary reproducers into permanent tests, add competing-evidence variants, and make completeness/scope explicit in deterministic results.
2. **Reindex and rerun the final v23 build.** Run Division 03, construction safety and the 50-case laboratory manual on both devices. Verify caption identity after reindex. Report every failure; retain figure-only questions as visible extraction gaps.
3. **Evidence contracts.** Add explicit outline parents, list identities/end markers and canonical table rows/cells/header paths with provenance. Implement genuine page-based degraded fallback and bounded subtree coverage. Version the index deliberately when persisted semantics change.
4. **Stage-aware portfolio.** Capture extraction → structure → candidates → selected context → answer → citations separately. Expand held-out document styles, conditional values, follow-ups, negative premises and repeated row labels. Keep runtime comparisons on cool devices with model/build/index identity recorded.
5. **Measured retrieval/runtime work.** Address the documented PPE and emergency-list recall misses and Type B/Type C conditional answers using stage traces. Compare candidate strategies on a frozen portfolio. Keep speculative decoding experimental until it demonstrates benefit; keep thermal overrides debug/test-only and report their use.
6. **Aconex library extraction.** Begin after these contracts stabilize. Preserve host-owned authorization, account/project/revision identity and viewer navigation. Separate public API, index/extraction adapters and isolated inference runtime. Define unsupported-ABI capability behavior before packaging; avoid leaking Apryse/LiteRT types into the public API.

## Verification performed

- Full `:app:testDebugUnitTest`, `:app:assembleDebug`, `:app:compileDebugAndroidTestKotlin`, and `git diff --check`: passed.
- The updated APK was installed over the existing Pixel app without clearing its indexed documents.
- All five live QA banks ran through the production inference service. The final acceptance reports are retained in the tracked evidence archive under `docs/evidence/generic-hard-2026-09-11/accepted/`.
- The exact Type B and Type C regressions passed targeted reruns, followed by a full 23-case post-fix red-bank run.
- The unrelated pre-existing `ModelManagerScreen.kt` worktree edit was left untouched.

## Final closure update — index v29, 11 September 2026

Points 1–5 are now closed with permanent tests and physical-device evidence. Nothing A001 was fully reindexed to v29 and passes 12/12 strict construction, 23/23 broader construction, and 50/50 laboratory questions for both retrieval and answers. These final runs used the normal thermal gate, reported thermal status 1, and recorded `thermalOverride=false`.

The closure uncovered and fixed four additional root causes. First, the GPU delegate returned all-zero embeddings on both tested devices; startup now validates the chosen output and falls back to CPU. Second, the strategy benchmark now isolates dense and lexical candidate generation. Third, exact-count list routing uses the requested subject and rejects longer complete lists rather than truncating them. Fourth, short lowercase source continuations after headings are retained, recovering `impractical.` from SafetyManual page 14; named letter-labelled table rows also receive a bounded deterministic route.

The corrected 210-question comparison reports 96.19% hit@1/96.67% hit@5 for hybrid 0.05 and 97.14%/97.62% for both 0.10 and 0.20. The production default remains 0.05 because the measured difference is two questions on one synthetic corpus. Japanese and Chinese each reach 93.33% hit@5 with hybrid 0.10, superseding the earlier figures produced with zero embeddings while still falling short of a universal multilingual claim.

Pixel passes the v28 Division focused bank at 19/19 and the final generated bank at 21/21 without a timeout. The user disconnected Pixel before v29, so Pixel v29 remains a smoke/soak follow-up. Earlier v27 timeout and Binder-death evidence remains relevant until repeated-run service recovery is demonstrated.

The complete host gate passes with 292 tests, three skipped and zero failures/errors. See `ROAD_AHEAD_1_TO_5_ACCEPTANCE.md` for the accepted contracts, `CURRENT_CHANGES_REVIEW_AND_NEXT_PLAN.md` for the next work, and `docs/evidence/generic-hard-2026-09-11/` for the tracked raw reports, manifest, checksums, and reproduction notes.
