# V2.2 Structured Evidence — device results

Cycle: roadmap milestones 0, 1, 2, 3 and 5 (`ROADMAP_V2_2_STRUCTURED_EVIDENCE.md`). Index version 22.
Date: 2026-09-10. Model: Gemma 4 E2B (`gemma-4-E2B-it.litertlm`) on GPU via LiteRT-LM 0.16.1;
EmbeddingGemma 300M; AppSearch 1.1.0.

## What changed

| Stage | Change |
|---|---|
| Grounding | `EvidenceValueLexer` typed grammar (NUMBER, PERCENT, FRACTION, RANGE, RATIO, MEASUREMENT with attached or spaced unit, STANDARD_ID, COMPOUND_ID) replaces the three regexes in `GroundingStreamFilter`; rejection reasons (`VALUE_ABSENT`, `UNIT_MISMATCH`, `IDENTIFIER_MISMATCH`, `CITATION`) travel in `GenerationStats.groundingReasons`; a combined citation marker with one invented id keeps its valid excerpts. |
| Outline | `Segment.Heading.kind`/`printedNumber`; `CHAPTER 13`, `Chapter XIII`, `APPENDIX A`, `Annexure II` recognised; a bare label adopts the adjacent title across a blank line; Roman-numeral, parenthesised and glyph bullets are list labels; `SectionRecord.kind`/`printedNumber`; `ManifestHealth` with quartile coverage; `INDEX_VERSION = 22`; stale namespaces of a document are swept after publish. |
| Planning | `StructuralPointer` (`chapter 13`, `chapter xiii`, `appendix a`) and titled requests (`the chapter on excavation`) resolve against nodes of that kind; chapter summaries read the whole subtree; `AnswerShape` (DEFINITION, PROCEDURE, NAVIGATION, TABLE) drives prompts, budgets and which deterministic leads may run; section pointers only for NAVIGATION wording. |
| Leads | Definition lead (verbatim defining sentence plus its criteria list), standard-reference lead (verbatim line naming `IS 3764:1992`), list lead generalised to any colon introduction, Roman/bullet labels, cross-page continuation, requested cardinality, and a prominence guard against low-ranked introductions. |
| Retrieval | Keyword terms keep numbers; prefix stems (`trench*`, `requir*`) widen recall; candidates over-fetched ×4 then re-ranked with IDF-weighted term coverage where acronyms and numbers in the question weigh double; defining sentences promoted for DEFINITION questions. |
| Overview | Top-level nodes sampled per document quartile (degraded outline falls back to page-stratified sampling); scope lines print `CHAPTER 13 — EXCAVATION`, never a synthesized `null`. |
| Runtime | `Conversation.getBenchmarkInfo()` prefill/decode tokens per second in `GenerationStats`; speculative decoding behind `files/mtp_enabled.marker` (off in these runs); thermal gate QA override `files/thermal_gate_disabled.marker` (used on the Pixel bench device only). |
| Evaluation | `safety_pdf_qa.json` red fixture (20 cases); runner selects documents by content hash; per-case stage (`RETRIEVAL`/`GROUNDING`/`ANSWER`/`PASS`), planner intent, deterministic lead, grounding reasons and runtime throughput in the JSON report; `requiredPageQuartiles` assertion for overviews; `ReindexDeviceTest` imports and re-indexes through the production service. |

Production rules contain no test-document titles, construction vocabulary, page numbers or expected values; those live only in fixtures.

## Nothing A001 (Android 16, 7.2 GB, Mali GPU) — safety.pdf, 209 pages, 1365 chunks, v22

Outline: 204 nodes, 27 of 28 chapters recognised as `CHAPTER`, 130 clauses, health not degraded, all
four quartiles covered. Index build: 1243 s (embedding ≈0.9 s/chunk on this device).

| Suite | First v22 run | Final build | Notes |
|---|---|---|---|
| `safety_pdf_qa` (20, red fixture) | 11/20 | **16/20** | 3 deterministic answers (list, standard reference, definition) at ~1.3 s; 17 generated at median TTFT 6.1 s, median completion 13.7 s, 8.2 tok/s decode, 240 tok/s prefill |
| `construction_safety_qa` (12, strict) | 8/12 | **11/12** | previous best on this document was 9/12 (v21 hardened run); retrieval 12/12 |

Fixed on device this cycle: `8m` / `150mm` / `6Mt` / `10,000 psi` compact values, `IS 3764:1992`,
`Summarize Chapter 13` and `the chapter on excavation`, document overview citing pages 3–206 with
chapter identities, HIRA five Roman-numbered steps, trench and confined-space definitions,
protective-system depth (1.2 m), risk matrix 21–25, amputation first-aid list, 48-hour training
duration and Rule 210 inquiry (list introductions now travel with their lists across pages).

Remaining failures and their stage:

| Case | Stage | Cause |
|---|---|---|
| `sp-type-b-slope` | ANSWER | Model states `1:1` but omits the depth qualifier (`20 feet (6Mt)`). Partial answer. |
| `sp-type-c-followup` | ANSWER | Follow-up retrieves page 105 but the model does not extract `1 1/2:1`. |
| `sp-ppe-when` | RETRIEVAL | Semantic ranking prefers the PPE training paragraph (p174) over the provision rule (p163). |
| `sp-emergencies-list` | RETRIEVAL | Fire-emergency list (p180) outranks the emergency-plan list (p196). |
| `cs-hierarchy-controls` | ANSWER | Elimination/Substitution/… exist only in a figure image (vision enrichment, later milestone). |

## Pixel 10 (Android 17, 12 GB) — safety.pdf and Division 03, v22

The Pixel reported thermal status SEVERE for most of the session (charging, 43 °C battery); the
indexing gate had to be bypassed with the QA marker and GPU decode varied between 2 and 7 tok/s
between runs. Its quality numbers are valid; its latency numbers are not a baseline.

| Suite | Final build | Notes |
|---|---|---|
| `single_pdf_baseline` (19, Division 03, no-regression corpus) | **19/19** | 6 deterministic, 13 generated; median TTFT 3.7 s |
| `generated_live_qa` (21, Division 03) | **21/21** answers, 20/21 pages | `live-form-remove-75` gives the correct `75 percent` answer citing p47 instead of p10 |
| `construction_safety_qa` (12) | **10/12** | `cs-hierarchy-controls` (figure image); `cs-firstaid-amputation` summary omitted "plastic bag" on this run |
| `safety_pdf_qa` (20) | **14/20** (+2 re-run) | same four misses as the Nothing plus `sp-ceo-refusal` and `sp-confined-space-definition`, both of which pass after two scorer/filter fixes below |

Two Pixel-only misses were artefacts rather than pipeline defects and were fixed in the same
build: the refusal detector did not recognise "do not contain", and the model suffixed list letters
to excerpt ids (`[E5a]`), which the filter now maps back to `E5`. Both cases pass on re-run.

Between builds the Division 03 corpus briefly dropped to 18/19 (`summary-03300-curing` timed out at
120 s while streaming a correct answer at 2 tok/s under throttling) and the generated set to 20/21
(`live-03310-liquid-ratio`: the `0.45` list line was retrieved first but its introducing sentence on
the previous page was dropped by context selection). The second was a real defect and is what
introduced the `INTRODUCTION` neighbour kind; both suites are green on the final build.

## Milestone 4 — Tables (index v23)

Device: Nothing A001 only. Re-index 1160 s, 209 pages, 1364 chunks, 10 tables, outline unchanged
(204 nodes, 27 chapters, not degraded). Search-time candidate window stays ×4; ×8 briefly pulled
the incident-investigation list over HIRA and was reverted.

| Suite | v22 final | v23 M4 | Notes |
|---|---|---|---|
| `safety_pdf_qa` | 16/20 | **19/23** | 3 new table cases added; all 4 table questions (`21-25`, `13-20`, Possible, Fatal) answered by `TABLE_ROW_LEAD` in 1.1–1.2 s. Same four non-table misses as v22. |
| `construction_safety_qa` | 11/12 | **11/12** | `cs-risk-matrix-level` is now `TABLE_ROW_LEAD` at 1.2 s; `cs-hierarchy-controls` still a figure image. |

This v23 index stored tables without captions because uppercase `TABLE 5.1 …` lines were classified
as headings and stripped before clustering. `StructureAnalyzer` now leaves caption lines in the
page; the next reindex will populate `TableRecord.tableNumber` / caption. Row-key leads already
answer from the grid without that identity.

## Milestone 6a — Golden measurement (Nothing A001, index v23)

Date: 2026-09-10. Same model/backend as above. External datasets (ITP / drawing schedule) and Pixel
confirmation are **6b**, not this run.

Reindex of the three on-device PDFs, then a caption/list follow-up reindex of `SafetyManual.pdf`
only (255 chunks, still degraded outline: one top-level node). Lab captions now include
`Table (1.1)`, `Table 1`, `Table 4` and `Table 5`. Construction `TABLE 5.1` grids on pp.10–12 and
most Division 03 grids remain uncaptioned; row-key leads do not need the number.

Generic fixes landed before the scored run: parenthesized captions (`Table (1.1)`), parenthesized
score cells no longer stolen as lists, explicit table ids restrict the row lead, a unique row with
several named columns is answered as one row (`Class D` on the fire table), and a bare `Class`
header row is not treated as a row key (that false lead had broken plywood).

| Suite | Result | Notes |
|---|---|---|
| `safety_manual_qa` (50, lab; 17 TABLE) | **39/50** answers, 45/50 retrieval; **9/17 tables** | v21 was 40/50 and 8/17 tables. Median generated TTFT 6.7 s, 9.4 tok/s. Table leads: fire Class D, laser Class 4, laser Table 5 distances. Risk-matrix 25 is generated from page 10 after `Table (1.1)` resolved. |
| `single_pdf_baseline` (19, Division 03) | **17/19** answers, 19/19 retrieval | First Nothing v23 measurement (Pixel v22 was 19/19). `table-slump` cites p62 but omits the 1–3 in superplasticizer range. `script1-curb-level` is correct and cites p49 but emits `\pm` (scorer flags raw LaTeX). Plywood recovered after the generic `Class` row-key guard. 3 table leads. |
| `generated_live_qa` (21, Division 03) | **19/21** answers, 20/21 pages | `live-form-remove-75` still answers `75 percent` from p47 instead of p10. `live-overview` is a grounding/citation-marker reject. Plywood class recovered. |
| `safety_pdf_qa` (23) | **19/23** | All 4 table cases stay `TABLE_ROW_LEAD`. Same four non-table misses as M4 (`sp-type-b-slope`, `sp-type-c-followup`, `sp-ppe-when`, `sp-emergencies-list`). |
| `construction_safety_qa` (12) | **11/12** | `cs-risk-matrix-level` stays `TABLE_ROW_LEAD`; `cs-hierarchy-controls` is still a figure. |

Lab TABLE misses left on the board (not 6b work):

| Case | Stage | Cause |
|---|---|---|
| `safety-risk-matrix-possible-moderate` | RETRIEVAL | Score `9` lives in a split matrix; reconstruction still drops that cell. |
| `safety-severity-four` | ANSWER | Model reads list item 3 (Moderate) instead of 4 (Major / serious injury). |
| `safety-extreme-risk-response` | RETRIEVAL | `Table (1.4)` is still an unnumbered grid on pp.11–12; the model refuses. |
| `safety-engineering-controls` | ANSWER | `redesign` / `Isolation` are on p13; the model lists guarding/enclosures instead. |
| `safety-ppe-table-eye-seal` | ANSWER | Lead hits the Protection×Eyes cell; column titles lost `Splash Goggles`. |
| `safety-ppe-table-face-coverage` | RETRIEVAL | Cites the p18 prose recommendation, not Table 1 on p19. |
| `safety-fire-class-e` | ANSWER | Electrical row retrieved on p31; answer names powder / CO₂ but not `Class E`. |
| `safety-laser-class-3b` | RETRIEVAL | Table 5 distance row on p34 outranks Table 4 on p33 (`5-500 mW`, `Required`). |

Non-table lab misses unchanged: `safety-mechanical-hazards` (grounding/citation), `safety-ppe-last-control` (`impractical`), `safety-housekeeping` (`immediately`).

## Known limitations carried to 6b / later

- Paraphrase recall: a chunk whose wording differs strongly from the question can fall outside the
  48 fetched candidates; the IDF-weighted, anchor-boosted re-rank only reorders what AppSearch
  returns. A larger candidate window or a second embedding query is the next lever.
- Figure content (hierarchy of controls) needs the vision enrichment milestone.
- Conditional-value lead is implemented; `sp-type-b-slope` still dropped `20 feet (6Mt)` on this
  Nothing run (lead did not fire; model kept `1:1` only).
- Unbordered / parenthesized risk matrices still split cells (`Catastrop` / `hic`, Possible×Moderate
  `9`). Caption numbers in parentheses now attach when the line is near the grid.
- Speculative decoding was left off (`mtp_enabled.marker` absent); measure it on a cool device.
- 6b: an ITP/QA-QC with dense tables, a scanned drawing schedule, and Pixel confirmation.

## Reading the reports

Each run writes `single_pdf_baseline-<suite>.json` to the app's external files directory with, per
case: `stage`, `plannerIntent`, `answeredBy`, `groundingReasons`, `retrievedPages`, model TTFT,
prefill tokens and tokens/s, decode tokens/s, and the full answer. Suite totals include `byStage`
and median TTFT / decode / prefill so a regression names its stage and its phase.
