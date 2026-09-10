# Roadmap V2.2 — Structured Evidence

Status: active. Supersedes the ordering in `GENERIC_RAG_HARDENING_PLAN.md` and
`STRUCTURED_PDF_RAG_V2_2_PLAN.md`; both remain the detailed design references. This document is the
single sequence the team builds against.

## Why one roadmap

V2.1 reached 19/19 on the CSI Division 03 specification. The same pipeline dropped to 40/50 on a
laboratory safety manual (8/17 on tables) and to 10 clean / 5 partial / 6 failed on 21 live questions
against the 209-page construction safety manual. Every miss maps to a stage contract shaped by one
document family: value grammar, heading grammar, list grammar, overview sampling, table extraction.

Three plans proposed three next steps (tables first, hardening first, library extraction first) and
two of them bump the index version independently. This roadmap merges them into one index bump and one
milestone order. Production rules never contain a test-document title, construction keyword, page
number, or expected value; those live only in fixtures.

## Milestones

| # | Milestone | Deliverable | Exit criterion |
|---|---|---|---|
| 0 | Lock evidence | `safety_pdf_qa.json` red fixture from the live failures; device runner selects the document by content hash; report names the failing stage (retrieval / grounding / answer) | Every live failure reproduces from an instrumentation run without a chat screen |
| 1 | Typed grounding | `EvidenceValueLexer`: NUMBER, PERCENT, FRACTION, RANGE, RATIO, MEASUREMENT (compact or spaced unit), STANDARD_ID, COMPOUND_ID. One grammar shared by `GroundingStreamFilter` and the deterministic leads | `8m`, `150mm`, `6Mt`, `1½:1`, `IS 3764:1992`, `10,000 psi` survive at every stream split; synthetic unsupported values stay blocked |
| 2 | Outline v22 | Heading kinds (PART, CHAPTER, SECTION, CLAUSE, APPENDIX, HEADING) with adjacent-title association; `SectionRecord.kind` / `printedNumber`; manifest health (quartile coverage); `INDEX_VERSION = 22` | `Chapter 13` and Roman/adjacent-title chapters resolve; TOC rows are not boundaries; overview cites all document quartiles or is labelled degraded |
| 3 | Planner and leads | `chapter N` / `chapter on X` structural pointers; DEFINITION answer shape (never a navigation lead); list labels generalised (Arabic, alphabetic, Roman, bullets); list lead follows continuation edges across pages and honours requested cardinality | Both chapter-summary phrasings pass; confined-space definition answers as a definition; HIRA returns 5/5 |
| 4 | Tables | Unbordered grid reconstruction in `TableClusterer`; header-qualified row facts; `TableRecord` identity in the manifest; planner table pointers; row-key / matrix cell leads | Risk matrix, class tables and cover tables answer from the index alone |
| 5 | Gemma 4 runtime | `BenchmarkInfo` prefill/decode tokens per second in `GenerationStats`; speculative decoding (MTP) behind a device-gated flag; constrained decoding evaluated for FACT/TABLE output | TTFT and tokens/s gates on Pixel 10 and Nothing A001; no thermal-severe run in a quality comparison |
| 6 | Golden portfolio | CSI spec, construction safety manual, lab safety manual, an ITP/QA-QC document with dense tables, a scanned drawing schedule; stage metrics per case | Release gate replaces phrase-only pass flags |
| 7 | Library extraction | `rag-api` / `rag-runtime` / `rag-apryse` per `FUTURE_VALUE_ADDS.md` | Only after 0–6 are green on both devices |

Milestones 0–5 are implemented. Milestone 4 landed on index version 23 (same outline shape as v22,
plus `TableRecord`). Milestone 6 is split: **6a is the measurement we can run now** on the three
PDFs already on device; **6b** adds the missing document types when those files exist.

## Milestone 6 — Golden portfolio

### Why this milestone, and why it is split

M4 proved table leads on one construction manual. The original table failure was a *different*
document: the 50-case laboratory safety manual (40/50 overall, 8/17 tables). Until that bank and
the CSI Division 03 no-regression suite run on the same index, we do not know whether v23
generalises. An ITP/QA-QC package and a scanned drawing schedule are still required for a release
gate, but they are not on the bench devices yet. Inventing those PDFs would bias the gate.

| Slice | Documents | Exit |
|---|---|---|
| **6a** (measured) | CSI Division 03, construction safety manual (`safety.pdf`), laboratory `SafetyManual.pdf` | Lab tables **9/17** (was 8/17); `safety.pdf` tables stay deterministic (19/23, 11/12); per-stage JSON on Nothing. Division 03 on Nothing v23 is **17/19** / **19/21** (Pixel v22 was 19/19). Captions: `Table (1.1)` / `1` / `4` / `5` on the lab manual. |
| **6b** (later) | an ITP/QA-QC with dense tables; a scanned drawing schedule; Pixel confirmation | Release gate replaces phrase-only pass flags; both devices |

Production rules still contain no test-document title, construction keyword, page number or expected
value.

### 6a sequence

1. Review remaining device misses (below). Apply only generic, stage-correct fixes.
2. Reindex all three documents on Nothing A001 so the caption-heading fix populates `TableRecord`.
3. Run, in order: `safety_manual_qa` (the M4 generalisation check), `single_pdf_baseline` +
   `generated_live_qa` (Division 03 no-regression), `safety_pdf_qa` + `construction_safety_qa`.
4. Record stage counts, `answeredBy`, TTFT and table-lead latency in `V2_2_DEVICE_RESULTS.md`.
5. Do not bump `INDEX_VERSION` unless a schema change appears. Do not start M7.

6a is **measured and stopped** here. Do not invent ITP or drawing-schedule PDFs in this cycle.

### Edge-case review (not all are defects)

| # | Claim | Verdict |
|---|---|---|
| 1 | Unscoped FACT writes blank `sourceSectionId`, so "What about Type C?" cannot inherit §13.3 | **Leave.** `sourceSectionId` is planner-owned (`GenerationStats` contract; overview test requires blank). Inferring the top excerpt would section-filter the follow-up and can drop a sibling clause on the same page. Type C already retrieved page 105; the miss is generation (row 3). Follow-up retrieval already rewrites with the previous question. |
| 2 | `anchorTerms` drops single-letter `Type B` / `Class D` | **Fix.** Add those letters as rerank anchors only. They stay out of `keywordTerms` so AppSearch is not flooded with `b`. |
| 3 | Model keeps `1:1` and drops `20 feet (6Mt)` | **Fix with a lead.** Copy the unique evidence sentence that names one typed class (`type`/`class`/`grade`/`group` + label) together with a conditioned value. Same pattern as the list and definition leads. No soil/slope/excavation vocabulary. |
| 4 | Current v23 index has blank table captions | **Reindex.** Code already leaves `TABLE N …` lines in the page. |

### 6a out of scope

Vision / figure text (hierarchy of controls). Speculative decoding. Library extraction. New ITP or
drawing-schedule fixtures. Changing `sourceSectionId` inheritance. A second embedding query.

## Stage contracts changed in this cycle

### Grounding (M1)

`GroundingStreamFilter` keeps its role as the trust boundary. It now derives its lexicon from
`EvidenceValueLexer` instead of three regular expressions:

- a measurement is a numeric core plus an optional unit token that is attached (`8m`) or separated
  by whitespace (`8 m`); the unit must be observed in evidence adjacent to a number, never guessed
  from the generated text;
- ranges (`21-25`, `21 to 25`), ratios (`1:1`, `1½:1`), fractions (`2 1/2`, `½`) and standard
  identifiers (`IS 3764:1992`, `ASTM C150`) are single tokens whose components are also allowed;
- normalisation is NFKC plus Unicode fraction folding, thousands separators removed, unit case folded;
- an unsupported value is still replaced by `[unverified value]` and recorded with a reason
  (`VALUE_ABSENT`, `UNIT_MISMATCH`, `IDENTIFIER_MISMATCH`) for stage attribution.

### Outline (M2)

`Segment.Heading` gains `kind`. `StructureAnalyzer` recognises `CHAPTER 13`, `CHAPTER XIII`,
`PART IV`, `APPENDIX A` and associates a bare structural label with a geometrically adjacent title
line using position, gap and title shape. Table-of-contents pages remain suppressed.
`SectionRecord` gains `kind` and `printedNumber`; `ManifestHealth` reports quartile coverage and
marks a manifest `DEGRADED` when the top-level nodes do not span the document. Document overview
samples top-level nodes by page quartile and falls back to page-stratified sampling when degraded.

### Planning (M3)

`QueryPlanner` resolves `chapter 13`, `chapter xiii`, `part 4`, `appendix a` against
`printedNumber` and `kind`, and `chapter on excavation` against titles of that kind. A DEFINITION
shape (`what is a`, `what makes`, `define`, `meaning of`) is passed to `ContextSelector` and excludes
the section-pointer lead, which now requires explicit navigation wording. `StructureAnalyzer`
accepts Roman-numeral and bullet labels; the list lead follows `continuesToChunkIndex` across pages.

### Tables (M4)

`TableClusterer` reconstructs unbordered grids: wrapped cells stay in their row, captions attach
from the adjacent line, and uppercase `TABLE N …` lines are not heading boundaries.
`ScriptAwareChunker` embeds header-qualified row facts (`Risk: 21-25; Control: Very high`) and
assigns a stable `tableId` / printed `tableNumber`. The v23 manifest stores `TableRecord` entries.
`QueryPlanner` resolves `table 5.1` and titled requests (`the likelihood table`) without treating
those numbers as section ids. `buildTableLead` answers a unique row-key or row×column cell from
the fetched grid; otherwise generation proceeds over the same structured evidence.

### Runtime (M5)

`GemmaEngine` reads `Conversation.getBenchmarkInfo()` after each turn and reports prefill and decode
tokens per second so latency regressions are attributed to the correct phase.
`ExperimentalFlags.enableSpeculativeDecoding` is exposed as an engine option; it stays off until both
devices show a decode gain without a thermal penalty.

## Verification

- Tier A (JVM): lexer, heading, list, planner, manifest health, overview sampling, table grid,
  row-key / matrix lead, and table-identity tests.
- Tier B/C (device): hash-selected suites on Nothing A001. M4: `safety_pdf_qa`, `construction_safety_qa`.
  M6a: those plus `safety_manual_qa`, `single_pdf_baseline`, `generated_live_qa` after a caption reindex.
- Results are recorded in `V2_2_DEVICE_RESULTS.md` with device, backend, index version, TTFT, tokens
  per second and per-stage pass counts.
