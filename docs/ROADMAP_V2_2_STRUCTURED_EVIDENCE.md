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
| 4 | Tables | `TableGridBuilder` for unbordered aligned matrices; header-qualified row facts in the manifest (see V2.2 plan) | Risk matrix, class tables and cover tables answer from the index alone |
| 5 | Gemma 4 runtime | `BenchmarkInfo` prefill/decode tokens per second in `GenerationStats`; speculative decoding (MTP) behind a device-gated flag; constrained decoding evaluated for FACT/TABLE output | TTFT and tokens/s gates on Pixel 10 and Nothing A001; no thermal-severe run in a quality comparison |
| 6 | Golden portfolio | CSI spec, construction safety manual, lab safety manual, an ITP/QA-QC document with dense tables, a scanned drawing schedule; stage metrics per case | Release gate replaces phrase-only pass flags |
| 7 | Library extraction | `rag-api` / `rag-runtime` / `rag-apryse` per `FUTURE_VALUE_ADDS.md` | Only after 0–6 are green on both devices |

Milestones 0–3 and 5 are implemented in this cycle. Milestone 4 starts once the v22 outline is stable
on both devices so the table identity lands on the same manifest shape.

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

### Runtime (M5)

`GemmaEngine` reads `Conversation.getBenchmarkInfo()` after each turn and reports prefill and decode
tokens per second so latency regressions are attributed to the correct phase.
`ExperimentalFlags.enableSpeculativeDecoding` is exposed as an engine option; it stays off until both
devices show a decode gain without a thermal penalty.

## Verification

- Tier A (JVM): lexer, heading, list, planner, manifest health, overview sampling tests.
- Tier B/C (device): `SinglePdfBaselineDeviceTest` suites selected by content hash on Pixel 10 and
  Nothing A001 after re-indexing to v22: `single_pdf_baseline.json`, `generated_live_qa.json`,
  `construction_safety_qa.json`, `safety_pdf_qa.json`.
- Results are recorded in `V2_2_DEVICE_RESULTS.md` with device, backend, index version, TTFT, tokens
  per second and per-stage pass counts.
