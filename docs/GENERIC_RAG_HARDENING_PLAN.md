# Generic on-device RAG hardening plan

Status: proposed implementation plan based on the 209-page `safety.pdf` live run on Nothing A001.

This plan is deliberately document-agnostic. Construction terms, known page numbers, and expected
answers from the test PDF belong only in test fixtures. Production behavior must be driven by
layout, structure, query intent, and evidence.

## 1. What the live run established

The current foundation is sound:

- A 209-page PDF indexed into 1,427 chunks without a crash.
- Dense/hybrid retrieval found the correct pages for most factual questions.
- Gemma generated on GPU at roughly 8.9-14.9 tokens/second, with normal first-token latency of
  about 3.7-11.3 seconds.
- Direct facts, safe refusal, and conversational inheritance worked in representative cases.
- Tables with preserved row/header context and whitespace-separated units produced good answers.

The 21-question live acceptance run produced 10 clean passes, 5 partial answers, and 6 failures.
The failures form four reusable capability gaps:

| Capability | Observed symptom | Primary failing stage |
|---|---|---|
| Value grounding | Supported `8m`, `150mm`, `6Mt`, and `1992` were suppressed | Grounding/token normalization |
| Structural lookup | Chapter summary questions refused | Structure manifest and query planning |
| List completeness | One of five HIRA steps was returned | List extraction/chunk association and answer policy |
| Document overview | Only the final appendix was summarized; `SECTION null` was displayed | Manifest quality and overview sampling |

The confined-space definition is a fifth useful regression: retrieval reached the right content,
but the response acted like a navigation answer rather than a definition. That is a query-routing
and deterministic-lead selection problem.

## 2. Why these defects were missed earlier

### 2.1 Corpus-shape bias

Most fixtures and live regressions were based on CSI-style specifications. The implementation and
tests therefore cover `SECTION 03300`, `PART 1`, and dotted headings well, but not the equally common
`CHAPTER 13` followed by a title on the next line. The earlier safety fixture is also a different
manual, despite its similar name.

### 2.2 Example coverage instead of grammar coverage

Grounding tests cover decimals, fractions, specification IDs, and spaced value/unit pairs. They do
not exercise the equivalent compact forms (`8m`, `150mm`), case variants (`6Mt`), year-bearing
standard identifiers, ranges, Unicode fractions, or units split across streamed tokens. The regex
currently requires whitespace before a unit while the numeric regex rejects a following letter, so
compact units cannot enter the allowed lexicon.

### 2.3 No explicit outline contract

The manifest stores a flat list of section records and reconstructs document breadth from inferred
levels. It does not represent heading kind, parent identity, or a validated document outline. An
uppercase line can become a generic section, and an empty or malformed extracted heading can still
influence overview selection. Even sampling over a flat list cannot repair a wrong hierarchy.

### 2.4 End-to-end results did not identify the failing stage

A correct citation page was treated as encouraging, but the evaluator did not separately gate:

1. source extraction,
2. structure recognition,
3. retrieval,
4. context selection,
5. answer completeness,
6. grounding preservation, and
7. citation accuracy.

That allowed “retrieval passed, answer failed” cases to remain hidden behind a single overall result.

### 2.5 Missing capability matrix

There was no required matrix crossing question intent with evidence shape. For example, numeric fact,
definition, complete list, table row, chapter summary, document summary, follow-up, and unanswerable
questions should each be tested against paragraphs, lists, tables, compact units, page boundaries,
and differently styled headings.

## 3. Architectural changes

### 3.1 Introduce a document-neutral outline model

Replace the implicit “all headings are sections” assumption with an explicit structural node:

```text
OutlineNode
  id
  kind: DOCUMENT | PART | CHAPTER | SECTION | CLAUSE | HEADING
  printedNumber: String?
  title: String
  level: Int
  parentId: String?
  startPage / endPage
  orderedChunkIds
  confidence and evidence
```

Keep `specificationNumber` as optional domain metadata; do not use it as the general hierarchy.
Chunks should reference `outlineNodeId` and retain the complete structural path.

Generic parser changes:

- Recognize `Chapter 13`, `CHAPTER XIII`, `Part IV`, section/clause forms, dotted numbers, and styled
  unnumbered headings using standard Kotlin regular expressions and layout signals.
- Merge a bare structural label with a geometrically adjacent title, such as `CHAPTER 13` followed by
  `EXCAVATION`. The rule must depend on position, style, gap, and title shape—not title words.
- Continue suppressing table-of-contents rows as actual boundaries, but retain a high-confidence TOC
  as optional outline evidence.
- Build parent/child relationships with a level stack and close page spans when the next peer or
  ancestor begins.
- Never synthesize strings containing `null`; absent printed identifiers remain absent.

This is an index-schema change. Increment `DocumentStructureManifest.INDEX_VERSION` and require a
re-index; no Room migration is needed during prototype development.

### 3.2 Make manifest health an indexing gate

Add a `ManifestHealthReport` with deterministic invariants:

- every indexed chunk occurs exactly once and order is monotonic;
- every node has a nonblank, non-`null` title or is the document root;
- child page spans are contained by their parent;
- roots and top-level nodes cover the document rather than only its final pages;
- no impossible level jumps or parent cycles;
- the first/middle/final page ranges have structural coverage, or the manifest is explicitly marked
  `DEGRADED`;
- direct-fetch IDs exist and restore manifest order.

Publishing remains atomic. If health is degraded:

- factual questions fall back to ordinary hybrid retrieval;
- section/chapter summary first tries heading/title retrieval and reconstructs a local page scope;
- document overview uses page-stratified sampling;
- corrupt direct-fetch data never throws through the user flow.

### 3.3 Generalize structural query planning

Represent an explicit reference as `StructuralPointer(kind, printedNumber)` instead of two special
fields for specification and section.

Planner behavior:

1. Detect overview, structural summary, definition, list, table/detail, fact, and navigation intents.
2. Resolve an explicit pointer (`chapter 13`, `section 3.20`) against matching outline-node kinds.
3. Resolve a title (`chapter on excavation`) by normalized token overlap and bounded Kotlin
   Levenshtein distance. No third-party string or regex dependency is needed.
4. If the exact structure is absent, perform heading/title retrieval, then select the nearest valid
   structural ancestor.
5. Ask for clarification only when two candidates remain within the safety margin.
6. Fall back to hybrid retrieval for an unresolved fact; reserve refusal for insufficient evidence,
   not merely an imperfect manifest.

Navigation leads must only run for explicit navigation wording such as “which section” or “where in
the document.” Definition wording such as “what makes,” “define,” or “what is” must be answered from
the definition evidence rather than converted into a structural pointer.

### 3.4 Replace regex-only numeric grounding with typed evidence values

Create a pure `EvidenceValueLexer` that extracts typed values before generation:

```text
NUMBER, PERCENT, FRACTION, RANGE, RATIO, MEASUREMENT,
DATE_OR_YEAR, STANDARD_ID, COMPOUND_ID
```

For a measurement, preserve the numeric core and an optional adjacent unit with either zero or more
spaces. Normalize Unicode, thousands separators, decimal forms, unit case, and whitespace for
comparison while preserving the source spelling in output. Match the longest valid evidence token.

Important safety constraints:

- A following word is not automatically a unit. It must be a recognized unit observed in evidence.
- Do not whitelist document values or construction units in production code.
- Do not accept a number merely because it appeared in the question.
- An answer may reproduce only values present in selected evidence or validated structural pointers.
- An unsupported value remains blocked; a supported value must never become `[unverified value]`.
- Record a debug-only reason such as `VALUE_ABSENT`, `UNIT_MISMATCH`, or `IDENTIFIER_MISMATCH` so a
  failed live case can be assigned to the correct stage.

For deterministic table/list answers, copy exact source spans and citations instead of asking the
model to regenerate values. This improves both accuracy and response time.

### 3.5 Preserve lists as first-class structures

Extend list parsing to support Arabic numbers, alphabetic labels, Roman numerals, bullets, and
checkmarks. Store a stable list ID, item ordinal, label style, and continuation edges when a list
crosses a chunk or page boundary.

The current deterministic list lead is restricted to same-page `LIST` chunks, a narrow introduction
ending in a colon, and Arabic/alphabetic labels. Replace it with a `StructuredListAnswerer` that:

- resolves the list introduction and subject by lexical evidence;
- follows list and chunk continuation edges with deduplication;
- preserves all contiguous items up to the context/output safety limit;
- honors an explicit requested count;
- distinguishes “all steps” from “key steps” and never silently returns one item for a complete-list
  request;
- streams exact source-derived items immediately, using Gemma only when synthesis is required.

### 3.6 Rebuild overview selection around coverage

Document overview must not be ordinary top-k semantic retrieval. It should use the validated outline:

- include the document title/introduction or a trusted TOC;
- select top-level nodes with page-range diversity;
- reserve representatives from the first, middle, and final document quartiles;
- cap repeated siblings from one appendix or chapter;
- use child nodes only when a root contains most of the document;
- deduplicate continuation overlap before consuming the context budget.

If the outline is degraded, use page-stratified sampling plus heading retrieval and label the result
as a concise overview rather than pretending a final-page sample is comprehensive.

The immediate `Document scope` response should be deterministic and citation-backed. Gemma can then
add a bounded synthesis. This retains the speed advantage and prevents `SECTION null` output.

## 4. Test design that would have caught these failures

### 4.1 Freeze the live failures before changing behavior

Add red tests for:

- `8m`, `8 m`, `150mm`, `150 mm`, `6Mt`, `6 m`, `10,000 psi`, `1½:1`, and `IS 3764:1992`;
- streamed token splits at every character boundary of each form;
- `CHAPTER 13` plus a next-line title and Roman-numeral chapters;
- the same strings on a TOC page, where they must not become boundaries;
- explicit and title-only chapter summaries;
- a five-item Roman-numeral list split across chunks and pages;
- a 209-page outline whose final appendix contains many headings;
- definition wording adjacent to a structural cross-reference.

### 4.2 Use a capability matrix

Every release test pack should deliberately cover:

| Intent | Required variants |
|---|---|
| Short fact | text, compact unit, range/ratio, identifier/year |
| Definition | direct definition, definition near a cross-reference |
| Complete list | numeric, Roman, bullets, cross-chunk, cross-page |
| Table/detail | exact row, repeated row label under different parents, flattened table |
| Structural summary | explicit number, title-only, ambiguous title, missing manifest |
| Document overview | short/long, TOC/no TOC, numbered/unnumbered structure |
| Conversation | pronoun follow-up, topic switch, clarification |
| Unanswerable | absent person/value, plausible but unsupported claim |

Tests should report uncovered matrix cells. A high score on only covered cells is not a release pass.

### 4.3 Measure each pipeline stage independently

Each QA case should hold:

- document ID and extraction fingerprint;
- question and intent;
- gold pages/chunks or source spans;
- expected normalized values and identifiers;
- required and forbidden facts;
- expected list cardinality;
- expected answerability and citations.

Report these metrics separately:

- extraction fidelity and table/list structure accuracy;
- outline node kind/title/parent/page-span F1;
- retrieval Recall@1/5, MRR, and nDCG@10;
- selected-context recall, precision, and duplicate-token ratio;
- required-fact coverage and forbidden-fact leakage;
- numeric/value preservation and unsupported-value block rate;
- list completeness;
- citation page/source precision and recall;
- refusal precision/recall;
- indexing time per page/chunk, retrieval p50/p95, time to first token p50/p95, total latency,
  tokens/second, peak PSS, and thermal state.

Do not reduce these to one opaque score. A failure report should identify the first failing stage.

The device runner must select a document by exact content hash and extraction fingerprint, not by
filename similarity or a broad page-count predicate.

## 5. Open benchmark strategy

There is no single public benchmark that exercises PDF rendering, Apryse extraction, tables,
hierarchy, retrieval, grounded generation, citations, multi-turn behavior, and mobile performance.
Use a small portfolio and keep the local golden-PDF suite as the end-to-end release gate.

### Recommended public datasets

| Purpose | Dataset | How to use it here |
|---|---|---|
| Retrieval generalization | [BEIR](https://github.com/beir-cellar/beir) | Start with SciFact, NFCorpus, and FiQA. Run the on-device hybrid retriever and score Recall@k, MRR, and nDCG. |
| End-to-end RAG labels | [RAGBench](https://arxiv.org/abs/2407.11005) | Sample `emanual`, `techqa`, `finqa`, `tatqa`, and `cuad`; use its evidence/answer labels for stage-level tests. |
| Table plus text and numeric reasoning | [TAT-QA](https://aclanthology.org/2021.acl-long.254/) | Test row/paragraph evidence selection, exact value/scale, and numeric answer grounding. |
| Free-form table answers | [FeTaQA](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00446/109273/FeTaQA-Free-form-Table-Question-Answering) | Test supporting-cell retrieval and complete free-form synthesis from a table. |
| Multi-turn retrieval | [MTRAG](https://github.com/IBM/mt-rag-benchmark) | Sample human conversations for follow-ups, clarification, partial, and unanswerable turns. |
| Long-document QA and summaries | [LongBench](https://github.com/THUDM/LongBench) | Use small samples from MultiFieldQA-en/Qasper and GovReport to test QA and summary behavior at different lengths. |
| Image/OCR track | [DocVQA](https://site.docvqa.org/datasets/docvqa) | Use separately for image-heavy pages and ML Kit OCR extraction; it is not a replacement for the text-PDF RAG gate. |

Licenses and redistribution terms must be reviewed per dataset before checking data into the
repository. Benchmark conversion scripts and metadata can live in the repository while large source
files remain external.

### Important limitation

Most public RAG benchmarks provide pre-extracted text or tables. Running them only from their JSON
proves retrieval/generation quality but does not test the PDF ingestion pipeline. Maintain two modes:

1. **Seeded benchmark mode:** ingest the supplied passages directly, isolating retrieval and answer
   quality.
2. **Rendered end-to-end mode:** convert a license-compatible subset into deterministic PDFs, or use
   original source PDFs when provided, then run extraction through the production path.

Never compare these modes as if they measured the same thing.

## 6. On-device execution tiers

Running an entire public benchmark through Gemma on every phone commit would be too slow and would
mix quality regressions with thermal throttling. Use three tiers.

### Tier A: JVM tests on every change

- parser, outline, value lexer, list association, context deduplication, and evaluator tests;
- table-driven equivalence classes and deterministic randomized/property-style inputs;
- no model and no device dependency.

### Tier B: on-device retrieval gate

- seed a fixed 100-300-query benchmark subset into the real AppSearch/index path;
- test dense, lexical, and hybrid retrieval independently;
- run on at least one arm64 device;
- no Gemma generation except a small smoke set, keeping this gate fast and stable.

### Tier C: physical-device end-to-end release suite

- three to five PDFs spanning structured specification, general manual, table-heavy report,
  image-heavy/scanned document, and weak/unnumbered headings;
- 20-50 questions per document with the capability matrix above;
- cold and warm runs on Pixel and Nothing-class devices, GPU and CPU fallback;
- record thermal status, memory, TTFT, total latency, throughput, citations, and stage metrics.

Use relative performance gates against a stored device/model baseline, plus product ceilings. Initial
ceilings should be calibrated from repeated runs; suggested starting checks are retrieval p95 below
500 ms, deterministic source-derived answers visible below 1 second, no more than 10% TTFT or peak-PSS
regression, and no thermal-severe run included in a quality comparison.

LLM-as-a-judge may be an offline diagnostic, not the release oracle. The same on-device Gemma judging
its own answer introduces correlated bias and doubles memory/latency pressure. Gold evidence spans,
exact normalized values, fact coverage, citation checks, and refusal labels should determine the
automated gate; manually review disagreements and only then use an independent judge as secondary
evidence.

## 7. Implementation sequence

### Milestone 0 — Lock the evidence

1. Convert every current Safety PDF failure into a red unit or device regression.
2. Capture the indexed outline and stage diagnostics for the failing queries.
3. Record the existing Division 03 and earlier Safety Manual scores as no-regression baselines.

Exit: every observed failure is reproducible without relying on a manually inspected chat screen.

### Milestone 1 — Typed grounding

1. Add `EvidenceValueLexer` and exhaustive pure unit tests.
2. Integrate it behind the existing streaming filter contract.
3. Add chunk-boundary streaming tests and grounding reason telemetry.

Exit: all supported compact/spaced values survive; all synthetic unsupported values remain blocked;
no meaningful CPU regression in a microbenchmark.

### Milestone 2 — Outline schema and parser

1. Add `OutlineNode`, heading kind, parent ID, confidence, and manifest health.
2. Recognize generic chapter/part forms and adjacent titles.
3. Increment the index version and re-index all test documents.
4. Implement degraded-manifest fallbacks.

Exit: chapter title/number queries resolve; TOC entries do not create duplicate boundaries; manifest
coverage spans the whole document.

### Milestone 3 — Planner and overview

1. Add structural pointer kinds and definition/navigation separation.
2. Resolve explicit number, title, and heading-retrieval fallbacks.
3. Replace flat overview sampling with outline/page diversity.
4. Remove all user-visible placeholder identities.

Exit: both chapter-summary phrasings work; document overview represents the complete manual; a bad
manifest still permits safe factual fallback.

### Milestone 4 — Structured lists

1. Generalize list labels and persist list identity/ordinal/continuation.
2. Deduplicate cross-chunk/page continuations.
3. Add deterministic complete-list answers with cardinality checks.

Exit: HIRA returns all five source steps, and adversarial nearby lists do not hijack the answer.

### Milestone 5 — Evaluation harness and benchmark adapters

1. Extend QA JSON with gold spans, values, cardinality, citations, and answerability.
2. Emit stage-level JSON and a human-readable report.
3. Add BEIR/RAGBench/TAT-QA/MTRAG subset adapters without runtime app dependencies.
4. Add the three execution tiers and baseline comparison.

Exit: a failed benchmark names the failing stage and retains device/model/index/thermal metadata.

## 8. Release acceptance for this hardening cycle

- The 21-question live Safety PDF set has no known failure or silent partial answer.
- Existing Division 03 and earlier Safety Manual sets show no quality regression.
- Supported-value preservation is 100% on the value-form matrix; unsupported-value blocking is 100%
  on the adversarial set.
- Explicit and title-only structural summaries pass across at least three unrelated document styles.
- Document overviews cite representative content from all document quartiles when applicable.
- Complete-list tests preserve expected cardinality across labels, chunks, and pages.
- Retrieval Recall@5 and citation correctness meet the stored baseline or improve.
- Deterministic-answer latency remains sub-second; generated-answer performance stays within the
  calibrated device regression budget.
- No production rule contains a test-document title, construction keyword, page number, or expected
  answer value.

## 9. Recommendation

Implement the milestones in order rather than tuning prompts one failure at a time. Milestones 1-4
repair deterministic pipeline contracts; prompt tuning should happen only after those contracts are
green. Begin with the current Safety PDF as the red regression corpus, retain Division 03 as the
no-regression corpus, and introduce public benchmark subsets during Milestone 5. This gives fast local
feedback now without delaying the architectural corrections, while building an evidence base that
will generalize beyond a single PDF.
