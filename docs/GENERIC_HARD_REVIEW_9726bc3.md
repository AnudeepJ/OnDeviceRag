# Follow-up review: 9726bc3

Reviewed 2026-09-10 against previous reviewed commit `61016ea`: 13 committed files, 406 insertions, 32 deletions. Also inspected the pre-existing uncommitted ModelManagerScreen UI edit; left it unchanged.

## Verdict

**Request changes. Keep this as an experimental branch; it is not ready for release or Aconex library extraction.**

Resolution update: the seven reproduced code defects and strict hash selection were fixed in the
working tree after this review. Permanent JVM regressions, the full unit suite, instrumentation
compilation and APK assembly pass. The final APK cold-launched on a Pixel 10 without a crash. The
Pixel has no indexed PDFs, so the three-PDF live quality gate still needs to run on the Nothing phone
or another device holding the v23 indexes before release acceptance.

The commit improves caption recognition and the normal resolved-table path, and documents a useful three-PDF measurement. It does not close the previous correctness review. Targeted JVM verification confirms three unchanged production defects, incomplete table-scope protection, and three new implementation problems.

## Findings in the pulled changes

### P1 — Conditional answers do not match the requested property

`AnswerQuestionUseCase.kt:849-873` checks a typed label, a condition cue and a numeric value, but not whether the sentence answers the question's requested property or subject. A unique matching sentence becomes decisive and skips generation.

Reproduced question: `What is the inspection interval for Type B equipment?`

Only evidence: `Type B equipment must remain below 20 metres when operating.`

Actual decisive answer: that operating-height sentence, with a page citation. It contains no inspection interval. Matching the same type is insufficient to answer the question. Require property/subject and condition compatibility, and retain ambiguity or generation when compatibility cannot be established. Follow-up questions need their resolved conversational subject passed to this answerer.

### P1 — Table-number fallback matches prefixes

`QueryPlanner.kt:292-295` first compares exact numbers, then tries a regex with no identifier boundary after the requested number. Its trailing whitespace and closing parenthesis are optional.

Reproduced: a manifest containing only `Table 1.10 Limits` resolves a request for `Table 1.1` to that table. When both exist, the same rule can create ambiguity and remove the intended scope.

Parse a complete caption identifier and compare it exactly. Support embedded/malformed captions without allowing `1.1` to match `1.10` or `1` to match `10`.

### P1 — Missing resolved-table evidence silently broadens scope

`AnswerQuestionUseCase.kt:500-503` uses `candidates.filter { it.tableId == resolvedTableId }.ifEmpty { candidates }`.

Reproduced: the requested Table 5.1 has no candidates; only Table 5.2 contains Class D → 99. Passing Table 5.1's resolved ID still returns the other table's value as decisive. This can occur when direct fetching fails and the existing merge path retains hybrid hits.

If explicit scope has no usable evidence, decline the deterministic answer or report unavailable evidence. Do not substitute unrelated candidates. The normal path with both tables present now passes the earlier regression. However, the constraint is still local to `buildTableLead`: earlier list/standard leads and subsequent conditional/generated answers also need to honor explicit scope.

### P2 — New single-letter reranking anchors have no effect

`HybridQuery.kt:86` adds B/D-style labels to `anchors`, but `rerank` at `:109-116` constructs scored stems exclusively from `keywordTerms`. Those terms deliberately exclude single letters. Consequently the new anchor never participates in scoring.

Reproduced: equally scored `Type C soil slope` and `Type B soil slope`, in that order, remain in that order for `What is the slope for Type B soil?` despite the B anchor. The added test checks anchor extraction, not ranking behavior.

Include typed-label features in local ranking independently of AppSearch keyword terms. Prefer matching the typed pair (`type B`) rather than any isolated B token. Add ranking assertions with sibling types and unrelated letter labels.

## Earlier review findings

| Finding | Current status |
|---|---|
| Definition stops inside `1.5 metres` | Still reproduced: `A widget is a component measuring 1. [Page 1]`. |
| Invented `[E10]` maps to existing E1 | Still reproduced, without a grounding failure. |
| List with missing known continuation is decisive | Still reproduced for `List all widget setup steps`. |
| Explicit table loses to unrelated caption | Normal resolved-table path fixed; missing-evidence fallback still fails. |
| Requested benchmark hash falls back to filename | Unchanged; strict document identity is still not enforced. |

The earlier definition/citation/list findings remain P1 correctness blockers. See `GENERIC_HARD_REVIEW_2026_09_10.md` for their original evidence and architectural assessment.

## What improved and what the measurements mean

- Parenthesized captions, nearby-caption attachment and matrix/list separation are useful incremental extraction changes.
- Passing `resolvedTableId` to table leads fixes the normal competing-caption reproducer.
- Returning a whole unique row can preserve multiple requested fields; proper header associations remain necessary.
- Rejecting a bare Class header avoids the documented false plywood lead.
- Measuring the laboratory manual and Division 03 on v23 addresses an important gap in the earlier evidence.

Recorded device results in `V2_2_DEVICE_RESULTS.md` are not fresh measurements from this review:

| Suite | Recorded result |
|---|---|
| Laboratory safety | 39/50 answers; tables 9/17, versus historical 40/50 and 8/17 |
| Division 03 baseline | 17/19 answers |
| Division 03 generated | 19/21 answers, 20/21 pages |
| Construction safety new bank | 19/23 |
| Construction safety strict bank | 11/12 |

The table gain is one case, while overall lab correctness is one case lower. Division 03's prior Pixel v22 results are not a controlled same-device comparison, so the score difference alone cannot establish a commit-caused regression. Nevertheless, the current suites fail their acceptance criteria. The report explicitly records that the conditional lead did not fix the Type B case on device. Calling 6a **measured** is accurate; calling the branch accepted would not be.

## Required next step

Close the seven reproduced correctness/behavior checks and strict hash selection before adding more heuristic leads. Verify explicit scope across all answer paths, then rerun the frozen three-PDF suite on the final build/index. Separate extraction failures, answer omissions and scorer-format disagreements rather than weakening assertions to improve totals. Proceed with the larger portfolio and library extraction only after the existing correctness gate is met.

## Verification

- Existing `:app:testDebugUnitTest` and `:app:compileDebugAndroidTestKotlin`: passed.
- Eight targeted review probes: one passed, seven failed on the expected assertions.
- Probe source and XML retained at `/tmp/generic-hard-review-9726bc3/`.
- Temporary probe file removed from the application source tree.
- No production changes, phone installation or new device runs performed.
- Existing ModelManagerScreen edit and earlier review document preserved.
