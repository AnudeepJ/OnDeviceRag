# Future Value Adds

This document holds intentionally deferred product increments. They are not current commitments or
prerequisites for the single-PDF RAG flow.

## Citation-to-original-PDF evidence viewer

### Value

Let a user verify an AI answer against the original PDF, rather than only reading the extracted
chunk in the citation sheet. Tapping a citation opens the source document at its cited page and
visually highlights the supporting text when it can be located exactly.

Apryse's `TextSearch` is suitable for the page-local exact-match/highlight part of this experience.
It supports text and regular-expression matching and supplies the highlight geometry. It is not a
semantic search engine and must not replace AppSearch hybrid retrieval.

### Scope and flow

```text
Chat answer citation
  -> existing CitationSheet / citation metadata
  -> open a read-only Apryse PDFViewCtrl at citation.pageNumber
  -> search only that page for a short, exact source excerpt
  -> if found, display Apryse's highlight and allow next/previous exact match
  -> if not found, show the cited page without a highlight and retain the extracted chunk text
```

The source document must be opened from the app-managed document record, not from an arbitrary
path supplied by UI state. The viewer is read-only: it must not add annotations or modify the
source PDF.

### Proposed implementation

1. Add a Compose-hosted, read-only PDF viewer screen backed by Apryse `PDFViewCtrl` and
   `ToolManager`. The app currently has the Apryse extraction wrapper but no PDF viewer.
2. Extend the citation-navigation contract to include the source document reference, page number,
   and a short highlight candidate. Keep `chunkId` as the canonical lookup key.
3. Generate the highlight candidate from a stable, distinctive exact excerpt of `bodyText` during
   indexing or citation presentation. Prefer a complete sentence or table row; do not use a whole
   chunk or a model-generated paraphrase.
4. On opening the viewer, use Apryse `TextSearch` with `e_page_stop` and highlight mode, restricted
   to the cited page. Run it off the main thread and cancel it when the screen closes or the user
   selects another citation.
5. If the candidate cannot be found, fall back silently to the cited page and label the state
   "Source page" rather than claiming an exact highlight. The citation sheet remains available as
   the accessible text alternative.
6. Add a separate explicit "Find in document" UI later if useful. It may support exact strings,
   whole-word matching, and regex, but it must remain separate from the chat/RAG query field.

### Important constraints

- `TextSearch` operates on the original PDF text layer. Pages indexed through ML Kit OCR may not
  contain searchable native text, so OCR-only/scanned pages must use the page-only fallback.
- Chunk text is normalised and may be reflowed, header/footer-stripped, or table-reconstructed.
  It is not always a byte-for-byte string from the PDF. Only search a retained exact excerpt and
  never treat a miss as a bad citation.
- Restricting search to the cited page prevents a full-document scan each time a citation is
  opened and avoids highlighting an identical phrase elsewhere.
- Keep AppSearch as the system for semantic, keyword, cross-chunk, and eventual cross-document
  retrieval. Apryse only provides source-page navigation and exact visual evidence.

### Acceptance criteria

- A text-native PDF citation opens the intended document and page.
- For fixture citations with a retained exact excerpt, the intended source text is highlighted.
- For scanned/OCR-only pages and no-match cases, the page opens without error and no misleading
  highlight is shown.
- Opening, searching, and closing the viewer do not block chat generation or mutate the PDF.
- Instrumented tests cover text-native match, no-match, and page-only fallback; manual testing
  covers CJK, tables, repeated phrases, and a scanned PDF.

### Prerequisite / priority

Implement after the core single-PDF retrieval and citation flow is considered product-ready. It is
a trust and usability improvement, not a retrieval-performance project.
