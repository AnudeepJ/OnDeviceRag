# Future Value Adds

This document holds the next major architectural increment and later product value adds. The current
standalone application is the reference host used to prove and test the on-device RAG implementation;
the production destination is a reusable Android library integrated into the Aconex application in
`cyrus-android`.

## Next major increment: reusable library and Aconex integration

### Goal

Convert the proven on-device RAG implementation into host-neutral Android library modules and
integrate them into Aconex without copying the sample application's navigation, database, document
ownership or viewer implementation into the host.

The standalone application remains a first-class sample and device-test harness. It must consume the
same public library API as Aconex so sample success continues to represent the shipped integration.
New RAG logic must not be added directly to the sample application's `RagViewModel`, Activity or
Compose navigation when it belongs in a reusable module.

### Ownership boundary

The final responsibility split is:

```text
Aconex host application
  document authorization, project/account/revision identity
  document availability and lifecycle events
  existing Apryse PDFViewCtrl/ToolManager viewer and markup workflows
  product navigation, design system, analytics and feature flags
  authenticated model-distribution policy
                 |
                 | stable host interfaces
                 v
On-device RAG library
  index manifests, chunks, embeddings and retrieval
  source anchors and grounded citations
  Gemma/EmbeddingGemma runtime lifecycle
  private AppSearch/Room state and cleanup operations
  :rag_inference service and typed client API
                 ^
                 |
Standalone sample app
  reference adapters, Compose UX and device/evaluation harness
```

The library must not import `com.oracle.aconex.*` classes. Aconex-specific implementations live in
the host and implement interfaces defined by `rag-api`.

### Proposed library modules

- `rag-api`: ABI-neutral Kotlin/Parcelable contracts, result types, capability reporting and client
  interfaces. It must not expose Apryse, LiteRT, AppSearch, Room or Compose types.
- `rag-runtime`: the inference service, Gemma engine ownership, scheduling, cancellation, GPU/CPU
  fallback, model validation and service client implementation.
- `rag-index`: Apryse/ML Kit extraction, layout analysis, chunking, EmbeddingGemma, manifests,
  AppSearch and citation/source-anchor persistence.
- `rag-apryse`: the Apryse-specific extraction and evidence-highlight integration behind library
  interfaces. Keep the Apryse version aligned with the consuming host.
- `rag-ui-compose`: optional chat, model-management and citation UI. Aconex may use it, wrap it in its
  design system or implement its own UI over `rag-api`.
- `sample-app`: the current application after its reusable code has moved into the modules above.

These are dependency boundaries, not a requirement to create every Gradle module in one commit. The
first extraction may combine `rag-runtime` and `rag-index`, but public API, host adapters and optional
UI must remain separate from the beginning.

### Host-facing contracts

The public API should expose asynchronous operations with structured progress and cancellation:

```text
RagClient
  capabilities()
  installOrUpdateModels(request)
  index(documentSource, progressCallback)
  ask(documentKey, question, history, streamCallback)
  cancel(generationId)
  deleteDocument(documentKey)
  purgeScope(scope)
  diagnostics()

RagDocumentSource
  documentKey
  displayName
  verified local content source

CitationNavigator
  openCitation(citationTarget, mode)

ModelArtifactProvider
  resolve approved model manifest and authenticated download request

TelemetrySink
  receive privacy-reviewed structured events without prompt or document text
```

Do not make an arbitrary file path part of the public navigation contract. The host supplies an
authorised document source; the library computes/verifies its content hash and either uses a stable
same-UID source or copies it atomically into library-managed storage. The contract must define who
owns temporary copies and when they are deleted.

### Tenant, project and revision isolation

A content hash alone is not a sufficient production document identity. Two Aconex projects or
accounts may contain the same bytes while having different access rights and lifecycles. Use a key
equivalent to:

```text
RagScope
  accountId
  projectId

RagDocumentKey
  scope
  documentId
  revisionId
  contentHash
```

Derive AppSearch database/namespace identity and transcript ownership from this scoped key. Hash or
otherwise encode sensitive host identifiers before using them in file names or diagnostics. Every
lookup, deletion and citation must carry the scope explicitly; never search all local scopes and
filter afterward.

Aconex must call library cleanup hooks on logout, account removal, project removal, permission
revocation, document deletion and revision supersede. Cleanup must be idempotent, cancellable and
safe after process death. Superseding a document creates a new revision/index; it must not retarget
old citations silently.

### Dependency convergence

#### Apryse

Aconex already has established `PDFViewCtrl`, `ToolManager`, read-only viewing and markup flows. The
production integration must reuse those flows through `CitationNavigator`; it must not ship a second
competing viewer implementation. Upgrade Aconex and the RAG modules to one pinned, current Apryse
version before integration. `pdftron` and `tools` must always use exactly the same version, and the
resolved dependency graph and packaged native libraries must be verified in CI.

The sample app may provide its own Apryse viewer adapter to exercise the same citation contract.
Apryse types remain behind `rag-apryse` and must not leak through `rag-api`.

#### LiteRT-LM

Upgrade Aconex from LiteRT-LM `0.13.1` to the version validated by the RAG library (`0.16.1` at the
time of this plan), then run the existing form-AI tests against the upgraded runtime. Dependency
resolution must select exactly one LiteRT-LM version for the application.

Do not retain two independent long-lived Gemma engine owners. Replace the direct engine ownership in
Aconex's `Gemma4AiBackend` with a client of the shared on-device inference runtime, or move the common
engine/session layer below both the form-AI and RAG use cases. The shared runtime must provide fair
scheduling, one active generation per engine, caller-scoped cancellation and separate prompt/result
contracts so a form request cannot be mistaken for a RAG chat request.

#### Android and build tooling

Do not expose implementation-only AndroidX or Kotlin types in `rag-api`. Publish consumer ProGuard/R8
rules, manifest placeholders and resource prefixes with the library. CI must build the library and a
minimal consumer against Aconex's AGP/Kotlin/Room/Compose versions as well as the sample app. Library
storage uses its own database and AppSearch names and must not require a migration of Aconex's main
Room database.

### 64-bit and 32-bit compatibility

The locally resolved LiteRT-LM `0.13.1` and `0.16.1` Android artifacts contain native libraries only
for `arm64-v8a` and `x86_64`; neither contains `armeabi-v7a`. The on-device generation runtime is
therefore a 64-bit feature even though the pure `rag-api` contracts are ABI-neutral.

Use this policy:

1. **Preferred production path:** make the AI-enabled Aconex build `arm64-v8a` only. Aconex currently
   has minSdk 33, so validate the remaining 32-bit population with supported-device and enterprise
   deployment telemetry before removing `armeabi-v7a`. Do not infer safety from Android version
   alone.
2. **If 32-bit installation must remain supported:** produce explicit architecture variants. An
   `ai64` variant includes `rag-runtime`, LiteRT-LM and `arm64-v8a`; a `legacy32` variant includes
   `rag-api` plus an `UnsupportedRagClient`, excludes the native RAG runtime dependency, and packages
   `armeabi-v7a`. The legacy implementation returns a typed `UNSUPPORTED_ABI` capability instead of
   throwing or attempting native class loading.
3. Keep `x86_64` only for emulator/developer testing unless a desktop/ChromeOS product requirement
   explicitly needs it.
4. Do not rely solely on catching `UnsatisfiedLinkError`. Resolve capabilities before exposing the
   feature, downloading a model, binding the service or referencing native runtime classes.

Capability reporting should include at least:

```text
RagCapabilities
  processIs64Bit
  runtimeAbiSupported
  modelSupported
  modelInstalled
  embeddingInstalled
  availableStorageBytes
  memoryTier
  generationBackends
  indexingAvailable
  chatAvailable
  unavailableReason
```

Use `Process.is64Bit()` plus the packaged/supported ABI contract, not only `Build.SUPPORTED_ABIS`.
On unsupported devices, hide or disable model download and indexing before any 1.7 GB transfer. The
rest of Aconex remains fully usable.

A single universal build that contains other `armeabi-v7a` native libraries but only 64-bit
LiteRT-LM can install on a 32-bit device and later fail if an AI code path loads the missing native
runtime. Architecture variants that exclude `rag-runtime` from the 32-bit artifact are safer than a
runtime check around an otherwise packaged native feature.

### Inference-process integration

Declare the heavy service as a non-exported `:rag_inference` process unless profiling proves an
in-process runtime is safer. Android creates the host `Application` in every application process.
Before enabling this service in Aconex, add an early process-role branch to
`AconexApplication.onCreate()` so the inference process does not initialise Dagger's full UI graph,
Realm/Room migrations, WorkManager jobs, analytics, upload queues, document viewers or activity
lifecycle observers.

The inference branch should call `super.onCreate()`, perform only the minimum safe host setup needed
by the library service, and return. The service owns and lazily constructs Gemma, EmbeddingGemma,
AppSearch and extraction runtimes. Library components must not cast `applicationContext` to the
sample `RagApplication` or Aconex's `AconexApplication`.

All manifest components must be non-exported unless explicitly required, use `${applicationId}` for
authorities, and avoid generic provider/permission names that could collide with host components.
Binder contracts must stay below transaction limits; large PDFs, models, bitmaps and embeddings are
never transferred as byte arrays.

### Aconex viewer adapter

Aconex implements `CitationNavigator` using its existing `DocumentViewer`, `MarkUpPdfViewer` or the
appropriate document feature entry point. The adapter must:

- resolve the scoped document and revision through Aconex authorization and storage APIs;
- open the existing `PDFViewCtrl` rather than a library-owned viewer;
- navigate to the one-based cited page after document load;
- apply source-anchor evidence as a temporary, non-persisted overlay in evidence mode;
- preserve existing Aconex `ToolManager`, author, toolbar, markup and save behaviour in markup mode;
- remove evidence overlays without adding them to annotation modification/upload tracking;
- report unavailable, superseded or unauthorised documents without exposing raw paths.

The library provides source anchors and citation semantics; Aconex remains the authority for whether
the user may view, annotate, export or upload the document.

### Model distribution and privacy

The library must not embed a production CDN URL, access token or Aconex authentication dependency.
`ModelArtifactProvider` supplies a signed/approved manifest containing artifact ID, version, size,
SHA-256, compatible library/runtime version and download request. The library owns resumable copy,
atomic install, checksum validation and rollback; the host owns authentication, rollout and feature
eligibility.

Indexes, transcripts and models remain on device. Telemetry must use allow-listed structured fields
and must not emit prompts, answers, extracted text, file paths, account/project IDs or model tokens.
Diagnostic export containing such data requires an explicit separate user-controlled flow.

### Library acceptance criteria

- The sample app builds and passes the existing JVM/device RAG suites using only published module
  APIs, with no duplicate implementation under the app package.
- A minimal consumer and Aconex compile with one Apryse version and one LiteRT-LM version.
- The library has no compile-time dependency on `com.oracle.aconex.*` and `rag-api` has no native,
  Apryse, LiteRT, Room, AppSearch or Compose surface types.
- Account/project/revision isolation and every cleanup lifecycle event are covered by tests.
- `ai64` runs indexing and generation on a real arm64 device; `legacy32` installs and returns
  `UNSUPPORTED_ABI` without loading LiteRT or offering model download.
- Aconex startup in `:rag_inference` skips normal application initialisation and the main process
  retains its existing behaviour.
- Existing Aconex form AI and RAG cannot initialise competing Gemma engines concurrently.
- Citation navigation reuses the Aconex viewer and cannot bypass document authorization.
- Consumer R8, manifest merge, ABI packaging, process death and library upgrade tests pass for debug
  and release-equivalent builds.

### Delivery sequence

1. Freeze `rag-api` contracts, ownership boundaries, scoped document identity and capability/error
   types before moving implementation files.
2. Extract reusable code into library modules while keeping the current app green as `sample-app`.
3. Add consumer build/R8/manifest tests and publish a local versioned AAR or Maven artifact.
4. Converge Aconex and the library on the chosen Apryse and LiteRT-LM versions.
5. Implement the `ai64`/`legacy32` packaging decision and capability gates; validate real deployment
   requirements before removing 32-bit support.
6. Make `AconexApplication` process-aware and integrate the `:rag_inference` service client.
7. Route Aconex's existing Gemma form backend through the shared runtime and prove scheduling,
   cancellation and memory behaviour.
8. Implement Aconex document-source, lifecycle-cleanup, model-distribution, telemetry and citation
   viewer adapters.
9. Run the full RAG evaluation, existing form-AI regression suite, viewer/markup tests and performance
   tests on Pixel 10 and Nothing Phone from an Aconex build.

Complete this architectural increment before adding more host-specific features to the standalone
application. The library API may remain pre-1.0 while integration reveals necessary contract changes,
but persisted index formats and source-anchor versions must always migrate explicitly or request a
clean reindex.

## Citation-to-original-PDF evidence viewer

### Value

Let a user verify an AI answer against the original PDF instead of only reading the extracted chunk
in the citation sheet. Tapping a citation opens the app-managed source document at its cited page and
visually highlights the supporting text when the app can identify it deterministically.

The experience is a full Apryse PDF viewer. `PDFViewCtrl` supplies rendering, navigation, zoom and
selection, while `com.pdftron.pdf.tools.ToolManager` supplies the interaction and markup capabilities
needed by the product. Apryse `TextSearch` is suitable for page-local exact matching and supplies
highlight geometry. It is not a semantic search engine and must not replace AppSearch hybrid
retrieval.

### Architecture decision

The production Aconex integration reuses the existing Aconex `PDFViewCtrl`/`ToolManager` viewer in
the Aconex UI process through `CitationNavigator`. Aconex already loads Apryse for document viewing
and markup, so the RAG library must not introduce another production viewer or `:viewer` process.

The standalone sample app may implement its adapter with a Compose-hosted `PDFViewCtrl` and
`ToolManager`. Initialise Apryse lazily when its first viewer opens; it must not be loaded during
ordinary sample chat startup. This deliberately changes the sample's current rule that keeps Apryse
entirely inside `:inference`.

In both hosts, the UI/viewer process and `:rag_inference` may contain an Apryse native runtime while a
viewer and indexing or generation coexist. Measure combined peak memory on Pixel 10 and Nothing
Phone. A separate viewer process would add lifecycle and IPC complexity without automatically
reducing aggregate memory and is outside the planned production architecture unless later device
evidence demonstrates a clear need for crash isolation.

Do not host any viewer Activity in `:rag_inference`. Mixing UI and `PDFViewCtrl` with Gemma,
EmbeddingGemma, AppSearch and indexing would weaken the intended failure and memory isolation.

### Scope and flow

```text
Chat answer citation
  -> existing CitationSheet resolves (indexNamespace, chunkId)
  -> build a trusted CitationTarget including scoped documentKey, page and SourceAnchor
  -> call the host-provided CitationNavigator
  -> host resolves the scoped document revision and verifies authorization
  -> open the host's PDFViewCtrl at citation.pageNumber
  -> apply retained PDF-space geometry when it is trustworthy
  -> otherwise search only that page for the retained exact source text
  -> highlight only one deterministic match
  -> if no deterministic match exists, show the cited page without a highlight
```

The source document must be resolved from its trusted scoped document key and the host-managed
document record. The viewer must never accept an arbitrary file path from navigation state or an
external Intent. It must check that the file still exists, belongs to the expected revision and is
still authorised before opening it.

### Citation identity and source-anchor contract

`chunkId` is only stable inside an AppSearch namespace and is reused by subsequent index builds. The
canonical citation lookup identity is the pair `(indexNamespace, chunkId)`; never infer a namespace
by parsing a chunk ID.

The viewer navigation contract should be equivalent to:

```text
CitationTarget
  documentKey: scoped account/project/document/revision/content identity
  indexNamespace
  chunkId
  pageNumber
  sourceAnchor
```

An old chat may refer to a namespace removed during reindexing. In that case the app may still open
the verified source document and cited page, but it must not claim an exact highlight when the cited
chunk or its source anchor can no longer be resolved.

Add a versioned source anchor captured during extraction and propagated through structure analysis,
chunking, AppSearch storage, `Citation`, and the citation sheet:

```text
SourceAnchor
  anchorVersion
  sourceKind: TEXT_LAYER | OCR | NONE
  pageNumber
  exactRawText
  prefixContext
  suffixContext
  occurrenceOrdinal
  pdfSpaceQuads
```

- `exactRawText` is copied from the original PDF text layer before Unicode normalisation, whitespace
  normalisation, header/footer stripping, row reflow or table reconstruction.
- Retrieval continues to use normalised `bodyText` and `retrievalText`; source evidence is stored
  separately and is never included merely to improve ranking.
- `pdfSpaceQuads` should preserve the contributing words or lines rather than only one large union
  rectangle. Disjoint rectangles are required for wrapped sentences and reconstructed table rows.
- `prefixContext`, `suffixContext`, and `occurrenceOrdinal` disambiguate repeated exact strings on the
  same page when retained geometry cannot be applied.
- An OCR anchor may retain OCR geometry for a later image-overlay implementation, but must not be
  passed to native PDF text search when the page has no searchable text layer.

Adding this contract changes persisted index data and requires an index-version increment and clean
reindex during development.

### Highlight resolution

Use this ordered strategy:

1. If the source anchor contains trustworthy PDF-space quads for the verified source file and page,
   render a temporary evidence overlay at those coordinates.
2. Otherwise, for a `TEXT_LAYER` anchor with non-blank `exactRawText`, run Apryse `TextSearch` with
   `e_page_stop` and highlight mode on the cited page only.
3. Use retained context or occurrence information to select the intended result. Never silently
   choose the first result when multiple candidates remain plausible.
4. If the match is absent or ambiguous, open the cited page without a highlight and label the state
   **Source page**. Keep the extracted citation text available as the accessible alternative.

A search miss is not evidence that retrieval or the citation was wrong: the text layer may differ
from extracted output because of ligatures, CJK mappings, unusual encodings, reflow or document
replacement. Log the reason diagnostically without showing an alarming error to the user.

### Viewer and threading lifecycle

- Resolve the Apryse `tools` artifact at exactly the same version as `pdftron`. Aconex owns these
  production dependencies; the sample app declares them for its reference adapter.
- Initialise Apryse once per UI process. The sample does so lazily; Aconex retains its established
  viewer initialisation lifecycle.
- Create and mutate `PDFViewCtrl` and apply visual search/highlight results on the main thread.
- Perform file validation and page-local search away from the main thread using Apryse's required
  document locking/thread-safety rules. Do not access the same `PDFDoc` concurrently without the
  appropriate lock.
- Give every open or citation selection a request ID. Ignore late results belonging to a previous
  request, cancel active search when the citation changes, and close its native handle.
- On viewer disposal, detach listeners and `ToolManager`, clear temporary evidence overlays, close
  the `PDFDoc`, destroy the `PDFViewCtrl`, and release other viewer-owned resources.
- Restore the page, zoom and viewer mode after configuration recreation without repeating stale
  searches or leaking the old Android view.
- Opening the viewer during generation must not cancel, corrupt or deadlock the active Gemma turn.
  If device measurements show unsafe memory pressure, use an explicit user-visible memory policy
  rather than relying on low-memory process death.

### Evidence mode and markup mode

The viewer supports two deliberately separate modes:

1. **Evidence mode** opens from a citation. The indexed source PDF is read-only, citation highlights
   are temporary visual overlays, and no operation may save into the indexed PDF.
2. **Markup mode** enables the required `ToolManager` annotation capabilities. Aconex retains
   ownership of its existing annotation, working-file, upload and audit behaviour. The sample app
   persists annotations in an app-managed working copy or sidecar/XFDF associated with the document
   key. Neither host silently modifies the immutable PDF revision that produced the active index.

If the product later allows a marked-up PDF to become the searchable source, treat that as a new
document version: generate its content hash, index it into a new namespace, and keep old citations
bound to the previous immutable source version.

Disable annotation creation, form mutation and saving while in evidence mode. External links,
embedded actions and JavaScript must follow an explicit security policy rather than executing merely
because the document is open.

### Performance and memory gates

Before release, establish baselines on both target devices and convert them into enforced release
budgets. Initial targets for representative local PDFs are:

- cited page visible within 2 seconds at p95 for a cold viewer open and within 750 ms at p95 when the
  viewer runtime is warm;
- deterministic highlight visible within 1 second at p95 after the page is visible;
- switching between citations on already-open documents reflected within 500 ms at p95;
- no disk access, PDF search or native document work that blocks the main thread long enough to
  produce an ANR or visible sustained jank;
- no UI or inference process death while a supported Gemma model and the viewer coexist under the
  documented device-memory tier.

Record UI-process RSS, inference-process RSS, combined peak RSS, native heap before/after viewer
close, page-open latency and search latency. Repeatedly open and close the viewer to detect retained
native allocations. Revise numerical latency targets only from captured device evidence and document
the reason.

### Failure behaviour

- Missing, deleted, changed, malformed, password-protected or unsupported PDFs must produce a safe
  unavailable state and keep chat usable.
- A stale or missing AppSearch namespace may fall back to verified page-only navigation, never to a
  citation in another namespace.
- OCR-only pages, ambiguous matches and text-search misses use page-only fallback without a
  misleading highlight.
- Viewer or search failure must not delete the index, mutate the source, mark the citation invalid or
  terminate an active chat turn.

### Test plan

Pure unit tests should cover:

- citation identity and stale-namespace resolution;
- exact-source anchor selection before normalisation and reflow;
- prefix/suffix and occurrence disambiguation for repeated phrases;
- geometry propagation through paragraph splitting, overlap, merging and table reconstruction;
- evidence-versus-markup persistence policy;
- late-result rejection and cancellation state transitions.

Instrumented tests should cover:

- text-native exact match and correct intended occurrence;
- no-match, ambiguous-match and OCR page-only fallback;
- CJK text, ligatures, hyphenation, RTL text, wrapped sentences and tables;
- rotated and cropped pages and multi-column layouts;
- repeated phrases on the same page;
- rapid citation switching, close-during-search and reopen;
- configuration recreation, background/foreground, UI-process death and inference-process death;
- missing, deleted, malformed, encrypted and hash-mismatched source files;
- evidence mode does not persist annotations or other PDF changes;
- markup mode writes only to its working copy or sidecar and does not invalidate the active index;
- viewer/Gemma coexistence, repeated open/close memory behaviour and the latency gates above.

Manual accessibility testing must cover screen-reader labels, focus order, large text, keyboard
navigation where supported, zoom controls, highlight contrast and access to the extracted citation
text when visual evidence is unavailable.

### Acceptance criteria

- A citation opens the intended immutable document and correct one-based page.
- Citation lookup always uses `(indexNamespace, chunkId)` and cannot cross index versions.
- Fixture citations with retained geometry or one unambiguous exact match highlight the intended
  supporting text, including repeated-text fixtures.
- Scanned/OCR-only pages and absent or ambiguous matches open without error and never display a
  misleading highlight.
- Evidence mode cannot mutate the indexed PDF; markup is isolated in its working copy or sidecar.
- Opening, searching, switching and closing meet the measured Pixel 10 and Nothing Phone latency and
  memory gates without cancelling or corrupting chat generation.
- All native handles, listeners and temporary overlays are released across close, recreation and
  cancellation paths.

### Delivery sequence

1. Complete the reusable-library contracts and Aconex dependency/process convergence described in
   the preceding major increment.
2. Add the versioned raw-text/geometry source-anchor contract and reindex fixtures.
3. Implement the sample `CitationNavigator` with a Compose-hosted `PDFViewCtrl`/`ToolManager`, then
   implement the Aconex adapter using its existing viewer.
4. Deliver read-only evidence mode with deterministic highlight and page-only fallback without
   entering Aconex annotation modification tracking.
5. Connect markup entry points to Aconex's established markup workflow and keep the sample working
   copy/sidecar implementation as a reference fixture.
6. Add broader document navigation and explicit **Find in document** controls. Exact string,
   whole-word and regex search remain separate from the chat/RAG query field.
7. Reconsider viewer process isolation only if captured production evidence justifies its additional
   lifecycle and IPC complexity; it is not part of the planned Aconex integration.

### Prerequisite / priority

Implement after the core single-PDF retrieval and citation flow is considered product-ready and while
extracting the reusable library/Aconex adapters. This is a trust, review and usability improvement—not
a retrieval-performance project—and it must not weaken the fast, fully on-device answer path.
