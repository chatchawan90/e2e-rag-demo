# Development journal

**Project owner: Chatchawan Lakkhananukun.** Copyright © 2026 Chatchawan Lakkhananukun.

Recorded on **1 October 2026 (Asia/Bangkok)**. This journal reconstructs the work from
the development conversation, the resulting source and documentation, and verification
results recorded during the session. Earlier milestones are in conversation order;
individual implementation dates were not recorded. This is a project history, not a
verbatim conversation or a log of visitor activity.

## Current outcome

- Browser demo: <https://envsearch-demo.onrender.com/>
- Remote MCP: <https://envsearch-demo.onrender.com/mcp>, Streamable HTTP, anonymous.
- Public reference library: **11 documents and 479 passages**, covering US and Thai
  environmental regulations.
- English and Thai use the same multilingual embedding model and index.
- The local app supports uploads, evaluation runs, gold editing, monitoring and optional
  reranking. The public demo limits these features as described below.
- Code is published at <https://github.com/chatchawan90/e2e-rag-demo>.

See [the application guide](README.md) for usage and [the deployment guide](deploy/README.md)
for hosting instructions. Those files describe the implementation; this file records
why it evolved and what was verified.

## 1. Simplify retrieval and add metadata filters

The initial discussion concerned a simpler hybrid retrieval design and whether Thai
needed a separate vector database. We kept one index and combined BM25 keyword retrieval
with local multilingual vectors using reciprocal-rank fusion. A NumPy embedding matrix
is sufficient for this demo; an external vector database is not required.

- Embedding model: `intfloat/multilingual-e5-small`, using `query: ` and `passage: ` prefixes.
- Local vectors: `data/index/embeddings.npy`. `vectors.json` records the model, ordered
  chunk IDs and a source-text fingerprint to detect stale vectors.
- Model weights use the Hugging Face cache; `HF_HOME` can change its location.
- Search modes: `auto`, `bm25`, `vector`, `hybrid`. Auto uses hybrid when vectors exist.
- Jurisdiction, passage language, document IDs and custom document metadata are filtered
  before candidate selection. English questions can retrieve Thai passages when the
  language filter is left unrestricted.
- Thai keyword matching normalizes extracted PDF characters and Thai digits, and uses
  word segmentation with additional regulatory terms. Thai words are not reliably
  separated by spaces, and extracted spellings can differ from typed text.

We retained optional bilingual query expansion so hybrid retrieval without expansion
could be compared with the existing baseline, especially on cross-language examples.
Retrieval scores are ranking signals, not probabilities of relevance.

The index is an explicit build artifact: fetching PDFs or installing the package does
not build it. `envsearch build --vectors` prepares the hybrid index. `envsearch` is this
repository's Python package; its installed CLI entry point is why commands can be run
without typing `python` first.

## 2. Add a local browser app and PDF ingestion

The user chose a local application initially. We added Streamlit chat, a passage-finding
mode, sidebar filters, source citations and PDF page previews. Recent conversation is
used to resolve follow-up questions; conversation history stays in browser-session state.

The ingestion pipeline validates PDFs, extracts and normalizes page text, chunks content,
adds document metadata, embeds new chunks and updates the index. It detects duplicates
by file content. Uploaded PDFs and metadata persist under `data/uploads/`; rebuilds
include them. Staging, locking and atomic index replacement protect index consistency.

The local pipeline accepts text-based PDFs up to 50 MB and 1,000 pages. It rejects
encrypted, invalid and textless inputs with guidance. OCR was not implemented.

During frontend integration the user reported an import error for `answer` in
`envsearch.chat`. The resulting application uses the compatible chat interface and was
subsequently exercised by frontend tests and the deployment checks below. The original
exception was part of the development troubleshooting, not a remaining deployment error.

## 3. Make retrieval limits visible and add optional reranking

We clarified that search does **not** return all relevant passages. Each retrieval branch
selects a candidate pool, fusion combines it, and a document cap and final top-k cutoff
limit results. The default candidate pool is `max(4 * k, 20)` per branch/query, with
default `k=8` and at most three returned passages per document.

The local app exposes these settings and a **How passages were selected** trace.
Optional reranking uses `BAAI/bge-reranker-v2-m3` on the fused candidate pool before the
document cap and final cutoff. It cannot recover passages missing from that pool.
Reranking remains off by default because of its model download and CPU cost.

## 4. Add provider choice and scope routing

The user requested entering their own OpenAI key in the app. We added OpenAI alongside
Anthropic, with provider selection and session-held keys. The OpenAI integration uses
the Responses API with `store=False`; numbered citations map to supplied passages.
Local environment keys remain an alternative, and an entered key takes precedence.
Switching providers can send existing conversation context to the newly selected provider
on the next request; starting a new conversation clears that context.

An unrelated Thai question meaning “What is my name?” demonstrated a problem: nearest
passages could be returned even though the question was unrelated to the library.
The user chose a strict environmental-domain scope policy.

We added a scope check before retrieval, returning one of: in scope, out of scope, or
needs clarification. Out-of-scope and clarification responses retrieve no passages and
show no citations. The same call resolves follow-up references from recent conversation.
An environmental question can remain in scope even if the library lacks its answer.

The user then demonstrated that **Find passages** also needed the gate. Both browser
modes now use it and require a provider key. Classifier failures stop the request instead
of silently retrieving. CLI/MCP raw search and retrieval-only evaluation deliberately
remain direct retrieval. The cited-answer pipelines use scope checking.

No general web-search tool was added. Fetching reference documents and opening publisher
links are separate from searching the web to answer a question.

## 5. Add evaluation, monitoring and editable gold examples

Evaluation compares results against labeled questions. Real user questions can become
test cases after a person supplies and reviews the expected outcome and sources; live
traffic is not automatically assigned correctness scores.

- Retrieval metrics: Hit@k, MRR, known-gold document Recall@k and document nDCG@k.
- Answer checks: required keyword groups, references to gold documents, citation precision,
  false refusals, correct refusals and expected scope decisions.
- Gold labels are not supplied to the answering model and do not train it.
- Tags group evaluation slices such as English, Thai, cross-language and negative cases.
  They are evaluation labels, not retrieval filters.
- Local additions are stored in `data/gold/questions.json`, separately from the 24 seed
  cases in `evals/questions.yaml`, and become available in the frontend immediately.
- The gold editor validates structure and source/page availability, rejects duplicates,
  and uses locking and atomic writes. Humans still verify factual correctness.
- Saved reports preserve the questions, labels and outputs used for that run.

Recall and nDCG measure coverage of the labeled documents, not exhaustive passage
relevance. Document recall/nDCG do not use page labels; Hit@k/MRR honor page constraints.
Keyword and citation-target checks do not independently prove factual correctness.
Seed facts require verification against the PDFs before scores are treated as reliable.

Local monitoring writes operational events to `data/monitoring.sqlite3`: timing, errors,
retrieval counts, settings and final-answer token usage, with separate scope trace fields.
It excludes question text, answers, passages and keys. Evaluation reports separately
contain test questions and generated answers. Token reporting is not complete billing
accounting. MCP calls are not recorded in the web Monitoring tab.

We documented possible business measures—verified task resolution, time saved, appropriate
expert escalation, cost per resolved task and audited material errors—but did not implement
business-outcome collection or infer these measures from retrieval scores.

## 6. Make evaluation and gold pages easier to understand

The user requested clearly separated sections, plain explanations and hover help.

The Evaluation page now separates question selection, test type, execution and results.
It uses bordered sections, plain-language metric labels, help text, expandable advanced
settings and per-question expected-versus-actual results.

The Gold dataset page explains what an answer key is, why repeatable examples matter,
and why saving examples does not train the model. Browsing, adding examples and
checking/exporting the dataset are separate sections. Technical IDs and labels are
available under advanced details. A recent conversation question can be copied for
manual labeling; model answers are never automatically promoted to gold.

## 7. Explain MCP and prepare an independently hosted demo

The original MCP integration used FastMCP over stdio: a client launches a local process.
It is separate from the Streamlit browser server. The local tool set contains
`search_regulations`, `get_passage`, `list_documents` and the optional Anthropic-backed
`ask_with_citations`.

We expanded the Tools page with concrete client configuration, commands, key requirements,
verification prompts, MCP Inspector guidance and troubleshooting.

The user then requested both a browser demo and a real remote MCP URL, on free hosting
independent of their machine. A local tunnel was rejected. After investigating hosting,
we prepared a Render Free Docker deployment, and the user created the account and
launched the service. A paid Hugging Face hosting option was not used.

One container exposes one public port through nginx:

| Public route | Service |
| --- | --- |
| `/` | Streamlit browser app |
| `/mcp` | FastMCP Streamable HTTP |
| `/healthz` | Index health and document/passage counts |

Streamlit and the HTTP MCP server share one Python process and embedding runtime.
The public HTTP server exposes only the three read-only retrieval tools. It has an
explicit public-origin allowlist, a 64 KiB request-body limit and bounded search inputs.
Retrieval runs off the HTTP event loop with one expensive query at a time.

## 8. Separate the public demo from local user data

We added a public-only snapshot exporter. It includes only indexed public HTTP(S)
manifest documents with matching source URLs and no local upload path. The exported
library contains 11 documents and 479 passages; local user uploads are excluded.

The deployment includes the public PDFs, document/chunk records and vectors. It excludes
local uploads, private gold additions, reports, monitoring databases, `.env` and keys.
Docker uses an explicit build-context allowlist.

With `ENVSEARCH_PUBLIC_DEMO=1`, uploads, gold edits, evaluation execution, shared activity
logging and reranking are disabled. Starter gold examples remain viewable. Source PDFs
open at their publisher URLs. Browser visitors supply their own model API keys; the
public app ignores server key fallback and does not load `.env`. Remote MCP retrieval
needs no server-side answer key and exposes no paid answer tool.

Hosting is free under the selected plan; visitors' model-provider calls can still cost
money. Free hosting sleeps after inactivity and has usage limits. There is no paid disk
or dependency on durable runtime writes. Deployment instructions explain cold starts and
quota suspension. Availability and current hosting terms should be checked when changing
the deployment.

## 9. Fit the service within the free host's memory limit

Initial container experiments exceeded the 512 MiB target. We kept the local full-precision
backend and introduced a compact hosted backend using the same multilingual E5 model:

- Pinned model revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- Official int8 ONNX weights, externalized at build time to reduce startup copies.
- SentencePiece tokenization with the model's XLM-R token mapping.
- Hosted query limit of 256 model tokens and conservative ONNX CPU/memory settings.
- Encoder initialization before loading the web stack, and one shared runtime.
- Lazy local PDF imports, publisher links instead of hosted PDF rendering, and a compact
  gold question picker to avoid unnecessary memory use.

Review and integration testing also identified and fixed:

- Installed-package asset paths: Docker uses an editable install so repository assets resolve.
- Streamlit programmatic startup: configuration options are explicitly loaded before boot.
- Blocking HTTP retrieval: expensive search is moved to a bounded worker thread.
- Non-root nginx startup: all required temporary directories are explicitly under `/tmp`.
- Render YAML: `autoDeployTrigger: "off"` is quoted so YAML does not parse it as a boolean.

The final Linux AMD64 container passed checks with a 512 MiB memory limit and no extra
swap allowance. Recorded steady usage was about **336.7 MiB**, with peak usage
**357,949,440 bytes (about 341.4 MiB)** and no OOM in that verification run. This is a
smoke-test measurement, not a concurrent-load capacity guarantee.

## 10. Verification recorded during implementation

These results were recorded during the development session; they were not rerun merely
to write this journal.

| Check | Recorded result and scope |
| --- | --- |
| Full automated suite | **127 passed**; five PyMuPDF SWIG deprecation warnings. Includes retrieval, ingestion, scope, providers, UI, gold, public-mode and HTTP MCP tests. |
| Tokenizer comparison | SentencePiece token IDs matched the reference fast tokenizer for all 24 seed questions. |
| Embedding comparison | One English query had ONNX-versus-full-precision cosine similarity of **0.9976283**; this does not establish identical rankings for all queries. |
| Compact-backend seed evaluation | 24 cases, hybrid at k=8: Hit@k **1.0**, MRR **0.812**, document Recall@k **0.975**, document nDCG@k **0.867**. Metric eligibility rules and seed-label caveats apply. |
| Final Docker integration | Linux AMD64 image, no source mounts, 512 MiB memory limit; browser session protocol and real HTTP MCP retrieval passed. |
| Public HTTP protections | Tests cover the three-tool set, absence of the paid answer tool, query/body limits and Host/Origin validation. |

Paid model API calls were not part of the deployment smoke checks. Browser validation
used Streamlit tests and its actual WebSocket session protocol. In-app browser automation
could not initialize (`Cannot redefine property: process`), so these results do not
claim screenshot-based visual testing or an end-to-end paid chat test.

## 11. Publication and live verification — 1 October 2026

Published Git history before this journal:

| Commit | Time (Asia/Bangkok) | Change |
| --- | --- | --- |
| `418b5d6` | 13:41 | Application, tests, public snapshot and free browser/MCP deployment. Earlier development was consolidated into this initial commit. |
| `f48943a` | 13:43:20 | Keep the Render deployment trigger an explicit string. |

The user supplied the deployed URL, and we ran:

```bash
.venv/bin/python deploy/check_demo.py https://envsearch-demo.onrender.com
```

The command exited successfully with:

```text
Health: {'status': 'ok', 'documents': 11, 'passages': 479}
Public browser interface: passed
Public HTTP MCP tools: passed
Hybrid search passed; query length: 21
Hybrid search passed; query length: 49
Hybrid search passed; query length: 5500
```

This checked the health endpoint, a rendered Streamlit session, MCP initialization,
the exact public tool list, document listing, and hybrid search for a Thai query, an
English query and a long query. Search assertions checked successful passage results,
not the correctness of an answer. `get_passage` was present in the live tool list but
was not called by this smoke script. No visitor key or paid answer call was used.

Render automatic deployment is disabled in the blueprint. Future published changes
require an explicit deployment; updating GitHub alone does not verify or update the
running service. The live results above describe the deployment checked on this date.

## 12. Record project ownership — 1 October 2026

At the user's request, Envsearch identifies **Chatchawan Lakkhananukun** as its owner.
The attribution appears in the shared browser header, sidebar, footer and browser title;
local and public MCP setup pages and server initialization instructions; CLI help;
newly generated Markdown evaluation reports; Python package author/maintainer metadata;
Docker image metadata; and the project documentation. Shared browser attribution remains
visible across the application's tabs. Machine-readable retrieval results and source
citations retain their original document attribution.

The root `COPYRIGHT` notice identifies the project owner and preserves the attribution
and terms of source documents, third-party software and model weights. It is included
in the Docker image. The completed development journal is linked from the README.

The user reported redeploying. A fresh run of `deploy/check_demo.py` against the public
URL passed health, browser-session, MCP tool discovery, and Thai, English and long-query
hybrid retrieval checks. This checked the existing live service before publishing the
ownership changes; a subsequent Render deployment is needed to display the new name.

Validation after the ownership update: **127 tests passed** with the same five PyMuPDF
warnings. Additional checks confirmed the public app's owner captions and footer, the
HTTP initialization response and local stdio server instructions, CLI help, and package
author/maintainer metadata. A first manual HTTP check omitted the local Host port and
was correctly rejected by the existing allowlist; using `localhost:8602` passed.

## Remaining limitations and possible next work

- Add web search only if the product scope calls for it; it is not implemented.
- OCR and layout-aware table extraction remain absent. Split table context can require
  neighboring passages, and the library does not establish which laws are currently in force.
- Scope classification can falsely reject or accept a question; extend reviewed test cases.
- Broader relevance labels and a held-out evaluation set would strengthen retrieval claims.
- Business-outcome instrumentation and public visitor analytics are not implemented.
- Authentication, per-user document isolation and durable storage are needed before
  evolving the public reference demo into a private multi-user application.
- Concurrent-load testing and live paid-provider chat verification remain separate work.

For future entries, record the date, requested change, implementation decision, relevant
commit, actual checks and remaining limitations. Keep credentials and private conversation
content out of this journal.
