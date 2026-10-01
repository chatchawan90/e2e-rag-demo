# envsearch

**Owned by Chatchawan Lakkhananukun.** Copyright © 2026 Chatchawan Lakkhananukun.
See [the ownership notice](COPYRIGHT) for the scope of this attribution.

Cited question-answering over public environmental regulations, in **English and Thai**:

- **US EPA** hazardous-waste generator guidance (8 PDFs: satellite accumulation, generator categories, recordkeeping, empty containers, universal waste, characteristics, spills)
- **Thailand**: the Ministry of Industry notification on industrial waste management (B.E. 2566), and the MoI and MNRE factory effluent standards

Ask in either language and it answers from either corpus. Claude uses native `search_result` citations; the web app also supports OpenAI with numbered references to retrieved passages. Citation links identify sources, not independently verified claims. The repo includes a 24-item eval set with code-checked metrics, plus an MCP server so the same index works inside Claude Desktop.

```
$ envsearch ask "What is the maximum BOD allowed in factory wastewater in Thailand?"
Under Thailand's Ministry of Industry effluent standard, BOD in discharged factory
wastewater must not exceed 20 mg/L.[1] ...

Sources:
  [1] ประกาศกระทรวงอุตสาหกรรม เรื่อง กำหนดมาตรฐานควบคุมการระบายน้ำทิ้งจากโรงงาน พ.ศ. 2560
      — https://e-report.diw.go.th/...pdf#page=2
       "(4) บีโอดี (Biochemical Oxygen Demand) ไม่เกิน 20 มิลลิกรัมต่อลิตร"
```
*(This shows the output format. Run it yourself for real output.)*

---

## Quick start

For the implementation history, design decisions and recorded verification results,
see the [development journal](DEVELOPMENT_JOURNAL.md).

For a free hosted browser demo and remote MCP URL, see [Deploy on Render](deploy/README.md).
The deployment uses a separate public-only library and visitors' own chat API keys.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                          # offline tests on a synthetic fixture corpus

envsearch fetch                    # downloads the 11 PDFs in corpus/manifest.yaml into data/raw/
envsearch build                    # extract -> normalise -> chunk -> BM25 index in data/index/
envsearch eval verify              # checks the eval set's gold answers really appear in the PDFs

envsearch search "satellite accumulation area 55 gallons"      # no API key needed
export ANTHROPIC_API_KEY=sk-ant-...
envsearch ask "ค่า pH ของน้ำทิ้งจากโรงงานต้องอยู่ในช่วงใด" --debug
envsearch eval run                 # writes evals/reports/eval-<timestamp>.{md,json}
```

Models are set by environment variables: `ENVSEARCH_ANSWER_MODEL` (default `claude-sonnet-5-5`) and `ENVSEARCH_REWRITE_MODEL` (default `claude-haiku-4-5`). Top-k is `ENVSEARCH_TOP_K`, default 8.

## Local chat and PDF uploads

```bash
source .venv/bin/activate
pip install -e ".[dev,vectors,web]"
envsearch serve
```

Open **http://127.0.0.1:8501**. The app only listens on your own machine.
Use `envsearch serve --port 8502` if you need a different port.

- **Conversation:** type a question in English or Thai, see the answer stream live, and
  expand its citations to view the exact PDF page. Follow-up questions use the recent
  conversation to resolve references before retrieval. Conversation history stays in the
  browser session; **New conversation** clears it.
  Chat is strictly scoped to environmental regulations and compliance. A lightweight
  scope check runs before retrieval: unrelated requests get a short localized refusal;
  ambiguous requests get a clarification. Both return no passages or citations. In-scope
  questions proceed to retrieval even when the library may lack the answer. Recent
  conversation resolves follow-ups in this same call, replacing the separate follow-up rewrite.
- **Find passages:** uses the same scope check as chat, then shows source excerpts without
  generating an answer. Both conversation modes require a provider key for scope checking.
  Without a key, the input stays disabled; the app does not switch to an unchecked search.
  Sidebar filters cover jurisdiction,
  topic, documents, passage language, and retrieval mode.
- **Add PDFs:** upload a PDF, optionally set its title, jurisdiction, topic, and other
  `key=value` metadata, then select **Add to library**. Progress shows extraction,
  chunking, embedding, and saving. New documents are available immediately.
- **Library:** inspect document metadata and preview source PDFs.
- **Evaluation:** follow three bordered steps: choose questions, choose whether to check
  sources or sources and answers, then run the test. A separate Results section shows
  plain-language scores and questions needing attention. Hover the **?** icons for
  explanations, including the technical metric names and their limits. Advanced search
  settings, additional scores, and raw data are expandable. Compare expected and actual
  results per question, or reopen/download saved reports from `data/evals/`.
  Answer evaluation and query expansion use API credits.
- **Gold dataset:** an introduction explains the dataset as a human-reviewed answer key
  and why repeatable test questions help measure quality after changes. Bordered sections
  separate browsing examples, adding questions, and checking/exporting the answer key.
  Hover **?** for explanations of expectations, source passages, keyword groups, and tags.
  Gold labels are used to score evaluation runs, never passed to the answering model;
  saving examples does not train the assistant. Expand sources to read passages and open
  original PDF pages; technical IDs and saved labels are in an Advanced section.
  Use **Add a test question** to create a case immediately, or **Use latest conversation
  question** to copy a real question for manual labeling. Supply the expected outcome,
  source documents, optional pages, required keyword groups, tags, and notes, then save.
  Use one keyword group per line and `|` for alternatives (e.g. `55 | fifty-five`).
  Refusal/clarification cases do not require document or keyword labels. Rewrite follow-ups
  as standalone questions; conversation history is not included in these saved cases.
  Saving requires no API key and no index rebuild. The new case appears in Evaluation
  immediately and persists in `data/gold/questions.json`, separately from the seed set.
  Duplicate questions are rejected and concurrent saves are protected by a lock and
  atomic replacement. The form checks structure and source/page availability; humans
  still need to verify correctness. It never automatically promotes model answers to gold.
  **Check saved examples** checks document text and keywords across the whole dataset;
  it does not validate facts, page labels, or refusal/clarification decisions.
  **Download gold dataset as JSON** exports the combined set in a format accepted by
  `envsearch eval run --questions /path/to/gold-questions.json`.
- **Monitoring:** local search/chat/upload/evaluation timing, errors, retrieval counts,
  and final-answer token usage, stored in `data/monitoring.sqlite3`. The operational log
  excludes questions, answers, passage text, and keys. Evaluation reports separately
  contain the gold questions and generated answers. Token counts exclude query rewriting.
- **Tools:** follow the MCP setup guide: prepare the library, copy this machine's
  configuration into Claude Desktop (or another stdio client), verify real tool calls,
  and choose between client-generated answers and the optional Anthropic answer tool.
  Download a configuration snippet without keys, test directly with MCP Inspector,
  and consult troubleshooting or the separate remote-hosting requirements.

For live answers, select **Anthropic** or **OpenAI**, then enter that provider's API key
in the sidebar. It is held in the local server's browser-session state and is not written
to disk or monitoring logs. Alternatively, configure `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
in the environment or the ignored `.env` file (see `.env.example`). An entered key overrides
the selected provider's environment key. Questions, recent conversation, and retrieved
passages are sent to the selected provider when using chat. Switching providers sends
the existing conversation context to the newly selected provider on the next question;
use **New conversation** to clear it first if desired. Embedding and PDF extraction run locally.

OpenAI uses the Responses API with streaming and `store=False`, defaulting to `gpt-4.1-mini`
(override in Answer settings or with `ENVSEARCH_OPENAI_MODEL`). Its numbered citations
are linked only to supplied passages; this checks the reference target, not whether the
passage proves the claim. OpenAI follow-up rewrites, query expansion, and web answer
evaluations use the same selected model/key. The CLI and MCP cited-answer tool continue
to use Anthropic. Provider selection does not rebuild or change the embedding index.

Scope checking uses the selected provider (Anthropic's rewrite model or the selected
OpenAI model) and requires an API call even for a rejected question. It does not send PDF
content to the classifier. Invalid classifier output or an API failure stops the request;
it never silently falls through to retrieval. The scope is the environmental domain,
not the current corpus or jurisdiction: an unsupported environmental question remains
in scope. Classification is model-based and should be evaluated for false rejections.
The CLI `ask`, MCP `ask_with_citations`, and full answer evaluations use the same gate.
**Find passages** also uses this gate and recent conversation to resolve follow-ups.
CLI/MCP search and retrieval-only evaluations remain raw retrieval for inspecting the index.

The PDF pipeline accepts text-based PDFs up to 50 MB and 1,000 pages. Password-protected,
invalid, and textless PDFs are rejected with guidance; automatic OCR is not included.
Duplicate detection uses the file's content, so renaming a file does not index it twice.
For an existing vector index, only new chunks are embedded with the same model; existing
vectors and page citations are preserved. BM25 uses positive IDF weights so a library
containing just one short PDF is searchable too.

Uploads persist under `data/uploads/<document-id>/source.pdf`, with a `document.json`
metadata sidecar. `envsearch build` includes those uploaded documents on subsequent
rebuilds. Index writes use a staged directory and a lock; readers see a complete index,
and interrupted index replacement can recover from its backup. Restart a running MCP
server after uploading to refresh its cached index.

The same ingestion pipeline is available from the command line:

```bash
envsearch ingest ./new-regulation.pdf --title "Factory wastewater guidance" \
  --jurisdiction TH --filter topic=effluent
```

`ingest` enables vectors by default. Use `--no-vectors` for a new or existing BM25-only
library; an existing hybrid index always keeps its vectors.

## Hybrid retrieval and metadata filters

Add local multilingual vector search to the same index:

```bash
pip install -e ".[dev,vectors]"
envsearch build --vectors          # after fetch; first run downloads the embedding model
envsearch search "What is the BOD limit in factory wastewater?" --mode hybrid --jurisdiction TH
envsearch search "hazardous waste storage" --mode vector --doc epa-saa
envsearch ask "What is the BOD limit?" --mode hybrid --jurisdiction TH --no-expand
```

English and Thai share one embedding matrix using
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small).
No vector database or embedding API key is needed. Model weights are cached by Hugging Face;
set `HF_HOME` to choose that cache location. After downloading, embeddings run locally.
Vectors are saved as `data/index/embeddings.npy`; `vectors.json` records the model, ordered
chunk IDs, and source-text fingerprint so stale vectors fail with a rebuild instruction.

`--mode auto` (the default) selects hybrid if the index has vectors, otherwise BM25.
`--mode bm25`, `--mode vector`, and `--mode hybrid` make comparisons explicit. Vector and
hybrid modes require `build --vectors` and the optional dependencies; they never silently
fall back. Running `build` without `--vectors` produces a BM25-only index and removes old vectors.
Hybrid retrieval merges BM25 and cosine-similarity rankings with equal-weight reciprocal-rank
fusion, then keeps at most three passages per document. Printed scores are fused ranking
scores, not probabilities. The model uses `query: ` / `passage: ` prefixes and truncates inputs
at its 512-token limit; chunking still preserves page boundaries.

Retrieval does **not** return every relevant passage. Each branch keeps up to
`max(4 * k, 20)` candidates per query (32 for the default `k=8`); fusion combines those
lists, then the document cap and final `k` limit apply. Use `--candidates`, `--per-doc-cap`,
and `-k`, or the sidebar Search options, to change these limits. Increasing them can
improve coverage but does not guarantee completeness or relevance.

Optional `--rerank` (the **Rerank candidates** checkbox) applies the local multilingual
[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) cross-encoder
to the top fused candidate pool **before** the document cap and final cutoff. It reads
the query together with each passage to score their relevance; it cannot recover a
passage absent from the candidate pool. It uses the `vectors` dependencies, downloads
additional model weights on first use, and can be slow on CPU. Query/passage pairs are
truncated to 512 tokens. Reranked scores are model scores, not probabilities.

```bash
envsearch search "BOD limit" --mode hybrid --rerank --candidates 40 --per-doc-cap 4 -k 8
envsearch eval run --retrieval-only --mode hybrid --no-expand --rerank
```

Reranking is off by default. Compare saved evaluation reports with it on and off,
especially the `xl` slice, before choosing a default. Each frontend answer/search has
a **How passages were selected** trace showing filtering and candidate/returned counts.

Filters work with `search`, `ask`, and `eval run` in every mode:

| Flag | Meaning |
|---|---|
| `--jurisdiction TH` | Country/regulatory jurisdiction; case insensitive |
| `--lang th` | Passage language, independently of jurisdiction |
| `--doc ID` | Allowed document; repeat to allow several IDs |
| `--filter topic=effluent` | Exact, case-sensitive custom metadata match; repeat with different keys |

All filter categories are ANDed and applied **before** either retriever selects candidates.
Leave `--lang` unset for cross-language retrieval: an English question can retrieve a Thai
passage. Unknown metadata keys or nonmatching values return no passages.

Add optional string metadata to entries in `corpus/manifest.yaml`, then rebuild:

```yaml
metadata:
  topic: effluent
  edition: "2024"  # illustrative; use your document's actual metadata
```

Quote dates and numbers to keep them strings. Metadata is stored once per document in
`docs.json` and applies to all of its chunks. The shipped manifest includes topics `hazardous-waste`, `industrial-waste`, and `effluent`:

```bash
envsearch search "BOD limit" --jurisdiction TH --filter topic=effluent
```

The existing bilingual rewrite remains available (`search --expand`, or enabled by default
for `ask` and `eval run`). Use `--no-expand` with `ask`/`eval run` to try hybrid without that
API call. Dense search returns the closest passages even for unrelated questions; a hit is
not evidence that the corpus answers the question. The answer pipeline still uses its
grounding/refusal prompt and citations.

## Claude Desktop (MCP)

The **Tools** tab provides exact paths and copyable commands for the current machine,
step-by-step client setup, test prompts with expected results, and troubleshooting.
This repo already implements MCP over **stdio**: the client launches a local process.
Streamlit at `localhost:8501` is not an MCP endpoint and does not need to stay running.

Add this to `claude_desktop_config.json`. On macOS the file is in `~/Library/Application Support/Claude/`; on Windows it's in `%APPDATA%\Claude\`. Use absolute paths.

```json
{
  "mcpServers": {
    "envsearch": {
      "command": "/ABSOLUTE/PATH/envsearch/.venv/bin/envsearch-mcp",
      "env": {
        "ENVSEARCH_HOME": "/ABSOLUTE/PATH/envsearch/data"
      }
    }
  }
}
```

Restart Claude Desktop. The server provides four tools:

| tool | what it does |
|---|---|
| `search_regulations(query, k, language, doc_ids, jurisdiction, metadata, mode, rerank, candidate_k, per_doc_cap)` | Filtered BM25/vector/hybrid retrieval with optional reranking; returns passages labelled with title, page, jurisdiction and a `#page=N` URL. Desktop Claude reads them and cites them. |
| `get_passage(chunk_id, context)` | Returns a passage plus its neighbours, for tables or clauses that get cut off. |
| `list_documents()` | Lists what's indexed. |
| `ask_with_citations(question, language, doc_ids, jurisdiction, metadata, mode, expand)` | Runs the filtered retrieval and answer pipeline (API-native citations). Needs an API key. |

The MCP tools default to hybrid when vectors are present, otherwise BM25. In BM25 mode,
the search tool's description tells Desktop Claude to search again in the source language.
For example, use `jurisdiction="TH", language="any", metadata={"topic": "effluent"}`
to search Thai effluent documents in either language (after rebuilding).
Restart the MCP server after rebuilding its cached index.

The first three tools need no server-side answer API key; the MCP client's own model
may have separate charges. `ask_with_citations` uses Anthropic only. Keys entered in
the web app and its OpenAI provider selection are not passed to the MCP process, and
the MCP entrypoint does not load the web app's `.env` file. Sidebar filters also do
not carry over: pass them explicitly in tool arguments. Raw search skips scope routing;
the cited-answer pipeline checks scope. MCP calls are not logged by the web Monitoring tab.
For `ask_with_citations`, add `ANTHROPIC_API_KEY` to the server's `env` object or use
your client's secure environment settings, then restart it. Keep credentials private.

The public demo also supports **Streamable HTTP** at `/mcp`, using `envsearch.mcp_http`.
It exposes three read-only tools (search, passage lookup, document listing), with no
server-side paid answer calls. See [Deploy on Render](deploy/README.md) for the combined
browser/MCP container. Local stdio still exposes all four tools. Anonymous HTTP is
intended only for the curated public library, not private documents.

Client references: [local MCP setup](https://modelcontextprotocol.io/docs/develop/connect-local-servers),
[MCP Inspector](https://github.com/modelcontextprotocol/inspector), and
[remote MCP connections](https://modelcontextprotocol.io/docs/develop/connect-remote-servers).

---

## How it works

```
manifest ─fetch─▶ PDFs ─extract per page─▶ normalise ─chunk─▶ BM25 + optional multilingual vectors

question ─▶ optional bilingual rewrite ─▶ metadata filters ─┬─ BM25 ────┐
                                                          └─ vectors ─┤
                                                                      ▼
                                              RRF fusion, ≤3 hits/doc ─▶ top-k

top-k ─▶ page-linked `search_result` blocks ─▶ Claude ─▶ native citations ─▶ [n] footnotes
```

**Decisions worth defending**

1. **Page-exact citations.** Chunks never cross a page boundary, and each `search_result.source` is `url#page=N`. Clicking a citation opens the PDF at the right page. Each chunk is split into small text blocks because Claude cites whole blocks, so smaller blocks give tighter quotes.
2. **BM25 plus optional multilingual embeddings.** BM25 keeps exact regulatory terms like "55 gallons", "40 CFR 262.15" or "บีโอดี" useful. Multilingual vectors add matching across wording and languages. A local NumPy matrix is sufficient for this small corpus, and rank fusion avoids directly mixing incomparable BM25 and cosine scores. Keep BM25-only and query-expanded runs as baselines; the eval set determines whether hybrid helps.
3. **Thai text is fixed before indexing, not after.** Government gazette PDFs extract "ำ" as "ํ"+"า", sometimes with the tone mark in the middle ("นํ้า"). They also write numbers in Thai digits ("๒๐"). A user types "น้ำ" and "20", so without `textnorm.normalize` Thai recall quietly falls to near zero. Thai is word-segmented with PyThaiNLP `newmm`, plus a small dictionary of transliterated terms (บีโอดี, ซีโอดี…) that the default dictionary breaks into syllables.
4. **Refusal is a first-class outcome.** The system prompt defines an exact not-found sentence. The eval scores unanswerable questions separately, including a jurisdiction trap: "Under Thai law, is an SAA limited to 55 gallons?" For a compliance tool, a confident US answer to a Thai question is worse than no answer.

## Evaluation

`evals/questions.yaml` has 24 items: 11 English→US, 4 Thai→Thai, 5 cross-lingual, and 4 that should be refused. Scoring is done by code, with no LLM judge:

The frontend combines these seed cases with local additions in `data/gold/questions.json`
(under `ENVSEARCH_HOME` if configured). The CLI defaults to the seed file; pass `--questions`
with the downloaded combined set, or the local file to run only local additions.
Previously saved evaluation reports remain snapshots of the questions and labels used.

| metric | meaning |
|---|---|
| `retrieval.hit@k` / `retrieval.mrr` | Did a gold document reach Claude, and how high was it ranked? |
| `retrieval.doc_recall@k` | Fraction of known gold documents represented in the first k passages |
| `retrieval.doc_ndcg@k` | Binary nDCG over distinct documents in first-seen order after the passage cutoff |
| `answer.correct` | Not refused, and every `must_include` keyword group matched (after normalising Thai digits and commas) |
| `answer.grounded` | At least one citation points to a gold document |
| `answer.cite_prec` | Share of citations that point to gold documents |
| `answer.false_refusal` / `refusal.correct` | Over-caution, and correct refusals on unanswerable items |
| `scope.correct` | Scope decision matches `expected_scope` (default `in_scope`); full answer evaluations only |

Results are sliced by tag (`en`, `th`, `xl`, `neg`, `trap`…), so you can see where it fails, such as cross-lingual retrieval versus Thai answers.

Scope decisions are saved in per-question reports and chat retrieval traces. An in-scope
unanswerable question rejected as `out_of_scope` no longer counts as a correct refusal.
Evaluation items can set `expected_scope: out_of_scope` or `expected_scope: clarify` with
`answerable: false` to test those routes. Scope-check timing and input/output tokens are
recorded separately in operational traces; the **Answer tokens** metric covers final
generation only. Neither routing logs nor operational traces store the question text.

Document recall and nDCG ignore `gold_pages`; Hit@k/MRR honor those page constraints.
Duplicate passages from one document do not earn extra document credit. nDCG uses
`sum(relevance / log2(rank + 1))`, normalized by the ideal ordering of up to
`min(k, number of gold documents)` relevant documents. It ranks the unique documents
actually present in the first k passages, not a fresh top-k document search. No-gold and
unanswerable items are excluded from these two averages.

The labels are incomplete relevance judgments: unjudged documents score as nonrelevant,
and every listed gold document counts in the recall denominator even if it is an
alternative acceptable source. With one gold document, document recall is just document
hit rate. These are **known-gold coverage and ranking proxies**, not exhaustive passage
recall. For passage Recall@k, label all relevant chunk IDs; for graded nDCG, assign human
relevance grades (e.g. 0=irrelevant, 1=partial, 2=direct evidence). Keep a held-out set and
compare the same questions, corpus, k, and filters between retrieval configurations.

Business outcomes require additional task instrumentation. The Monitoring tab explains
candidate measures: verified task resolution, time saved versus a manual baseline,
appropriate expert escalation, cost per resolved task, and audited material error rate.
They are not inferred from retrieval scores or currently recorded as business outcomes.
For this regulatory assistant, prioritize correct jurisdiction/thresholds and successful
source verification alongside speed. Report feedback coverage to avoid treating missing
feedback as success. Current token usage excludes rewriting, so it is not a complete cost measure.

Useful runs:

```bash
envsearch eval run --retrieval-only --mode bm25 --no-expand # keyword baseline: no key
envsearch eval run --retrieval-only --mode bm25             # + bilingual expansion (API key)
envsearch eval run --retrieval-only --mode hybrid --no-expand # hybrid on original questions
envsearch eval run --retrieval-only --mode vector --no-expand --only xl # cross-language vectors
envsearch eval run                                # full pipeline
envsearch eval run --only xl,trap                 # just the hard slices
envsearch eval draft --doc th-diw-waste-2566 -n 8 # Claude proposes items -> evals/drafts.yaml; review, then merge
```

> **Seed-set caveat:** the gold facts were written from general knowledge of these regulations, not copied from the downloaded PDFs. Run `envsearch eval verify` first. It checks each keyword actually appears in the gold document and lists any that don't, so fix those before trusting a score.

## Adding documents

Add an entry to `corpus/manifest.yaml` with an `id`, `title`, `lang`, `jurisdiction` and `url`, or a local `path`, then run `fetch && build`. PDF, HTML and TXT all work. If `build` prints `WARN … very little text extracted`, the PDF is scanned and needs OCR first (for example `ocrmypdf -l tha+eng`), which isn't included.

## Known limitations

- Tables in Thai PDFs extract in reading order, and a parameter can end up separated from its limit. `get_passage` with context helps; layout-aware table extraction would fix it properly.
- `th-diw-waste-2566` is downloaded from a provincial industry office mirror. The Royal Gazette copy would be the authoritative source.
- The 2012 EPA reference document and parts of the compendium predate or partly reflect the 2016 Generator Improvements Rule. The answers repeat what the documents say; they don't track what's currently in force. This is a research tool, not legal advice.

## Layout

```
src/envsearch/
  textnorm.py     Thai/EN normalisation + tokenisation
  corpus.py       manifest, fetch, per-page extraction, chunking
  index.py        metadata filters, BM25/hybrid retrieval, RRF, per-doc diversity cap
  vectors.py      local multilingual E5 embeddings, persistence, exact cosine search
  rerank.py       optional local multilingual cross-encoder
  ingest.py       PDF validation, duplicate detection, incremental ingestion
  chat.py         streaming cited answers with follow-up context
  providers.py    provider text calls and OpenAI Responses streaming/citation mapping
  scope.py        topic classification and follow-up resolution before retrieval
  web_app.py      local chat, PDF uploads, filters, and source previews
  web_panels.py   evaluation, gold-label inspection, monitoring, and MCP setup
  eval_ui.py      guided evaluation steps, score explanations, and saved results
  gold.py         validated local gold additions, atomic persistence, seed-set merge
  gold_ui.py      guided answer-key browsing, question form, source review, and export
  mcp_ui.py       local MCP setup, client configuration, verification, hosting guidance
  monitoring.py   local SQLite operational metrics without conversation content
  rewrite.py      bilingual query expansion (Haiku)
  answer.py       search_result blocks -> Claude -> footnoted citations
  evaluate.py     verify / run / draft + metrics and reports
  mcp_server.py   FastMCP stdio server for Claude Desktop
  cli.py
corpus/manifest.yaml   evals/questions.yaml   tests/ (synthetic fixtures, fake Claude client)
```
