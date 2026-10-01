# Free browser + MCP demo on Render

This deploys independently of your computer. One container serves the browser at `/`
and a real Streamable HTTP MCP server at `/mcp`. It runs on Render's **Free** plan.

[Deploy the demo](https://render.com/deploy?repo=https://github.com/chatchawan90/e2e-rag-demo)

1. Sign in to Render using GitHub. Do not add a payment method for a strict zero-bill setup.
2. Open the deployment link above and review the Blueprint. It defines one `envsearch-demo`
   Docker web service with `plan: free`, no paid database or disk, and no answer API keys.
3. Create/deploy the Blueprint. The first build installs dependencies and prepares the
   public embedding model. Wait for the service to become **Live**.
4. Open the service's `https://…onrender.com` URL for the browser demo.
5. Use that same URL plus **`/mcp`** in an MCP client that supports anonymous Streamable HTTP.
   The Tools tab shows the actual URL automatically, using Render's `RENDER_EXTERNAL_URL`.
6. Call `list_documents`, then `search_regulations` with `mode: "hybrid"` and a question.
   Expected library: 11 public regulatory documents and 479 passages.

Do not select a paid compute plan or add a disk. If Render requests payment to proceed,
stop and check that the service is on Free. This repository does not create paid resources.

## What visitors can try

- Browser chat in English or Thai using their own OpenAI or Anthropic API key.
  Provider calls can incur charges on the visitor's account; free hosting does not make
  paid model APIs free. Keys stay in the app's session memory and are not saved to disk.
- Library previews, starter gold examples, and multilingual keyword/vector/hybrid retrieval.
- MCP search, document listing, and passage lookup with no server-side answer key.

The public demo disables uploads, gold edits, evaluation runs, shared activity logs,
and reranking. These remain available in the local app. No local uploads, private gold
additions, evaluation reports, monitoring databases, `.env` files, or keys are in the
deployment snapshot. The HTTP server has no `ask_with_citations` tool.

## Free hosting limits

Render Free sleeps after 15 minutes without traffic. Wake the service by opening its
browser page before connecting MCP if the first connection times out. The workspace
has 750 free instance hours per month, plus bandwidth/build limits. With no payment
method, exhausted allowances can suspend the service instead of producing a bill.
This is a demonstration, not an always-warm or production-availability promise.
See [Render Free](https://render.com/docs/free) and [billing FAQ](https://render.com/docs/faq).

The image includes its reference index, PDFs and model. There is no paid persistent
disk; ephemeral filesystem changes are not relied on. Both interfaces share a compact
int8 ONNX version of `intfloat/multilingual-e5-small` to reduce memory use. Query
embeddings can differ slightly from the full-precision local model. The model revision
is pinned in `onnx_embeddings.py`. The hosted backend does not import PyTorch.
Hosted queries are truncated to 256 model tokens to bound memory. Source PDFs open
on the original publisher's site, and gold examples use a compact question picker.

## Local deployment verification

```bash
docker build -t envsearch-demo .
docker run --rm --memory=512m --memory-swap=512m -p 10000:10000 envsearch-demo
# In a second terminal, using the repo's development environment:
.venv/bin/python deploy/check_demo.py http://localhost:10000
```

The check opens the browser app's actual session protocol and calls the MCP server over
HTTP, including English/Thai hybrid retrieval and a long query. It makes no paid API calls.
After deployment, run the same check with the Render HTTPS URL.

To refresh the public snapshot, export to a new directory and review it before replacing
`deploy/demo-data`. The exporter includes only indexed HTTP(S) manifest documents whose
URLs match and whose records have no local upload path:

```bash
.venv/bin/python -m envsearch.demo --output .demo/reviewed-update
```

To run only the HTTP service, install `.[web,hosted]`, set `ENVSEARCH_HOME` to a public
snapshot and `ENVSEARCH_EMBEDDING_BACKEND=onnx`, then run:

```bash
python -m envsearch.mcp_http --host 0.0.0.0 --port 8000 --public-url https://your-host.example
```

The public origin allowlist is explicit. The server rejects unexpected Host/Origin
headers and limits request bodies to 64 KiB. Expensive HTTP retrieval runs one query
at a time, off the HTTP event loop. This public demo is not a private multi-user service;
add authentication and per-user document access before hosting private documents.
