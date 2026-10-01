"""Local Streamlit interface. Launch with `envsearch serve`."""
from __future__ import annotations

import os
import time
from collections import Counter
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import streamlit as st
from dotenv import load_dotenv
from filelock import FileLock

from envsearch.answer import Citation
from envsearch.chat import stream_answer, answer as chat_answer
from envsearch.config import REPO_ROOT, get_settings
from envsearch.corpus import raw_path
from envsearch.index import Index
from envsearch.rewrite import expand_query
from envsearch.scope import route_question
from envsearch.web_panels import evaluation_panel, gold_panel, monitoring_panel, mcp_panel, monitor

st.set_page_config(page_title="Envsearch · Document chat", page_icon="📑", layout="wide")
public_demo = os.environ.get("ENVSEARCH_PUBLIC_DEMO") == "1"
if not public_demo:
    load_dotenv(REPO_ROOT / ".env")
settings = get_settings()
st.markdown("""<style>
    .block-container {max-width: 1080px; padding-top: 2.1rem; padding-bottom: 3rem;}
    h1, h2, h3 {font-family: 'Avenir Next', Tahoma, sans-serif; letter-spacing: -.025em;}
    [data-testid="stSidebar"] {border-right: 1px solid #dfe6ef;}
    [data-testid="stChatMessage"] {border-radius: 12px; padding: 1.2rem;}
    [data-testid="stChatMessage"] p {line-height: 1.7;}
    [data-testid="stCaptionContainer"] {color: #52677f;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(max_entries=2)
def cached_index(directory: str, revision: int) -> Index:
    return Index.load(Path(directory))


def current_index() -> Index:
    settings.home.mkdir(parents=True, exist_ok=True)
    with FileLock(str(settings.index_dir) + ".lock"):
        Index._recover(settings.index_dir)
        path = settings.index_dir / "docs.json"
        if not path.exists():
            return Index([], {})
        # The save transaction replaces the directory; the lock prevents mixed reads.
        revision = path.stat().st_mtime_ns
    return cached_index(str(settings.index_dir), revision)


@st.dialog("Source document", width="large")
def show_pdf(doc_id: str, page: int):
    doc = idx.docs[doc_id]
    path = raw_path(doc, settings.raw_dir)
    st.write(doc.title)
    st.caption(f"Page {page}")
    if public_demo:
        st.link_button("Open original source PDF", f"{doc.url}#page={page}")
        return
    import pymupdf
    if path.exists() and path.suffix.lower() == ".pdf":
        # Show exactly the cited page, while keeping the full original available.
        with pymupdf.open(path) as pdf, pymupdf.open() as excerpt:
            excerpt.insert_pdf(pdf, from_page=page - 1, to_page=page - 1)
            st.pdf(excerpt.tobytes(), height=560)
        st.download_button("Download full PDF", path.read_bytes(), file_name=path.name, mime="application/pdf")
    elif urlparse(doc.url).scheme in {"http", "https"}:
        st.link_button("Open original source", f"{doc.url}#page={page}")
    else:
        st.info("The source PDF is no longer available at its saved location.")


def sources(citations, message_id):
    for cite in citations:
        with st.expander(f"[{cite.n}] {cite.title} · page {cite.page}"):
            st.write(cite.cited_text)
            if cite.doc_id in idx.docs:
                if st.button("View cited page", key=f"source-{message_id}-{cite.n}"):
                    show_pdf(cite.doc_id, cite.page)
            if urlparse(cite.url).scheme in {"http", "https"}:
                st.link_button("Open original source", cite.url)


try:
    idx = current_index()
except (ValueError, OSError) as exc:
    st.error(f"The saved index could not be loaded: {exc}")
    st.stop()

st.session_state.setdefault("messages", [])
with st.sidebar:
    st.title("Envsearch")
    st.caption("Environmental regulations, with sources.")
    st.divider()
    st.write(f"**{len({c.doc_id for c in idx.chunks})} documents** · {len(idx.chunks)} passages")
    st.caption("Hybrid search is ready" if idx.vectors else "Keyword search is ready")
    st.subheader("Search scope")
    jurisdictions = sorted({d.jurisdiction for d in idx.docs.values() if d.jurisdiction})
    jurisdiction = st.selectbox("Jurisdiction", ["Any", *jurisdictions])
    topics = sorted({d.metadata["topic"] for d in idx.docs.values() if d.metadata.get("topic")})
    topic = st.selectbox("Topic", ["Any", *topics])
    selected_docs = st.multiselect("Documents", list(idx.docs), format_func=lambda did: idx.docs[did].title,
                                   placeholder="All documents")
    with st.expander("Search options"):
        language = st.selectbox("Passage language", ["Any", "English", "Thai"])
        mode = st.selectbox("Retrieval", ["auto", "hybrid", "vector", "bm25"] if idx.vectors else ["auto", "bm25"])
        expand = st.checkbox("Expand queries in English and Thai", value=False)
        k = st.slider("Passages per answer", 1, 15, settings.top_k)
        candidate_k = st.number_input("Candidate pool per query / search branch", min_value=k, max_value=200,
                                      value=max(4 * k, 20), step=1)
        per_doc_cap = st.number_input("Maximum passages from one document", min_value=1, max_value=30, value=3)
        rerank = st.checkbox("Rerank candidates", value=False, disabled=public_demo,
                             help="Uses a local multilingual cross-encoder. First use downloads an additional model; CPU runs can be slow.")
        st.caption("Retrieval has a cutoff. Reranking scores the top fused candidate pool, then applies the document cap and final passage limit.")
    st.divider()
    provider_label = st.selectbox("Answer provider", ["Anthropic", "OpenAI"])
    provider = provider_label.lower()
    configured_key = "" if public_demo else os.environ.get(f"{provider.upper()}_API_KEY", "")
    # Streamlit removes hidden widget state; preserve the key separately when switching providers.
    key_widget, saved_key = f"session_{provider}_api_key", f"saved_{provider}_api_key"
    if key_widget not in st.session_state:
        st.session_state[key_widget] = st.session_state.get(saved_key, "")
    session_key = st.text_input(f"{provider_label} API key", type="password", key=key_widget,
                                help="Sent to this app's server over your connection and kept in memory for your session, not saved to disk. Used to call your selected AI provider. Only enter a key if you trust this demo host.")
    st.session_state[saved_key] = session_key
    api_key = session_key.strip() or configured_key
    if configured_key:
        st.caption(f"{provider_label} has an environment key. Enter a key above to override it for this session.")
    else:
        st.caption("Add your own key for the scope check and chat. Your provider may charge for these calls. Library previews and MCP search need no key." if public_demo else "Add a key for the scope check in both conversation modes. PDF uploads and library previews work without one.")
    with st.expander("Answer settings"):
        default_model = settings.answer_model if provider == "anthropic" else os.environ.get("ENVSEARCH_OPENAI_MODEL", "gpt-4.1-mini")
        model_widget, saved_model = f"model_{provider}", f"saved_model_{provider}"
        if model_widget not in st.session_state:
            st.session_state[model_widget] = st.session_state.get(saved_model, default_model)
        answer_model = st.text_input("Answer model", key=model_widget)
        st.session_state[saved_model] = answer_model
        rewrite_model = settings.rewrite_model if provider == "anthropic" else answer_model
    st.caption(f"Chat sends questions, recent conversation, and retrieved passages to {provider_label}.")
    if provider == "openai":
        st.caption("OpenAI citations link model-selected passage numbers to your PDFs; they are not independently verified claims.")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

filters = dict(mode=mode, jurisdiction=None if jurisdiction == "Any" else jurisdiction,
               metadata=None if topic == "Any" else {"topic": topic},
               lang={"Any": None, "English": "en", "Thai": "th"}[language],
               doc_ids=set(selected_docs) if selected_docs else None,
               candidate_k=candidate_k, per_doc_cap=per_doc_cap, rerank=rerank)


def client_factory():
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key, timeout=90, max_retries=1)
    return anthropic.Anthropic(api_key=api_key, timeout=90, max_retries=1)

st.title("Ask your documents")
st.caption("Ask in English or Thai. Follow the citations back to the page.")
st.caption("Both conversation modes are limited to environmental regulations and compliance. Unrelated questions are declined before document search.")
if public_demo:
    st.info("Public demo · Reference documents only. Browser chat uses your own API key; free MCP search needs no key. Uploads, dataset edits, evaluation runs and activity logs are available in the local app.")
chat_tab, upload_tab, library_tab, eval_tab, gold_tab, monitor_tab, tools_tab = st.tabs(
    ["Conversation", "Add PDFs", "Library", "Evaluation", "Gold dataset", "Monitoring", "Tools"])

with chat_tab:
    purpose = st.radio("Response", ["Chat with sources", "Find passages"],
                        index=0, horizontal=True)
    st.caption("Both modes check scope first. Find passages shows source excerpts without generating an answer.")
    if not api_key:
        st.info(f"Enter your {provider_label} API key to run the scope check before chatting or finding passages.")
    if not st.session_state.messages:
        st.subheader("Start with a question")
        st.write("What is the BOD limit for factory wastewater in Thailand?")
        st.write("พื้นที่สะสมของเสียอันตรายเก็บได้สูงสุดเท่าไร")
        st.caption("Choose a jurisdiction or topic to narrow the sources." if public_demo else "Choose a jurisdiction or topic to narrow the sources. Add your own documents in Add PDFs.")
    for number, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources(message.get("citations", []), number)
            if message.get("trace"):
                label = "Scope check" if message["trace"].get("scope_decision") in {"out_of_scope", "clarify"} else "How passages were selected"
                with st.expander(label):
                    st.json(message["trace"])
    prompt = st.chat_input("Ask a question about your documents…", max_chars=6000,
                           disabled=not idx.chunks or not api_key)
    if not idx.chunks:
        st.info("Add a PDF to start searching and chatting.")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        with st.chat_message("assistant"):
            status, output = st.empty(), st.empty()
            started, trace, usage, found = time.perf_counter(), {}, {}, False
            operation = "search" if purpose == "Find passages" else "chat"
            try:
                if purpose == "Find passages":
                    status.info("Checking the question's scope…")
                    with client_factory() as client:
                        route = route_question(prompt, client, model=rewrite_model, provider=provider,
                                               history=history, trace=trace)
                        if route.decision == "in_scope":
                            status.info("Searching your documents…")
                            queries = expand_query(route.query, client if expand else None, rewrite_model, provider=provider)
                            hits = idx.search(queries, k=k, trace=trace, **filters)
                            text = f"Retrieved {len(hits)} candidate passages. Check the sources for relevance." if hits else "No passages match this search and its filters."
                            citations = [Citation(h.rank, h.chunk.id, h.chunk.doc_id, h.chunk.page,
                                                  idx.docs[h.chunk.doc_id].title, idx.cite_url(h.chunk), h.chunk.text) for h in hits]
                        else:
                            text, citations = route.reply(prompt), []
                else:
                    text, citations = "", []
                    with client_factory() as client:
                        for event in stream_answer(prompt, idx, client, model=answer_model,
                                                   rewrite_model=rewrite_model, history=history, provider=provider,
                                                   expand=expand, k=k, trace=trace, **filters):
                            if event["type"] == "status":
                                status.info(event["text"])
                            elif event["type"] == "text":
                                text += event["text"]
                                output.markdown(text + " ▌")
                            else:
                                text, citations = event["answer"].text, event["answer"].citations
                                usage, found = event["answer"].usage, event["answer"].found
                status.empty()
                monitor(settings.home, operation, "ok", time.perf_counter() - started,
                        provider=provider, model=answer_model,
                        mode=mode, k=k, candidate_k=candidate_k, per_doc_cap=per_doc_cap, rerank=rerank,
                        returned=trace.get("returned", 0), citations=len(citations), found=found, usage=usage, trace=trace)
                st.session_state.messages.extend([
                    {"role": "user", "content": prompt, "chat": purpose == "Chat with sources"},
                    {"role": "assistant", "content": text, "citations": citations, "trace": trace,
                     "chat": purpose == "Chat with sources"}])
                st.rerun()
            except Exception as exc:
                monitor(settings.home, operation, "error", time.perf_counter() - started,
                        provider=provider, mode=mode, rerank=rerank, error_type=type(exc).__name__, trace=trace)
                status.error("The request failed. Check API access, the model, or local model downloads. "
                             f"Error type: {type(exc).__name__}")

with upload_tab:
    if public_demo:
        st.info("Uploads are disabled in this shared demo. Run the app locally to add your own PDFs.")
    else:
        st.subheader("Add a PDF to your library")
        st.write("Upload a text-based PDF. Its passages become searchable as soon as indexing finishes.")
        if notice := st.session_state.pop("upload_notice", None):
            st.success(notice)
        with st.form("upload_pdf"):
            uploaded = st.file_uploader("PDF document", type=["pdf"])
            title = st.text_input("Document title", placeholder="Use the filename if left blank")
            left, right = st.columns(2)
            upload_jurisdiction = left.text_input("Jurisdiction", placeholder="TH, US, or another jurisdiction")
            upload_topic = right.text_input("Topic", placeholder="e.g. effluent")
            extra = st.text_area("Additional metadata (optional)", placeholder="edition=2024\ndepartment=operations",
                                 help="One key=value per line. Values are stored as text.")
            use_vectors = st.checkbox("Enable multilingual vector search", value=True, disabled=idx.vectors is not None,
                                       help="If this index already has vectors, uploads always use the same model.")
            submitted = st.form_submit_button("Add to library", type="primary")
        st.caption("Up to 50 MB and 1,000 pages. Scanned PDFs need OCR first. Duplicate files are detected automatically.")
        if submitted:
            if uploaded is None:
                st.error("Choose a PDF to upload.")
            else:
                started = time.perf_counter()
                try:
                    metadata = {"topic": upload_topic.strip()} if upload_topic.strip() else {}
                    for line in extra.splitlines():
                        if not line.strip():
                            continue
                        key, sep, value = line.partition("=")
                        if not sep or not key.strip() or key.strip() in metadata:
                            raise ValueError("Use one unique key=value per metadata line; topic already has its own field.")
                        metadata[key.strip()] = value.strip()
                    with st.status("Adding your PDF…", expanded=True) as progress:
                        from envsearch.ingest import ingest_pdf
                        result = ingest_pdf(uploaded.getvalue(), uploaded.name, settings, title=title,
                                            jurisdiction=upload_jurisdiction, metadata=metadata,
                                            with_vectors=use_vectors, progress=progress.write)
                        progress.update(label="PDF is ready", state="complete")
                    monitor(settings.home, "ingestion", "ok", time.perf_counter() - started,
                            doc_id=result.doc_id, chunks=result.chunks, pages=result.pages)
                    st.session_state.upload_notice = (f"{result.title} is already in the library." if result.duplicate else
                                                      f"Added {result.title}: {result.pages} pages, {result.chunks} passages. Ready to search.")
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    monitor(settings.home, "ingestion", "error", time.perf_counter() - started, error_type=type(exc).__name__)
                    st.error(str(exc))

with library_tab:
    st.subheader("Your document library")
    counts = Counter(c.doc_id for c in idx.chunks)
    for doc in idx.docs.values():
        with st.expander(doc.title):
            st.caption(f"{doc.jurisdiction or 'Jurisdiction not set'} · {doc.lang} · {counts[doc.id]} passages")
            if doc.metadata:
                st.json(doc.metadata, expanded=True)
            if counts[doc.id] and st.button("Preview PDF", key=f"preview-{doc.id}"):
                show_pdf(doc.id, min(c.page for c in idx.chunks if c.doc_id == doc.id))
    st.caption("Conversation history stays in this browser session. This demo library contains public reference documents." if public_demo else "Conversation history stays in this browser session. Uploaded documents stay in your local library.")

with eval_tab:
    if public_demo:
        st.info("Run evaluations in your own local app. The Gold dataset tab shows example test questions; this demo does not publish visitors' answers or saved reports.")
    else:
        evaluation_panel(idx, settings, filters=filters, k=k, model=answer_model, rewrite_model=rewrite_model,
                         provider=provider, has_key=bool(api_key), expand=expand,
                         client_factory=client_factory, answer_fn=partial(chat_answer, provider=provider))
with gold_tab:
    gold_panel(idx, show_pdf)
with monitor_tab:
    if public_demo:
        st.info("Visitor activity logs are not collected or displayed in this shared demo.")
    else:
        monitoring_panel(settings)
with tools_tab:
    mcp_panel(settings)
