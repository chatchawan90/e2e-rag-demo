"""A guided evaluation workflow with plain-language results and optional technical detail."""
import json
import time
from datetime import datetime

import streamlit as st

from . import evaluate as ev
from .gold import load_dataset

GROUPS = {"en": "English questions", "th": "Thai questions", "xl": "Across languages",
          "neg": "Questions the library cannot answer", "trap": "Tricky questions",
          "threshold": "Numbers and limits", "us": "US rules", "th-reg": "Thai rules"}
METHODS = {"auto": "Automatic", "bm25": "Keyword search", "vector": "Meaning-based search", "hybrid": "Keywords + meaning"}
METRICS = {
    "retrieval.hit@k": ("Expected source found", "Of questions with labeled sources, how often did at least one expected document (and page, if specified) appear in the returned passages? Technical name: Hit@k. k is the maximum number of returned passages."),
    "retrieval.doc_recall@k": ("Expected sources covered", "What fraction of the known expected documents appeared in the returned passages? Finding 2 of 4 gives 50%. Duplicate documents count once; page labels are ignored. This measures known labels, not every relevant passage. Technical name: document Recall@k."),
    "retrieval.mrr": ("First useful result", "Rewards finding an expected source early. First place earns 1, second earns 0.5, and no match earns 0. Averaged across answerable questions. Higher is better. Technical name: MRR (mean reciprocal rank)."),
    "retrieval.doc_ndcg@k": ("Order of useful sources", "Rewards relevant documents appearing near the top, compared with an ideal order. Range 0–1; higher is better. Uses relevant/not-relevant labels, deduplicates documents after the passage cutoff, and ignores page labels. Unlabeled documents receive no credit. Technical name: binary document nDCG@k."),
    "answer.correct": ("Answer checks passed", "Among answerable questions, how often did an answer have a citation, match every required keyword group, and pass the expected scope check? These automatic checks do not verify every claim or prove factual correctness."),
    "answer.grounded": ("Answer cites an expected source", "Among answerable questions, how often did at least one citation point to a labeled source document? This checks the document identity, not whether its text proves the claim. Technical name: answer.grounded."),
    "answer.cite_prec": ("Citations pointing to expected sources", "Average fraction of citations pointing to labeled source documents, among answerable questions that received citations. Does not check individual claims or page labels. Technical name: citation precision."),
    "answer.false_refusal": ("Answerable questions left unanswered", "Fraction of answerable questions for which the system did not produce a supported, cited answer. Lower is better. Missing citations also count here. Technical name: false-refusal rate."),
    "refusal.correct": ("Correct decisions not to answer", "Among questions expected to receive no answer, how often did the system avoid answering and make the expected scope decision? Clarification cases are excluded."),
    "scope.correct": ("Scope decisions matched", "How often did the system choose the expected action: search environmental documents, decline an unrelated request, or ask for clarification? Technical name: scope accuracy."),
}


def _percent(value):
    return "Not checked" if value is None else f"{value:.0%}"


def _status(row):
    if row.get("scope_correct") is False:
        return "Needs attention: wrong scope decision"
    if row.get("hit") is False:
        return "Needs attention: expected source missing"
    if row.get("correct") is False:
        return "Needs attention: answer checks failed"
    if row.get("correct") is True:
        return "Automatic checks passed"
    if row.get("hit") is True:
        return "Source found; answer not checked"
    return "Not scored in a sources-only test"


def _report_name(path):
    try:
        parts = path.stem.split("-")
        stamp = datetime.strptime("-".join(parts[1:3]), "%Y%m%d-%H%M%S")
        return stamp.strftime("%d %b %Y, %H:%M:%S") + (f" · {parts[3][:4]}" if len(parts) > 3 else "")
    except (ValueError, IndexError):
        return path.stem


def _results(settings, index):
    def title(doc_id):
        return index.docs[doc_id].title if doc_id in index.docs else doc_id
    with st.container(border=True, key="eval_results"):
        st.subheader("Results", help="Saved snapshots of previous test runs. Changing settings above does not change an existing result; run a new test to compare.")
        paths = sorted((settings.home / "evals").glob("eval-*.json"), reverse=True)
        if not paths:
            st.info("Your results will appear here after you run a test. Start with ‘Find the right sources’ above.")
            return
        if st.session_state.get("eval_report") not in paths:
            st.session_state["eval_report"] = paths[0]
        report = st.selectbox("Choose a saved test", paths, key="eval_report", format_func=_report_name,
                              help="Each entry is a saved run. The latest run is selected after you run a new test.")
        try:
            result = json.loads(report.read_text(encoding="utf-8"))
            rows, config = result["rows"], result["config"]
            overall = result["summary"]["overall"]
        except (ValueError, OSError, KeyError, TypeError):
            st.error("This saved result could not be read. Select a different test or run a new one.")
            return
        source_only = config.get("retrieval_only", True)
        st.write(f"**{'Sources only' if source_only else 'Sources and answers'}** · {len(rows)} questions · "
                 f"{METHODS.get(config.get('mode'), config.get('mode', 'Saved search settings'))} · up to {config.get('k', '—')} passages per question")
        st.caption("You are viewing saved results. The setup above applies only to your next test.")
        if result.get("gold_validation"):
            st.warning(f"Review the answer key: {len(result['gold_validation'])} question(s) have missing source text or unmatched keywords. Scores may be misleading until these labels are checked.")
            with st.expander("Which expected answers need review?"):
                st.dataframe([{"Question ID": p["id"], "What to check": p["problem"]} for p in result["gold_validation"]], hide_index=True)
        cols = st.columns(3)
        for col, metric in zip(cols, ("retrieval.hit@k", "retrieval.doc_recall@k", "answer.correct")):
            label, help_text = METRICS[metric]
            col.metric(label, _percent(overall.get(metric)), help=help_text)
        scored = [r for r in rows if "hit" in r]
        if scored:
            st.write(f"An expected source was found for **{sum(bool(r['hit']) for r in scored)} of {len(scored)}** questions with source labels.")
        if source_only:
            st.info("Answers and scope decisions were not checked in this run. Choose ‘Check sources and answers’ to test those too.")
        st.caption("These checks compare against your answer key. A high score does not prove every answer is factually correct.")
        with st.expander("More scores and what they mean"):
            st.caption("Hover or focus the ? beside a score for its definition. ‘Not checked’ means the score is unavailable for this run.")
            for key in ("retrieval.mrr", "retrieval.doc_ndcg@k", "scope.correct", "refusal.correct",
                        "answer.grounded", "answer.cite_prec", "answer.false_refusal"):
                label, help_text = METRICS[key]
                value = overall.get(key)
                display = ("Not checked" if value is None else f"{value:.3f}") if key.startswith("retrieval.") else _percent(value)
                st.metric(label, display, help=help_text)
            st.markdown("**Quick glossary**")
            st.write("**Gold dataset** = the questions and expected results used as an answer key. **Tags** = question groups. "
                     "**Passage** = an excerpt from a document. **k** = the maximum passages returned. "
                     "**Recall** = coverage of labeled sources. **MRR / nDCG** = how well useful results are ordered. "
                     "**Scope** = whether the request belongs to this assistant's subject area.")
        st.subheader("Review individual questions", help="Look at the expected sources and actual results to understand a failed check. A passed keyword check still needs human review for factual correctness.")
        failures = [r for r in rows if _status(r).startswith("Needs attention")]
        st.write(f"**{len(failures)} question(s) need attention** based on the checks performed.")
        only_failures = st.checkbox("Show only questions needing attention", key="eval_failures",
                                    help="Show missing sources, incorrect scope decisions, and failed automatic answer checks.")
        shown = failures if only_failures else rows
        if only_failures and not shown:
            st.info("No failed checks in this saved run. Checks that were not run are still unverified.")
        if shown:
            st.dataframe([{"Question": r.get("question", r["id"]), "Result": _status(r), "Time (seconds)": r.get("seconds")}
                          for r in shown], hide_index=True, width="stretch")
            lookup = {r["id"]: r for r in shown}
            selected = st.selectbox("Open a question", list(lookup), key=f"eval_detail_{report.stem}",
                                   format_func=lambda rid: lookup[rid].get("question", rid),
                                   help="Open the question to compare what was expected with what the system returned.")
            row = lookup[selected]
            with st.container(border=True):
                st.write(f"**{row.get('question', row['id'])}**")
                st.write(_status(row))
                expected, actual = st.columns(2)
                with expected:
                    st.markdown("**Expected result**")
                    scope = row.get("expected_scope", "in_scope")
                    behavior = {"out_of_scope": "Decline without searching.", "clarify": "Ask for clarification."}.get(scope)
                    st.write(behavior or ("Answer from the documents." if row.get("answerable") else "Search, then explain that the documents do not support an answer."))
                    if row.get("gold_docs"):
                        st.write("Expected source documents:")
                        for did in row["gold_docs"]:
                            st.write(" • " + title(did))
                    if row.get("gold_pages"):
                        st.write("Source pages: " + ", ".join(map(str, row["gold_pages"])))
                    if row.get("must_include"):
                        st.write("Required answer terms (every line must match):")
                        for group in row["must_include"]:
                            st.write(" • " + " or ".join(group))
                with actual:
                    st.markdown("**Actual result**")
                    st.write(row.get("answer") or "No answer was generated in this test.")
                    retrieved = row.get("retrieved_chunks", [])
                    if retrieved:
                        st.dataframe([{"Position": h["rank"], "Document": title(h["doc_id"]), "Page": h["page"]} for h in retrieved], hide_index=True)
                    elif row.get("retrieved"):
                        st.write(row["retrieved"])
                    else:
                        st.caption("No passages were retrieved.")
        with st.expander("Advanced: settings, grouped scores, and raw data"):
            st.caption("For debugging and comparing runs. These are the settings saved with this test.")
            st.json(config, expanded=False)
            st.dataframe([{"Question group": name.replace("tag:", ""), "Questions": metrics.get("n"),
                           **{METRICS[key][0]: metrics.get(key) for key in METRICS}}
                          for name, metrics in result["summary"].items()], hide_index=True)
            if shown:
                st.json(row, expanded=False)
        st.download_button("Download this test report", report.read_bytes(), file_name=report.name, mime="application/json",
                           help="Download the complete saved results as JSON, including settings and per-question scores.")


def evaluation_panel(index, settings, *, filters, k, model, rewrite_model, provider, has_key,
                     client_factory, monitor_fn, answer_fn=None, expand=False):
    st.subheader("Check how well your assistant works")
    st.write("Run a set of questions with known expected results, then see what worked and what needs attention.")
    st.caption("Start with a sources-only test. Use the ? help beside each setting and score, or open the glossary in Results.")
    st.markdown("""<style>
      .st-key-eval_questions {border-top:4px solid #2459A6; background:#F5F8FD;}
      .st-key-eval_checks {border-top:4px solid #52677F; background:#F8FAFC;}
      .st-key-eval_run {border-top:4px solid #2459A6;}
      .st-key-eval_results {border-top:4px solid #24715A;}
    </style>""", unsafe_allow_html=True)
    try:
        items = load_dataset(settings.home)
    except (ValueError, OSError) as exc:
        st.error(f"The question set could not be loaded: {exc}")
        return
    with st.container(border=True, key="eval_questions"):
        st.subheader("1. Choose questions", help="These come from the Gold dataset tab: test questions with expected sources, keywords, or refusal behavior. The answers are not shown to the model.")
        st.write(f"{len(items)} saved questions are available. Leave the selections empty to test them all.")
        tags = st.multiselect("Question groups (optional)", sorted({t for i in items for t in i.tags}), key="eval_tags",
                              format_func=lambda tag: f"{GROUPS.get(tag, tag)} ({tag})" if tag in GROUPS else tag,
                              help="Groups are called tags in the dataset. For example, xl tests an English question against Thai sources (or the reverse). Select multiple groups to include questions matching any group. Tags do not filter documents.")
        selected = [i for i in items if not tags or set(tags).intersection(i.tags)]
        lookup = {i.id: i for i in selected}
        item_ids = st.multiselect("Choose individual questions (optional)", list(lookup), key="eval_ids",
                                  format_func=lambda rid: f"{lookup[rid].question} [{rid}]",
                                  help="Narrow the selected groups to particular questions. Leave empty to include every question in those groups.")
        if item_ids:
            selected = [i for i in selected if i.id in item_ids]
        st.caption(f"Selected: {len(selected)} questions. Add or review expected answers in the Gold dataset tab.")
    with st.container(border=True, key="eval_checks"):
        st.subheader("2. Choose what to check")
        test_type = st.radio("What would you like to check?", ["Find the right sources", "Check sources and answers"],
                             key="eval_type", help="Sources-only checks whether search returns expected documents; it skips scope routing and answer generation. Sources and answers also checks scope decisions, generates answers, and checks their keywords and citations. It uses API credits.")
        full = test_type == "Check sources and answers"
        if full:
            st.write(f"Tests the full question-to-answer flow using {provider.title()}. This uses your API key and credits.")
        else:
            st.write("Finds source passages without writing answers. It does not test refusals or scope decisions.")
            if not expand:
                st.caption("Runs locally without API credits. First use of a local search/reranking model may need a download.")
        options = dict(filters)
        with st.expander("Advanced search settings"):
            scoped = st.checkbox("Limit the test to the documents selected in the sidebar", value=False,
                                  help="Off means search the whole library, so a chat filter cannot accidentally hide an expected source. On applies the sidebar's jurisdiction, topic, document, and language filters.")
            st.write(f"**Search method:** {METHODS.get(options['mode'], options['mode'])} · **Maximum returned passages:** {k}")
            st.caption(f"Candidate pool: {options['candidate_k']} per query/search method. At most {options['per_doc_cap']} passages per document. Reranking: {'on' if options['rerank'] else 'off'}.")
            st.write("**Candidate pool** means passages considered before choosing the final results. **Reranking** means scoring those passages again to improve their order. Change these in the sidebar's Search options.")
        if not scoped:
            options.update(lang=None, jurisdiction=None, metadata=None, doc_ids=None)
        if expand:
            st.warning("Bilingual query expansion is enabled in the sidebar. It creates extra search phrases using your provider, so even a sources-only test uses API credits.")
    with st.container(border=True, key="eval_run"):
        st.subheader("3. Run test")
        st.write(f"**{len(selected)} questions** · **{'Sources and answers' if full else 'Sources only'}** · **{'Filtered documents' if scoped else 'Whole library'}**")
        needs_key = full or expand
        if needs_key and not has_key:
            st.info(f"Add your {provider.title()} API key in the sidebar to run this test. For a local test, choose sources only and turn off query expansion.")
        if not selected:
            st.info("Select at least one question, or add one in Gold dataset.")
        if not index.chunks:
            st.info("Add a PDF to your library before running this test.")
        disabled = not selected or not index.chunks or (needs_key and not has_key)
        if st.button("Run test", type="primary", disabled=disabled, help="Runs the selected questions with these settings and saves a new report. Existing reports are kept."):
            start, progress = time.perf_counter(), st.empty()
            progress.info("Starting your test…")
            completed = 0
            def report_progress(_):
                nonlocal completed
                completed += 1
                progress.info(f"Checked {completed} of {len(selected)} questions…")
            validation = ev.verify(selected, index)
            if validation:
                st.warning(f"{len(validation)} expected-answer checks need review. Details will appear with the results.")
            try:
                def run_with(client):
                    return ev.run(selected, index, client, model=model, rewrite_model=rewrite_model,
                                  k=k, retrieval_only=not full, expand=expand, log=report_progress,
                                  provider=provider, answer_fn=answer_fn, **options)
                if needs_key:
                    with client_factory() as client:
                        result = run_with(client)
                else:
                    result = run_with(None)
                result["gold_validation"] = validation
                path = ev.write_report(result, settings.home / "evals")
                st.session_state["eval_report"] = path.with_suffix(".json")
                monitor_fn(settings.home, "evaluation", "ok", time.perf_counter() - start,
                           provider=provider, model=model, eval_items=len(selected), mode=options["mode"],
                           rerank=options["rerank"], usage={key: sum(r.get("usage", {}).get(key, 0) for r in result["rows"])
                                                          for key in ("input_tokens", "output_tokens")})
                progress.empty()
                st.success("Test finished and saved. Review the results below.")
            except Exception as exc:
                monitor_fn(settings.home, "evaluation", "error", time.perf_counter() - start,
                           provider=provider, error_type=type(exc).__name__)
                st.error("The test could not finish. Check the selected model, API key/credits, and local model downloads, then try again.")
                with st.expander("Technical error details"):
                    st.code(type(exc).__name__)
    _results(settings, index)
