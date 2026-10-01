"""Reviewed, user-authored additions to the gold set."""
import json
import uuid
from dataclasses import asdict

import streamlit as st

from .evaluate import Item, verify
from .gold import load_dataset, save_item

OUTCOMES = {
    "Answer from documents": (True, "in_scope"),
    "On topic, but the documents cannot answer": (False, "in_scope"),
    "Off topic — decline without searching": (False, "out_of_scope"),
    "Unclear question — ask for clarification": (False, "clarify"),
}
FIELDS = ("question", "sources", "pages", "keywords", "tags", "notes")


def add_gold_form(index, home):
    if st.session_state.pop("gold_clear_form", False):
        for field in FIELDS:
            st.session_state.pop("gold_new_" + field, None)
    if notice := st.session_state.pop("gold_saved_notice", None):
        st.success(notice)
    with st.container(border=True, key="gold_add"):
        st.subheader("Add a test question", help="A gold question is a test question with a human-reviewed expected result. It becomes available in Evaluation as soon as you save it.")
        st.write("Use a real question people ask. Decide what a good response should do, then check the original documents before saving.")
        latest = next((m["content"] for m in reversed(st.session_state.get("messages", [])) if m["role"] == "user"), "")
        if st.button("Use latest conversation question", disabled=not latest,
                     help="Copies only your most recent question. You still choose the expected sources and answer checks yourself."):
            st.session_state["gold_new_question"] = latest
        st.caption("Copying uses only the question. Rewrite follow-ups as standalone questions before saving.")
        outcome = st.selectbox("What should the assistant do?", list(OUTCOMES), key="gold_new_outcome",
                               help="The expected outcome is the behavior you want to test. On topic means a question about the environmental documents this assistant covers. Off-topic and unclear questions should be handled before searching.")
        answerable, scope = OUTCOMES[outcome]
        if not answerable:
            st.info("This example tests when the assistant should avoid giving an answer. No source documents or answer keywords are needed. Choose ‘Check sources and answers’ in Evaluation to test this behavior.")
        with st.form("new_gold_question"):
            question = st.text_area("Question to test", key="gold_new_question", max_chars=6000,
                                    help="Write a complete question that makes sense on its own. Evaluation does not include earlier conversation messages.")
            docs, pages_text, keywords = [], "", ""
            if answerable:
                indexed_docs = sorted({c.doc_id for c in index.chunks})
                docs = st.multiselect("Expected source documents", indexed_docs,
                                      format_func=lambda did: index.docs[did].title, key="gold_new_sources",
                                      help="Select the documents that should support the answer. Evaluation compares retrieved sources with this list. Include all known relevant documents; missing labels can make useful results look wrong.")
                if not indexed_docs:
                    st.info("Add a PDF in the Add PDFs tab first, or choose an example that should not receive a document answer.")
                pages_text = st.text_input("Expected pages (optional)", key="gold_new_pages",
                                           help="Comma-separated page numbers, e.g. 1, 3. Leave blank for any page. Page labels support one selected source document.")
                keywords = st.text_area("Words or numbers the answer must include", key="gold_new_keywords",
                                        placeholder="55 | fifty-five\ngallons | gallon",
                                        help="These are keyword groups: one requirement per line. Use | for alternatives, such as 55 | fifty-five. Every line must have a match. Include acceptable Thai/English variants when relevant. Matching words alone does not prove an answer is correct.")
                st.caption("Example: put ‘55 | fifty-five’ on one line and ‘gallons | gallon’ on the next. The answer must match at least one option on each line.")
            tags = st.text_input("Question groups / tags (optional)", value="user-added", key="gold_new_tags",
                                 help="Comma-separated labels, such as en, threshold, user-added. Tags only help select and compare groups in Evaluation; they do not filter document search or change answers.")
            notes = st.text_area("Review notes (optional)", key="gold_new_notes",
                                 help="Why this is the expected result, or what you checked in the source.")
            submitted = st.form_submit_button("Save test question", type="primary",
                                               help="Saves this question and its expected result locally. It becomes available in Evaluation immediately.")
        st.caption("Saving needs no API key or index rebuild. The question and labels are saved on this machine "
                   "and become available in Evaluation immediately.")
        if submitted:
            try:
                try:
                    pages = list(dict.fromkeys(int(p.strip()) for p in pages_text.split(",") if p.strip()))
                except ValueError:
                    raise ValueError("Pages must be comma-separated whole numbers, such as 1, 3.") from None
                item = Item(id="local-" + uuid.uuid4().hex[:12], question=question.strip(),
                            answerable=answerable, expected_scope=scope, gold_docs=docs, gold_pages=pages,
                            must_include=[[part.strip() for part in line.split("|")] for line in keywords.splitlines() if line.strip()],
                            tags=list(dict.fromkeys(tag.strip() for tag in tags.split(",") if tag.strip())), notes=notes.strip())
                save_item(item, home, index)
                problems = verify([item], index)
                notice = "Saved your test question. Open Evaluation to include it in a test."
                if problems:
                    notice += " Some keywords do not literally appear in the source; review the label check below (translations can differ)."
                st.session_state["gold_saved_notice"] = notice
                st.session_state["gold_clear_form"] = True
                st.rerun()
            except (ValueError, OSError, TimeoutError) as exc:
                st.error(str(exc))


def _outcome(item):
    return next((label for label, values in OUTCOMES.items()
                 if values == (item.answerable, item.expected_scope)), "Review expected behavior")


def _problem_text(problem):
    if problem.startswith("gold docs not indexed:"):
        return "An expected document has no searchable text in the library. Check the source selection or add the document again."
    if problem.startswith("keywords not found in gold docs:"):
        return "Some required words were not found in the expected documents. Review the words and source text; translations may use different wording."
    return problem


def gold_panel(index, show_pdf, home, *, editable=True):
    st.subheader("Your answer key for testing")
    st.markdown("""<style>
    .st-key-gold_intro {border-top: 4px solid #B87920; background: rgba(184,121,32,0.05);}
    .st-key-gold_browse {border-top: 4px solid #2459A6; background: rgba(36,89,166,0.04);}
    .st-key-gold_add {border-top: 4px solid #52677F;}
    .st-key-gold_check {border-top: 4px solid #24715A;}
    </style>""", unsafe_allow_html=True)
    with st.container(border=True, key="gold_intro"):
        st.subheader("What is a gold dataset?", help="Gold is the name for reference examples reviewed by people. A dataset is simply the collection of those examples.")
        st.write("A **gold dataset is your answer key**: a collection of test questions with the sources, key facts, or behavior you expect. A person checks these expectations against the original documents.")
        st.write("**Why we need it:** a convincing answer can still be wrong. An answer key lets you check whether the assistant finds the right sources, includes required facts, and knows when to decline or ask for clarification. Run the same questions after a change to see whether results improve or get worse.")
        st.info("This page defines what you expect. The Evaluation tab runs the tests and compares results with this answer key. Saving an example does not train the assistant or give it the expected answers.")
        with st.expander("See a simple example and how to get started"):
            st.write("For a question such as ‘What is the allowed storage limit?’, first read the relevant PDF. If it says 55 gallons, select that document and page, then require ‘55 | fifty-five’ and ‘gallons | gallon’ on separate lines.")
            st.write("You can also test behavior: an unrelated question should be declined, and an unclear question should receive a clarification request.")
            st.write("Start with common real questions, tricky cases, and examples in both Thai and English. Verify the expected results yourself; a generated answer is not automatically a trusted answer key.")
            st.caption("Scores only cover the examples and labels you provide. Keyword checks do not prove every claim is correct, and this app does not train on these examples.")
    try:
        items = load_dataset(home)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    title = lambda did: index.docs[did].title if did in index.docs else f"Unavailable document ({did})"
    with st.container(border=True, key="gold_browse"):
        st.subheader("Browse saved examples", help="These examples combine the demo's starter questions with questions saved on this machine. Being listed here does not mean an example has passed evaluation or been verified as correct.")
        st.caption(f"{len(items)} test questions in your answer key. Review their expected results against the PDFs before relying on scores.")
        tags = st.multiselect("Filter examples by group", sorted({t for item in items for t in item.tags}), key="gold_tags",
                              help="Groups are evaluation tags. Choose one or more to see questions with any selected tag. Leave empty to see all examples. This does not change document search or the Evaluation tab's selection.")
        selected = [item for item in items if not tags or set(tags).intersection(item.tags)]
        st.caption(f"Showing {len(selected)} of {len(items)} questions.")
        if not selected:
            st.info("No examples to show. Clear the group filter or add a test question below.")
        else:
            if editable:
                st.dataframe([{"Question": item.question, "Expected behavior": _outcome(item),
                               "Expected sources": ", ".join(title(did) for did in item.gold_docs) or "Not needed",
                               "Groups": ", ".join(item.tags)} for item in selected], hide_index=True, use_container_width=True)
            by_id = {item.id: item for item in selected}
            selected_id = st.selectbox("Open a saved question", list(by_id), key="gold_inspect",
                                       format_func=lambda qid: by_id[qid].question,
                                       help="Review what this question expects and read its source passages below. This is the answer key, not an actual model response.")
            item = by_id[selected_id]
            st.write("**Expected behavior:** " + _outcome(item))
            if item.answerable:
                st.write("**Expected sources:** " + ", ".join(title(did) for did in item.gold_docs))
                st.write("**Expected pages:** " + (", ".join(map(str, item.gold_pages)) or "Any page in the expected documents"))
                st.write("**Required words or numbers:** the answer needs at least one option from each line.")
                for group in item.must_include:
                    st.write("• " + " or ".join(group))
                problems = verify([item], index)
                if problems:
                    st.warning(_problem_text(problems[0]["problem"]))
                else:
                    st.caption("The documents are available and keyword checks pass. This does not establish factual correctness; a person still needs to review the source.")
                for did in item.gold_docs:
                    chunks = [chunk for chunk in index.chunks if chunk.doc_id == did and (not item.gold_pages or chunk.page in item.gold_pages)]
                    with st.expander("Read source: " + title(did)):
                        if not chunks:
                            st.warning("No searchable passages were found for the expected document/pages. Check the page numbers and the Library tab.")
                            continue
                        chunks_by_id = {chunk.id: chunk for chunk in chunks}
                        cid = st.selectbox("Choose a source passage", list(chunks_by_id), key=f"gold-{selected_id}-{did}",
                                           format_func=lambda cid, passages=chunks_by_id: f"Page {passages[cid].page} · {passages[cid].text[:90]}…",
                                           help="A passage is a small piece of text extracted from a PDF. Read it, then open the original page to confirm the expected facts.")
                        chunk = chunks_by_id[cid]
                        st.write(chunk.text)
                        if st.button("Open original PDF page", key=f"gold-pdf-{selected_id}-{did}"):
                            show_pdf(did, chunk.page)
            else:
                st.info("No answer keywords or source labels are required for this behavior. Use ‘Check sources and answers’ in Evaluation to test it.")
            if item.notes:
                st.write("**Review notes:** " + item.notes)
            with st.expander("Advanced: question ID and saved labels"):
                st.caption("Labels are the saved expectations used for scoring. The question ID is a stable reference for reports.")
                st.json(asdict(item))
    if editable:
        add_gold_form(index, home)
    else:
        st.info("This shared demo shows the starter answer key. Add and edit test questions in your own local installation.")
        return
    with st.container(border=True, key="gold_check"):
        st.subheader("Check and export your answer key", help="These checks look for missing document text and required words absent from that text. They do not run the assistant or certify the answer key as correct.")
        st.write("Before running an evaluation, check for missing documents or words that need review. This checks all saved examples, regardless of the group filter above.")
        st.caption("No API key or credits are needed. People must still verify the expected facts and behavior. Page numbers and refusal/clarification decisions are not validated by this check.")
        if st.button("Check saved examples", disabled=not items, help="Checks whether expected documents have searchable text and required keyword groups appear somewhere in those documents."):
            problems = verify(items, index)
            checked = sum(item.answerable for item in items)
            st.write(f"Checked source text and keywords for {checked} questions. {len(items) - checked} behavior-only questions need manual review.")
            if problems:
                st.warning(f"{len(problems)} questions need a closer look. Missing keywords can also be caused by translation differences.")
                by_id = {item.id: item for item in items}
                st.dataframe([{"Question": by_id[p["id"]].question, "What to review": _problem_text(p["problem"])} for p in problems], hide_index=True)
            elif checked:
                st.success("No missing source text or keyword problems found. This is a basic check, not a factual accuracy score.")
            else:
                st.info("There are no source or keyword labels to check. Review the expected behavior for each question manually.")
        st.download_button("Download gold dataset as JSON", json.dumps({"items": [asdict(item) for item in items]}, ensure_ascii=False, indent=2),
                           file_name="gold-questions.json", mime="application/json",
                           help="Downloads every saved question and its expected result, including examples outside the current filter. JSON is a structured text file for backup or use with the command-line evaluator. This page does not import files.")
