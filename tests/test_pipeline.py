import json

import pymupdf

from envsearch.answer import NOT_FOUND, answer, split_citable, to_search_results
from envsearch.corpus import Document, chunk_document, extract_pages
from envsearch.evaluate import Item, keyword_groups_ok, run, summarize, verify
from envsearch.textnorm import normalize, tokenize

from .conftest import fake_client, response, sr_citation, text_block


# ---------------------------------------------------------------- text normalisation

def test_sara_am_and_thai_digits_are_normalised():
    broken = "กําหนด ๒๐ มิลลิกรัม"            # as extracted from gazette PDFs
    assert normalize(broken) == "กำหนด 20 มิลลิกรัม"
    # both tone-mark orders seen in real PDFs
    assert normalize("น\u0e49\u0e4d\u0e32ทิ้ง") == normalize("น\u0e4d\u0e49\u0e32ทิ้ง") == "น้ำทิ้ง"


def test_thai_tokenisation_segments_words_and_keeps_english():
    toks = tokenize("บีโอดี (Biochemical Oxygen Demand) ไม่เกิน ๒๐ มิลลิกรัมต่อลิตร")
    assert "บีโอดี" in toks and "biochemical" in toks and "20" in toks
    assert "มิลลิกรัม" in toks


# ---------------------------------------------------------------- extraction + chunking

def test_pdf_extraction_is_per_page(tmp_path):
    pdf = pymupdf.open()
    for text in ["First page about containers.", "Second page about manifests."]:
        pdf.new_page().insert_text((72, 72), text)
    path = tmp_path / "x.pdf"
    pdf.save(path)
    pages = extract_pages(path)
    assert len(pages) == 2 and "manifests" in pages[1]


def test_chunks_never_cross_pages_and_ids_encode_page():
    doc = Document(id="d", title="t", url="u")
    long_para = "Sentence about hazardous waste storage rules. " * 60
    chunks = chunk_document(doc, [long_para, "Second page text that is long enough to keep."])
    assert {c.page for c in chunks} == {1, 2}
    assert len([c for c in chunks if c.page == 1]) > 1          # long page was split
    assert all(c.id.startswith(f"d:p{c.page}:") for c in chunks)


# ---------------------------------------------------------------- retrieval

def test_english_query_finds_english_doc(index):
    hits = index.search(["satellite accumulation 55 gallons"], k=3)
    assert hits[0].chunk.doc_id == "fx-saa" and hits[0].chunk.page == 1


def test_thai_query_matches_broken_sara_am_and_thai_digits(index):
    # The fixture stores "นําทิ้ง" and "๒๐"; a user types "น้ำทิ้ง" and "20".
    hits = index.search(["บีโอดี น้ำทิ้ง 20"], k=3)
    assert hits and hits[0].chunk.doc_id == "fx-th-eff"


def test_rrf_fuses_bilingual_queries(index):
    hits = index.search(["BOD limit factory effluent", "บีโอดี น้ำทิ้ง"], k=5)
    assert "fx-th-eff" in {h.chunk.doc_id for h in hits}


def test_language_filter(index):
    assert all(h.chunk.lang == "th" for h in index.search(["pH"], k=5, lang="th"))


# ---------------------------------------------------------------- answer + citations

def test_search_result_blocks_shape(index):
    hits = index.search(["satellite accumulation"], k=2)
    blocks = to_search_results(hits, index)
    b = blocks[0]
    assert b["type"] == "search_result" and b["citations"] == {"enabled": True}
    assert b["source"].endswith(f"#page={hits[0].chunk.page}")
    assert all(x["type"] == "text" and x["text"] for x in b["content"])


def test_split_citable_breaks_sentences():
    parts = split_citable("First rule applies. Second rule applies. Third rule applies.")
    assert len(parts) == 3


def test_answer_maps_citations_to_chunks(index):
    def respond(kw):
        if "tools" in kw or kw.get("max_tokens") == 300:     # query expansion call
            return response(text_block('{"en": ["satellite accumulation area limit"], "th": []}'))
        n_results = sum(1 for b in kw["messages"][0]["content"] if b["type"] == "search_result")
        assert n_results >= 1
        return response(
            text_block("Up to 55 gallons of non-acute hazardous waste.", [sr_citation(0, 1, 2, "up to 55 gallons")]),
            text_block(" Acute: one quart.", [sr_citation(0, 2, 3, "one quart")]),
        )
    client = fake_client(respond)
    a = answer("How much waste can a satellite accumulation area hold?", index, client, model="m", rewrite_model="r")
    assert a.found and len(a.citations) == 2
    assert a.citations[0].doc_id == "fx-saa" and a.citations[0].page == 1
    assert "[1]" in a.text and "[2]" in a.text
    assert "satellite accumulation area limit" in a.queries
    assert "Sources:" in a.render()


def test_refusal_is_detected(index):
    client = fake_client(lambda kw: response(text_block(f"{NOT_FOUND} The documents cover US hazardous waste.")))
    a = answer("What is the EU REACH deadline for hazardous waste?", index, client, model="m", expand=False)
    assert not a.found and a.citations == []


# ---------------------------------------------------------------- eval

def test_keyword_groups_normalise_commas_and_thai_digits():
    ok, _ = keyword_groups_ok("ไม่เกิน ๒๐ มก./ล. or 1,000 kg", [["20"], ["1000 kg"]])
    assert ok


def test_verify_flags_bad_gold_labels(index):
    items = [Item(id="good", question="q", gold_docs=["fx-cat"], must_include=[["100 kilograms"]]),
             Item(id="bad", question="q", gold_docs=["fx-cat"], must_include=[["55 gallons"]]),
             Item(id="missing", question="q", gold_docs=["nope"], must_include=[["x"]])]
    assert {p["id"] for p in verify(items, index)} == {"bad", "missing"}


def test_eval_run_end_to_end_with_fake_claude(index):
    def respond(kw):
        q = kw["messages"][0]["content"][-1]["text"]
        if "REACH" in q:
            return response(text_block(NOT_FOUND))
        return response(text_block("55 gallons", [sr_citation(0, 0, 1, "55 gallons")]))
    items = [Item(id="saa", question="satellite accumulation gallons", gold_docs=["fx-saa"], must_include=[["55"]], tags=["en"]),
             Item(id="neg", question="EU REACH registration hazardous waste", answerable=False, tags=["neg"])]
    result = run(items, index, fake_client(respond), model="m", rewrite_model="r", k=3, expand=False, log=lambda *_: None)
    s = result["summary"]["overall"]
    assert s["retrieval.hit@k"] == 1.0 and s["answer.correct"] == 1.0 and s["refusal.correct"] == 1.0
    json.dumps(result)  # report must be serialisable


def test_retrieval_only_eval_needs_no_client(index):
    items = [Item(id="cat", question="very small quantity generator kilograms", gold_docs=["fx-cat"])]
    result = run(items, index, None, model="m", rewrite_model="r", retrieval_only=True, expand=False, log=lambda *_: None)
    assert result["summary"]["overall"]["retrieval.hit@k"] == 1.0
    assert summarize(result["rows"])["overall"]["n"] == 1
