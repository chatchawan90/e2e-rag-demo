from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from envsearch.evaluate import Item
from envsearch.gold import load_dataset, save_item, local_path


def example():
    return Item("local-test", "How much waste can a satellite area hold?", gold_docs=["fx-saa"],
                must_include=[["55", "fifty-five"]], tags=["en", "user-added"], gold_pages=[1])


def test_gold_persists_and_keeps_seed_unchanged(index, tmp_path):
    seed = tmp_path / "seed.yaml"
    original = 'items:\n  - id: seed\n    question: Seed question\n'
    seed.write_text(original)
    save_item(example(), tmp_path, index, seed_path=seed)
    items = load_dataset(tmp_path, seed_path=seed)
    assert [i.id for i in items] == ["seed", "local-test"]
    assert items[-1].must_include == [["55", "fifty-five"]]
    assert seed.read_text() == original


def test_duplicate_gold_rejected_without_changing_store(index, tmp_path):
    save_item(example(), tmp_path, index)
    before = local_path(tmp_path).read_bytes()
    with pytest.raises(ValueError, match="already"):
        save_item(replace(example(), id="another", question="  HOW MUCH WASTE CAN A SATELLITE AREA HOLD?  "), tmp_path, index)
    assert local_path(tmp_path).read_bytes() == before


@pytest.mark.parametrize("changes", [{"question":" "}, {"gold_docs":[]}, {"gold_docs":["unknown"]},
    {"gold_pages":[999]}, {"must_include":[]}, {"must_include":[[""]]},
    {"expected_scope":"out_of_scope"}])
def test_invalid_labels_not_saved(index, tmp_path, changes):
    with pytest.raises(ValueError):
        save_item(replace(example(), **changes), tmp_path, index)
    assert not local_path(tmp_path).exists()


def test_scope_and_missing_evidence_questions_need_no_sources(index, tmp_path):
    for scope in ("in_scope", "out_of_scope", "clarify"):
        save_item(Item("local-" + scope, "Question " + scope, answerable=False, expected_scope=scope), tmp_path, index)
    assert len([i for i in load_dataset(tmp_path) if i.id.startswith("local-")]) == 3


def test_concurrent_additions_do_not_overwrite_each_other(index, tmp_path):
    def save(n):
        save_item(replace(example(), id=f"local-{n}", question=f"Question {n}?"), tmp_path, index)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(8)))
    assert len([i for i in load_dataset(tmp_path) if i.id.startswith("local-")]) == 8


def test_corrupt_local_dataset_is_not_overwritten(index, tmp_path):
    path = local_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("broken json")
    with pytest.raises(ValueError):
        save_item(example(), tmp_path, index)
    assert path.read_text() == "broken json"
