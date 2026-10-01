import os

import numpy as np
import pytest

from keepsakedb import Keepsake
from keepsakedb.embedders import HashEmbedder

FACTS = [
    ("我養了一隻貓叫Mochi，牠最愛吃罐頭", ["pets"]),
    ("我最喜歡吃麻辣鍋，越辣越好", ["food"]),
    ("我平常都騎 Gogoro 上班，白色的", ["commute"]),
    ("十一月我要跑新竹馬拉松，報了半馬", ["sports"]),
    ("我媽媽住在台中，她開了一間花店", ["family"]),
]


@pytest.fixture(params=["numpy", "sqlite-vec"])
def ks(tmp_path, request):
    if request.param == "sqlite-vec":
        pytest.importorskip("sqlite_vec")
    k = Keepsake(str(tmp_path / "t.db"), embedder=HashEmbedder(16), dense_index=request.param)
    for text, tags in FACTS:
        k.add(text, tags)
    yield k
    k.close()


def test_bm25_cjk_substring(ks):
    hits = ks.search("罐頭", mode="bm25")
    assert hits and hits[0].text.startswith("我養了一隻貓") and hits[0].bm25 > 0
    assert ks.search("gogoro", mode="bm25")[0].tags == ["commute"]
    assert ks.search("半馬", mode="bm25")[0].tags == ["sports"]


def test_hybrid_plumbing_and_labels(ks):
    hits = ks.search("麻辣鍋", k=3)
    assert hits[0].tags == ["food"]
    assert hits[0].bm25 is not None and hits[0].dense is not None
    assert all(h.label in ("plain", "high") for h in hits)
    # exact same text embeds identically -> cosine 1.0 -> high label
    hits = ks.search(FACTS[1][0], k=1)
    assert hits[0].label == "high" and hits[0].dense == pytest.approx(1.0, abs=1e-5)


def test_archive_is_a_state_with_reason(ks):
    n = ks.archive(tag="sports", reason="user finds it painful")
    assert n == 1
    assert ks.search("半馬", mode="bm25") == []  # hidden by default
    meta = ks.search("半馬", mode="bm25", archived="meta")
    assert meta and meta[0].label == "archived" and meta[0].text == "" and meta[0].archive_reason == "user finds it painful"
    block = ks.context("半馬 馬拉松", mode="bm25")
    assert "[archived: sports — user finds it painful — contents withheld; offer to discuss]" in block
    ks.unarchive(tag="sports")
    assert ks.search("半馬", mode="bm25")[0].tags == ["sports"]


def test_supersede_history_delete(ks):
    old = ks.search("半馬", mode="bm25")[0]
    new_id = ks.supersede(old.id, "馬拉松我升級了，從半馬改報全馬", reason="user correction")
    assert ks.get(old.id).archived and ks.get(old.id).superseded and ks.get(new_id).meta["supersedes"] == old.id
    ops = [h["op"] for h in ks.history(old.id)]
    assert ops == ["add", "supersede"]
    # superseded facts are hidden even from the meta-info view; only explicit include shows them
    assert all(not h.superseded for h in ks.search("半馬", mode="bm25", archived="meta"))
    assert any(h.superseded for h in ks.search("半馬", mode="bm25", archived="include"))
    assert ks.unarchive([old.id]) == 0  # not a suppressed topic; stays superseded
    assert ks.tags() == {"pets": 1, "food": 1, "commute": 1, "family": 1, "sports": 1}
    assert ks.delete([new_id], reason="purge") == 1
    assert ks.get(new_id) is None and ks.history(new_id)[-1]["op"] == "delete"
    assert ks.count() == 4 and ks.count(include_archived=True) == 5


def test_reopen_loads_dense(tmp_path):
    p = str(tmp_path / "r.db")
    k = Keepsake(p, embedder=HashEmbedder(8))
    k.add("我不吃香菜", ["food"])
    k.close()
    k2 = Keepsake(p, embedder=HashEmbedder(8))
    assert k2.dims == 8 and len(k2.dense) == 1
    assert k2.search("香菜", k=1)[0].dense == pytest.approx(1.0, abs=1e-5) or k2.search("香菜", k=1)[0].bm25 > 0
    with pytest.raises(ValueError):
        Keepsake(p, tokenizer="jieba")
    k2.close()


def test_bm25_only_store(tmp_path):
    k = Keepsake(str(tmp_path / "b.db"))  # no embedder: lightweight lexical memory
    k.add("我固定每週三晚上跟大學同學打線上遊戲，玩 LOL")
    assert k.search("LOL", k=1)[0].dense is None
    assert k.search("線上遊戲", k=1)[0].bm25 > 0
    k.close()
