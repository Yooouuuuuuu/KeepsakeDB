import json

from keepsakedb import Keepsake
from keepsakedb.embedders import HashEmbedder
from keepsakedb.llms import ScriptedLLM
from keepsakedb.pipeline import Pipeline, PipelineConfig


def test_layers_add_search_ring(tmp_path):
    ks = Keepsake(str(tmp_path / "l.db"), embedder=HashEmbedder(16))
    a = ks.add("製作人養了一隻叫小黑的貓", ["pets"])  # long by default
    n = ks.add("製作人在整理事務所的名單", [], {"kind": "note"}, layer="short")
    assert ks.get(a).layer == "long" and ks.get(n).layer == "short"
    assert [m.id for m in ks.search("小黑的貓", k=5, layer="long")] == [a]
    assert all(m.layer == "short" for m in ks.search("小黑的貓", k=5, layer="short"))  # dense leg has no floor; the layer filter holds
    assert all(m.layer == "short" for m in ks.search("名單", k=5, layer="short"))
    r = ks.search_layers("名單 小黑", layers={"long": 5, "short": 5})
    assert set(r) == {"long", "short"} and r["long"][0].id == a and r["short"][0].id == n
    assert ks.stats()["layers"] == {"long": 1, "short": 1}
    for i in range(12):
        ks.add(f"第 {i} 則紀錄", [], layer="short")
    assert ks.prune_layer("short", keep=5) == 8
    assert ks.count(layer="short") == 5 and ks.count(layer="long") == 1
    assert ks.stats()["layers"]["short"] == 5
    # the newest survive the ring
    assert sorted(m.text for m in ks.search("紀錄", k=10, layer="short")) == sorted(f"第 {i} 則紀錄" for i in range(7, 12))
    # a correction keeps its layer
    nid = ks.supersede(a, "製作人養了一隻叫小白的貓", reason="renamed")
    assert ks.get(nid).layer == "long"
    ks.close()


def test_search_layers_reopens_after_external_write(tmp_path):
    path = str(tmp_path / "s.db")
    ks = Keepsake(path, embedder=HashEmbedder(16))
    ks.add("一則長期記憶", ["x"])
    other = Keepsake(path, embedder=HashEmbedder(16))
    other.add("另一個連線寫的紀錄", [], layer="short")
    other.close()
    r = ks.search_layers("紀錄 記憶", layers=["long", "short"], k=3)
    assert r["long"] and r["short"] and r["short"][0].text == "另一個連線寫的紀錄"
    ks.close()


def test_pipeline_note_goes_to_short_layer(tmp_path):
    ks = Keepsake(str(tmp_path / "p.db"), embedder=HashEmbedder(16))
    llm = ScriptedLLM([json.dumps({"ops": [{"op": "add", "text": "製作人最常用事務所的名單分頁", "tags": ["work"]}],
                                   "note": "製作人下午在整理事務所的名單"}, ensure_ascii=False)])
    p = Pipeline(ks, llm, PipelineConfig(short_layer=True, short_cap=3, surface="事務所"))
    r = p.observe("我下午在整理名單，名單分頁我最常用")
    assert r.error is None and r.note == "製作人下午在整理事務所的名單"
    assert [o["op"] for o in r.applied] == ["add", "note"]
    assert ks.count(layer="long") == 1 and ks.count(layer="short") == 1
    assert ks.search("名單", k=5, layer="short")[0].meta["surface"] == "事務所"
    assert "note" in llm.calls[0][0]  # the rule reached the prompt
    block = p.context_for("名單")
    assert block.startswith("Memory:") and "Facts (long-term):" in block and "Notes (recent, dated):" in block
    assert "事務所] 製作人下午在整理事務所的名單" in block
    # null note -> nothing written; the ring keeps the newest 3
    llm.replies = [json.dumps({"ops": [{"op": "noop"}], "note": None})]
    assert p.observe("嗨").note is None and ks.count(layer="short") == 1
    for i in range(4):
        llm.replies = [json.dumps({"ops": [{"op": "noop"}], "note": f"紀錄 {i}"}, ensure_ascii=False)]
        p.observe(f"第 {i} 回合")
    assert ks.count(layer="short") == 3 and ks.count(layer="long") == 1
    # short layer off: no note rule, no note written
    q = Pipeline(ks, ScriptedLLM([json.dumps({"ops": [{"op": "noop"}], "note": "ignored"})]),
                 PipelineConfig(short_layer=False))
    assert q.observe("隨便").note is None and ks.count(layer="short") == 3
    assert "9. Note" not in q.llm.calls[0][0]
    ks.close()
