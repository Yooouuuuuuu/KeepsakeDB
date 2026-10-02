import json

from keepsakedb import Keepsake
from keepsakedb.embedders import HashEmbedder
from keepsakedb.integrate import AgentMemory
from keepsakedb.llms import ScriptedLLM, extract_json
from keepsakedb.pipeline import Pipeline, PipelineConfig


def ops(*items):
    return json.dumps({"ops": list(items)}, ensure_ascii=False)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"ops": [{"op": "noop"}]}\n```')["ops"] == [{"op": "noop"}]
    assert extract_json('{"ops": []} trailing')["ops"] == []


def test_observe_add_supersede_archive(tmp_path):
    ks = Keepsake(str(tmp_path / "p.db"), embedder=HashEmbedder(16))
    llm = ScriptedLLM([
        ops({"op": "add", "text": "使用者十一月要跑新竹馬拉松（半馬）", "tags": ["Sports", "plans"]}),
        "",  # filled below
    ])
    p = Pipeline(ks, llm, PipelineConfig(language="zh-TW"))
    r = p.observe("十一月我要跑新竹馬拉松，報了半馬")
    assert r.error is None and r.applied[0]["op"] == "add" and r.applied[0]["tags"] == ["sports", "plans"]
    mid = r.applied[0]["id"]
    assert "zh-TW" in llm.calls[0][0]  # language reaches the prompt
    # correction -> supersede the candidate; the stale memory is archived with the reason
    llm.replies = [ops({"op": "supersede", "id": mid, "text": "使用者改報全馬（原本半馬）", "reason": "user corrected"})]
    r = p.observe("馬拉松我升級了，從半馬改報全馬")
    assert r.applied[0]["op"] == "supersede" and ks.get(mid).archived
    new_id = r.applied[0]["new_id"]
    assert ks.get(new_id).tags == ["plans", "sports"] and ks.get(new_id).meta["supersedes"] == mid
    # an op on an id that was not a candidate is refused, not applied
    llm.replies = [ops({"op": "update", "id": 9999, "text": "x"}, {"op": "noop"})]
    r = p.observe("隨便聊聊")
    assert r.applied == [] and "not among candidates" in r.skipped[0]["why"]
    # archive by tag with reason -> context shows meta-info only; consent opens it
    llm.replies = [ops({"op": "archive", "tag": "sports", "reason": "user finds it painful"})]
    r = p.observe("先別跟我提馬拉松的事，太痛了")
    assert r.applied[0]["count"] == 1
    block = p.context_for("全馬")
    assert "contents withheld" in block and "全馬" not in block
    assert "半馬" not in p.context_for("馬拉松")  # the superseded fact never surfaces
    p.grant_consent("sports", turns=2)
    assert "全馬" in p.context_for("全馬") and "半馬）" not in p.context_for("馬拉松")
    p.turn(); p.turn()
    assert "全馬" not in p.context_for("全馬")  # window closed
    p.grant_consent("sports")
    p.end_session()
    assert p.consented_tags() == []
    ks.close()


def test_llm_failure_never_breaks_turn(tmp_path):
    ks = Keepsake(str(tmp_path / "f.db"), embedder=HashEmbedder(8))
    p = Pipeline(ks, ScriptedLLM(["not json at all"]))
    r = p.observe("hello")
    assert r.error and r.applied == []
    p2 = Pipeline(ks, ScriptedLLM([ops({"op": "add", "text": "x", "tags": []})]),
                  PipelineConfig(prefilter=lambda t: len(t) > 10))
    assert p2.observe("short").applied == [] and ks.count() == 0
    ks.close()


def test_agent_memory_background(tmp_path):
    ks = Keepsake(str(tmp_path / "a.db"), embedder=HashEmbedder(8))
    llm = ScriptedLLM([ops({"op": "add", "text": "使用者不吃香菜", "tags": ["food"]})])
    mem = AgentMemory(ks, llm)
    assert mem.before_turn("我不吃香菜") == ""  # nothing stored yet
    mem.after_turn("我不吃香菜", "好的我記住了")
    mem.flush()
    assert ks.count() == 1
    # the hash embedder has no semantics, so probe with an overlapping keyword (BM25 leg)
    assert "不吃香菜" in mem.before_turn("晚餐加香菜可以嗎")
    mem.end_session()
    ks.close()


def test_observe_retries_once_on_empty_reply(tmp_path):
    ks = Keepsake(str(tmp_path / "r.db"), embedder=HashEmbedder(8))
    llm = ScriptedLLM(["", ops({"op": "add", "text": "使用者要去東京十天", "tags": ["travel"]})])
    p = Pipeline(ks, llm, PipelineConfig(retry_delay_s=0))
    r = p.observe("日本行程改了喔，東京那趟從一週改成十天")
    assert r.error is None and r.applied[0]["op"] == "add" and len(llm.calls) == 2
    p2 = Pipeline(ks, ScriptedLLM(["", ""]), PipelineConfig(retry_delay_s=0))
    r2 = p2.observe("再說一次")
    assert r2.applied == [] and "after 2 attempts" in r2.error
    ks.close()


def test_subject_marker_is_a_store_constant(tmp_path):
    ks = Keepsake(str(tmp_path / "s.db"), embedder=HashEmbedder(8))
    llm = ScriptedLLM([ops({"op": "noop"})])
    p = Pipeline(ks, llm, PipelineConfig(subject="製作人", user_name="承宇"))
    p.observe("我叫承宇")
    system = llm.calls[0][0]
    assert "«製作人»" in system and '"承宇"' in system and ks._meta("subject") == "製作人"
    p2 = Pipeline(ks, ScriptedLLM([ops({"op": "noop"})]), PipelineConfig(subject="使用者"))
    assert p2.subject == "製作人"  # the store's marker wins
    ks2 = Keepsake(str(tmp_path / "n.db"), embedder=HashEmbedder(8))
    llm2 = ScriptedLLM([ops({"op": "noop"})])
    Pipeline(ks2, llm2).observe("hi")
    assert "«" not in llm2.calls[0][0]  # no subject configured: rule absent
    ks.close(); ks2.close()
