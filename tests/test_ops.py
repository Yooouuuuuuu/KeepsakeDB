"""Ownership / command queue, external-change reload, backups, explain, logs, CLI."""
import json
import os
import time
from datetime import datetime, timedelta

import pytest

from keepsakedb import Keepsake, cli
from keepsakedb.embedders import HashEmbedder
from keepsakedb.llms import ScriptedLLM
from keepsakedb.pipeline import Pipeline


def test_owner_queue_and_reader_reload(tmp_path):
    p = str(tmp_path / "o.db")
    owner = Keepsake(p, embedder=HashEmbedder(16), owner=True)
    reader = Keepsake(p, embedder=HashEmbedder(16))          # e.g. 556d or the CLI
    assert not reader.owner_alive()
    mid = owner.add("我不吃香菜", ["food"])
    # the reader opened before the write; data_version makes it reload the new row
    hit = reader.search("我不吃香菜", k=1, mode="dense")[0]
    assert hit.id == mid and hit.dense == pytest.approx(1.0, abs=1e-5)
    owner.archive([mid], reason="test")
    assert reader.search("我不吃香菜", k=1, mode="dense") == []
    owner.unarchive([mid])
    # a write from the reader side goes through the queue; the owner drains it
    owner.drain_commands()                                   # publishes the heartbeat
    assert reader.owner_alive()
    cid = reader.enqueue("delete", {"ids": [mid], "reason": "cli"}, by="test")
    assert reader.pending_commands()[0]["op"] == "delete"
    assert owner.drain_commands() == 1
    assert reader.wait_command(cid, timeout=2) == {"count": 1}
    assert owner.count() == 0 and owner.search("香菜", mode="bm25") == []
    with pytest.raises(ValueError):
        reader.enqueue("format-disk")
    with pytest.raises(RuntimeError):
        reader.drain_commands()
    # owner loop drains on a timer
    owner.start_owner_loop(interval=0.1)
    cid = reader.enqueue("ping")
    assert reader.wait_command(cid, timeout=3)["ok"] is True
    owner.close()
    time.sleep(0.05)
    assert not reader.owner_alive()
    reader.close()


def test_backup_snapshot_rotate_restore(tmp_path):
    p = str(tmp_path / "b.db")
    bdir = str(tmp_path / "backups")
    k = Keepsake(p, embedder=HashEmbedder(1024), backup_dir=bdir)  # real-sized vectors: text-only must be smaller
    ids = [k.add(t, ["t"]) for t in ("我養了一隻貓叫Mochi", "我最喜歡吃麻辣鍋", "我平常都騎 Gogoro 上班")]
    assert k._meta("last_snapshot_day") is not None          # daily snapshot on first write
    assert len(os.listdir(bdir)) == 1
    snap_full = k.backup(str(tmp_path / "full.db"))
    snap_text = k.backup(str(tmp_path / "text.db"), include_embeddings=False)
    assert os.path.getsize(snap_text) < os.path.getsize(snap_full)
    k.add("一條之後才加的記憶", ["t"])
    assert k.count() == 4
    k.restore(snap_text)                                      # text-only -> re-embedded (embedder attached)
    assert k.count() == 3 and len(k.dense) == 3
    # the hash embedder has no semantics: probe with the exact text to prove the vectors are back
    assert k.search("我最喜歡吃麻辣鍋", k=1, mode="dense")[0].dense == pytest.approx(1.0, abs=1e-5)
    assert k._meta("text_only") is None
    # rotation keeps 7 daily + weekly + monthly tiers, judged from the newest snapshot
    rdir = str(tmp_path / "rot")
    os.makedirs(rdir)
    base = datetime(2026, 3, 31, 12, 0, 0)
    for d in range(60):
        open(os.path.join(rdir, f"b-{(base - timedelta(days=d)).strftime('%Y%m%d-%H%M%S')}.db"), "w").close()
    open(os.path.join(rdir, "other-20260101-000000.db"), "w").close()  # another store's file: untouched
    deleted = k.rotate(rdir)
    left = sorted(os.listdir(rdir))
    assert "other-20260101-000000.db" in left
    kept = [f for f in left if f.startswith("b-")]
    assert 10 <= len(kept) <= 14 and len(deleted) == 60 - len(kept)
    assert "b-20260331-120000.db" in kept and any(f < "b-20260305" for f in kept)   # newest + a monthly one
    k.close()


def test_explain_and_logs(tmp_path):
    k = Keepsake(str(tmp_path / "e.db"), embedder=HashEmbedder(16))
    k.add("我養了一隻貓叫Mochi，牠最愛吃罐頭", ["pets"])
    k.add("我平常都騎 Gogoro 上班", ["commute"])
    ex = k.explain("罐頭 Mochi", k=2)
    assert ex["fts_match"] == '"罐頭" OR "mochi"' and ex["rrf_k"] == 60.0
    assert ex["fused"][0]["bm25_rank"] == 1 and ex["fused"][0]["text"].startswith("我養了")
    assert {"dense", "bm25", "fused", "thresholds", "legs_k"} <= set(ex)
    # search log on by default; context marks injected rows; prune by age
    k.context("罐頭")
    assert any(r["injected"] == 1 for r in k.search_log())
    k.conn.execute("UPDATE search_log SET at = at - 200 * 86400")
    assert k.prune_logs(90) >= 1 and k.search_log() == []
    # pipeline log records candidates, raw reply, applied ops, model, latency
    pl = Pipeline(k, ScriptedLLM([json.dumps({"ops": [{"op": "add", "text": "使用者不吃香菜", "tags": ["food"]}]})]))
    pl.observe("我不吃香菜")
    row = k.pipeline_log_rows()[0]
    assert row["model"] == "scripted" and row["applied"][0]["op"] == "add" and row["ms"] >= 0 and "ops" in row["raw"]
    st = k.stats()
    assert st["memories"] == 3 and st["pipeline_log_rows"] == 1 and st["owner_alive"] is False
    k.close()


def test_cli_smoke(tmp_path, capsys):
    p = str(tmp_path / "c.db")
    k = Keepsake(p)
    k.add("我固定每週三晚上跟大學同學打線上遊戲，玩 LOL", ["hobbies"])
    k.add("公司附近那間「阿婆麵店」是我的愛店", ["food"])
    k.close()
    assert cli.main(["--db", p, "stats"]) == 0
    assert "memories" in capsys.readouterr().out
    assert cli.main(["--db", p, "search", "阿婆麵店", "--explain"]) == 0
    out = capsys.readouterr().out
    assert "fts match" in out and "阿婆麵店" in out
    assert cli.main(["--db", p, "archive", "--tag", "food", "--reason", "painful"]) == 0
    assert "'count': 1" in capsys.readouterr().out
    assert cli.main(["--db", p, "list", "--archived", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2 and any(r["archived"] for r in rows)
    assert cli.main(["--db", p, "delete", "1"]) == 0
    assert cli.main(["--db", p, "show", "1"]) == 1
    assert cli.main(["--db", p, "backup", "--dir", str(tmp_path / "bk"), "--text-only"]) == 0
    assert len(os.listdir(tmp_path / "bk")) == 1


def test_hold_and_patch(tmp_path):
    """A turn's memories can be held (kept, not shown, no hint) and later released as 舊帳 with a prefix."""
    p = str(tmp_path / "h.db")
    owner = Keepsake(p, embedder=HashEmbedder(16), owner=True)
    pl = Pipeline(owner, ScriptedLLM([]))
    mid = owner.add("他建議我走露一點的路線吸引流量", ["work"], meta={"turn_id": "t1"})
    other = owner.add("他喜歡吃咖哩", ["food"], meta={"turn_id": "t2"})
    assert owner.ids_for_turn("t1") == [mid]
    assert "露一點" in pl.context_for("走露一點的路線", k=5)
    # held: not injected, not hinted; still a search / observe candidate; the other memory unaffected
    assert owner.hold(turn_id="t1", reason="委屈帳") == 1
    assert owner.get(mid).held and not owner.get(other).held
    block = pl.context_for("走露一點的路線", k=5)
    assert "露一點" not in block and "archived" not in block
    assert any(m.id == mid for m in owner.search("走露一點的路線", k=5, archived="include"))
    # the command path (the psyche engine writes commands; the owner drains them)
    reader = Keepsake(p, embedder=HashEmbedder(16))
    reader.enqueue("hold", {"turn_id": "t1", "on": False, "reason": "forgiven"}, by="psyche")
    reader.enqueue("patch", {"turn_id": "t1", "add_tags": ["舊帳"], "meta": {"old_account_date": "2026/10/13", "prefix": "（舊帳 2026/10/13）"}}, by="psyche")
    owner.drain_commands()
    m = owner.get(mid)
    assert not m.held and "舊帳" in m.tags and m.meta["old_account_date"] == "2026/10/13"
    block = pl.context_for("走露一點的路線", k=5)
    assert "（舊帳 2026/10/13）他建議我走露一點的路線吸引流量" in block
    assert owner.tags()["舊帳"] == 1
