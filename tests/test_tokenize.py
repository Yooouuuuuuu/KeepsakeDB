import pytest

from keepsakedb.store.tokenize import BigramTokenizer, fts_query, index_string


def test_bigram_units_mixed():
    t = BigramTokenizer()
    assert t.units("我養了一隻貓叫Mochi，牠最愛吃罐頭") == [
        ["我養", "養了", "了一", "一隻", "隻貓", "貓叫"], ["mochi"], ["牠最", "最愛", "愛吃", "吃罐", "罐頭"]]
    assert t.units("騎 Gogoro 上班") == [["騎"], ["gogoro"], ["上班"]]


def test_fts_query_shapes():
    t = BigramTokenizer()
    assert fts_query(t, "罐頭") == '"罐頭"'
    assert fts_query(t, "新竹市") == '"新竹" OR "竹市"'              # bag of bigrams (questions)
    assert fts_query(t, "新竹市", phrase=True) == '"新竹 竹市"'       # exact substring (lookups)
    assert fts_query(t, "貓 gogoro") == '"貓"* OR "gogoro"'
    assert fts_query(t, "麻辣麻辣") == '"麻辣" OR "辣麻"'            # repeated bigrams deduped
    assert index_string(t, "罐頭 Mochi") == "罐頭 mochi"


def test_jieba_optional():
    pytest.importorskip("jieba")
    from keepsakedb.store.tokenize import JiebaTokenizer
    t = JiebaTokenizer()
    toks = [u[0] for u in t.units("我最喜歡吃麻辣鍋")]
    assert "麻辣" in toks or "麻辣鍋" in toks
