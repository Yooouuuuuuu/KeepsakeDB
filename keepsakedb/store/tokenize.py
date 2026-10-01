"""Tokenizers for the BM25 leg.

SQLite FTS5's built-in tokenizers cannot search Chinese/Japanese/Korean memories:
``unicode61`` keeps a whole CJK run as ONE token and ``trigram`` needs >= 3 chars per
term (so 2-char words such as 罐頭 never match). Hermes solves this with a C loadable
tokenizer (cjk_unicode61); KeepsakeDB keeps zero native dependencies by pre-tokenizing
in Python and letting FTS5 index a space-joined token string with ``unicode61``.

- ``bigram`` (default, no deps): CJK runs -> overlapping character bigrams (Lucene
  CJKAnalyzer / Hermes cjk_unicode61 semantics). At query time the bigrams of one run
  become an FTS5 *phrase*, which is exact substring matching down to 2 characters.
- ``jieba`` (optional): word segmentation, ``cut_for_search`` like Milvus Lite's
  JiebaAnalyzer. Exact-word matching, no substring behaviour.
"""
from __future__ import annotations

import re
from typing import List, Protocol

_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # Extension A
    (0x20000, 0x2A6DF),  # Extension B
    (0xF900, 0xFAFF),    # Compatibility Ideographs
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0xAC00, 0xD7AF),    # Hangul Syllables
)


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _fts_quote(tok: str) -> str:
    return '"' + tok.replace('"', '""') + '"'


class Tokenizer(Protocol):
    name: str

    def units(self, text: str) -> List[List[str]]:
        """Return search units: each unit is the ordered token list of one word/run."""
        ...


class BigramTokenizer:
    """CJK runs -> character bigrams; everything else -> lowercased \\w+ words."""

    name = "bigram"

    def units(self, text: str) -> List[List[str]]:
        out: List[List[str]] = []
        for m in _WORD_RE.finditer(text.lower()):
            word = m.group(0)
            run: List[str] = []
            latin: List[str] = []
            for ch in word:
                if is_cjk(ch):
                    if latin:
                        out.append(["".join(latin)])
                        latin = []
                    run.append(ch)
                else:
                    if run:
                        out.append(self._emit(run))
                        run = []
                    latin.append(ch)
            if run:
                out.append(self._emit(run))
            if latin:
                out.append(["".join(latin)])
        return out

    @staticmethod
    def _emit(run: List[str]) -> List[str]:
        s = "".join(run)
        return [s] if len(s) == 1 else [s[i:i + 2] for i in range(len(s) - 1)]


class JiebaTokenizer:
    """Word segmentation via jieba (``pip install keepsakedb[jieba]``)."""

    name = "jieba"

    def __init__(self, mode: str = "search", big_dict: bool = False) -> None:
        import jieba  # clear ImportError if missing

        self._jieba = jieba
        self._mode = mode
        if big_dict:  # traditional-Chinese-friendly dictionary shipped with jieba
            import os

            jieba.set_dictionary(os.path.join(os.path.dirname(jieba.__file__), "dict.txt.big"))
        jieba.setLogLevel(40)

    def units(self, text: str) -> List[List[str]]:
        raw = self._jieba.cut_for_search(text) if self._mode == "search" else self._jieba.cut(text)
        out: List[List[str]] = []
        for t in raw:
            t = t.strip().lower()
            if t and _WORD_RE.search(t):
                out.append([t])
        return out


def get_tokenizer(spec: "str | Tokenizer") -> Tokenizer:
    if not isinstance(spec, str):
        return spec
    if spec == "bigram":
        return BigramTokenizer()
    if spec == "jieba":
        return JiebaTokenizer()
    if spec == "jieba-big":
        return JiebaTokenizer(big_dict=True)
    raise ValueError(f"unknown tokenizer {spec!r} (bigram | jieba | jieba-big)")


def index_string(tok: Tokenizer, text: str) -> str:
    """Space-joined tokens for the FTS5 column."""
    return " ".join(t for unit in tok.units(text) for t in unit)


def fts_query(tok: Tokenizer, text: str, phrase: bool = False, prefix_single_cjk: bool = True) -> str:
    """FTS5 MATCH expression.

    Default (``phrase=False``): a bag of terms OR-ed together — BM25 sums the contribution
    of every token like Milvus does; right for questions and sentences ("我最愛吃的食物").
    ``phrase=True``: each CJK run becomes an FTS5 phrase of its bigrams = exact substring
    matching down to 2 characters (the Hermes cjk_unicode61 trick); right for looking up
    a name or a keyword ("阿婆麵店").
    """
    parts: List[str] = []
    for unit in tok.units(text):
        if len(unit) == 1:
            t = unit[0]
            if prefix_single_cjk and len(t) == 1 and is_cjk(t):
                parts.append(_fts_quote(t) + "*")   # single char: any bigram starting with it
            else:
                parts.append(_fts_quote(t))
        elif phrase:
            parts.append('"' + " ".join(x.replace('"', '""') for x in unit) + '"')
        else:
            parts.extend(_fts_quote(x) for x in unit)
    # dedupe while keeping order (repeated bigrams add nothing under OR)
    seen = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return " OR ".join(uniq)
