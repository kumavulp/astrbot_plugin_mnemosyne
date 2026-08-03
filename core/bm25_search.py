"""
bm25_search.py — BM25 关键词打分，配合 jieba 中文分词。
移植自 Ombre Brain (github.com/P0luz/Ombre-Brain)，适配 Mnemosyne 的检索后处理。

与 Ombre Brain 的差异：
- OB 对全库建持久倒排索引；Mnemosyne 的候选集来自 Milvus 向量召回（几十条），
  所以这里对候选集做轻量即时 BM25 打分，无需维护索引生命周期。
- rank_bm25 / jieba 均为软依赖：未安装时 score_candidates 返回空 dict，
  调用方回退到 fuzzy/精确匹配，不影响检索可用性。
"""
from __future__ import annotations

import logging

logger = logging.getLogger("Mnemosyne.bm25")

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False

try:
    import jieba as _jieba
    _jieba.setLogLevel(logging.WARNING)
    _JIEBA_AVAILABLE = True
except ImportError:
    _jieba = None  # type: ignore
    _JIEBA_AVAILABLE = False


def bm25_available() -> bool:
    return _BM25_AVAILABLE


def _tokenize(text: str) -> list[str]:
    """中文 jieba 搜索模式分词 + 英文空格切割，小写，过滤空串。"""
    if not text:
        return []
    text = text.lower()
    if _JIEBA_AVAILABLE:
        tokens = list(_jieba.cut_for_search(text))
    else:
        tokens = text.split()
    return [t for t in tokens if t.strip()]


def score_candidates(query: str, candidates: list[dict]) -> dict[int, float]:
    """
    对 Milvus 召回的候选记忆做即时 BM25 打分。

    Args:
        query: 用户查询文本
        candidates: 候选记忆列表，每项需含 content 字段；
                    以列表下标为 key 返回归一化分值

    Returns:
        {candidate_index: normalized_score}，最高分=1.0；
        无命中或依赖缺失时返回 {}
    """
    if not _BM25_AVAILABLE or not query or not candidates:
        return {}

    corpus: list[list[str]] = []
    idx_map: list[int] = []
    for i, item in enumerate(candidates):
        content = str(item.get("content", ""))
        tokens = _tokenize(content[:1500])
        if tokens:
            corpus.append(tokens)
            idx_map.append(i)

    if not corpus:
        return {}

    query_tokens = _tokenize(query)
    if not query_tokens:
        return {}

    try:
        index = _BM25Okapi(corpus)
        raw = index.get_scores(query_tokens)
    except Exception as e:
        logger.debug(f"BM25 打分失败，回退: {e}")
        return {}

    max_s = float(raw.max()) if raw.size > 0 else 0.0
    if max_s <= 0:
        return {}
    return {idx_map[j]: float(s) / max_s for j, s in enumerate(raw) if s > 0}
