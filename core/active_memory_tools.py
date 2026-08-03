"""
active_memory_tools.py — Kai 的主动记忆能力
参考 Ombre Brain 的 hold(pinned)/feel/trace(resolved)/breath 设计，
适配 Mnemosyne 的 Milvus 存储。

四个能力：
- pin_memory:     主动钉住一件重要的事，永不衰减
- write_feel:     写下第一人称感受（不是对话总结，是"我此刻怎么想"）
- resolve_memory: 标记某件事已翻篇，加速淡出
- recall_memory:  主动检索记忆
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from .constants import VECTOR_FIELD_NAME
from .tools import pack_memory_content, split_memory_content_meta
from .memory_operations import (
    _build_lightweight_graph_metadata,
    _store_summary_to_milvus,
    _get_persona_id,
    validate_session_id,
)

if TYPE_CHECKING:
    from ..main import Mnemosyne


async def _store_with_meta(
    plugin: "Mnemosyne",
    event,
    content: str,
    extra_meta: dict[str, Any],
    source: str,
) -> bool:
    """带额外元数据的直接写入（pin/feel 共用）。"""
    normalized = content.strip() if isinstance(content, str) else ""
    if not normalized:
        return False

    session_id = event.unified_msg_origin
    if not session_id or not validate_session_id(session_id):
        logger.error(f"[active_memory] 无效 session_id: {session_id}")
        return False

    if not plugin.embedding_provider:
        logger.error("[active_memory] Embedding Provider 不可用")
        return False

    try:
        embedding_vector = await plugin.embedding_provider.get_embedding(normalized)
    except Exception as e:
        logger.error(f"[active_memory] Embedding 失败: {e}")
        return False
    if not embedding_vector:
        return False

    persona_id = None
    try:
        persona_id = await _get_persona_id(plugin, event)
    except Exception:
        pass

    metadata = _build_lightweight_graph_metadata(
        normalized,
        context_history=[{
            "role": "user",
            "content": normalized,
            "metadata": {"speaker_id": str(event.get_sender_id())},
        }],
    )
    metadata["source"] = source
    metadata.update(extra_meta)
    stored_content = pack_memory_content(normalized, metadata)

    return await _store_summary_to_milvus(
        plugin=plugin,
        persona_id=persona_id,
        session_id=session_id,
        summary_text=stored_content,
        embedding_vector=embedding_vector,
    )


async def pin_memory(plugin: "Mnemosyne", event, content: str) -> bool:
    """钉住一条永久记忆。meta.pinned=True，检索衰减时恒定高权重。"""
    return await _store_with_meta(
        plugin, event, content,
        extra_meta={"pinned": True},
        source="kai_pin",
    )


async def write_feel(plugin: "Mnemosyne", event, content: str) -> bool:
    """写下第一人称感受。meta.type=feel，不衰减不增强，作为痕迹留存。"""
    prefixed = content if content.startswith("【") else f"【Kai的感受】{content}"
    return await _store_with_meta(
        plugin, event, prefixed,
        extra_meta={"type": "feel"},
        source="kai_feel",
    )


async def resolve_memory(plugin: "Mnemosyne", event, query: str) -> str:
    """
    搜索最匹配的一条记忆并标记 resolved=True（加速淡出）。
    Milvus 不支持原地更新：查出 → 改 meta → 重嵌入插入 → 删旧。
    返回被标记的记忆摘要，失败返回空串。
    """
    session_id = event.unified_msg_origin
    if not session_id or not plugin.milvus_manager or not plugin.embedding_provider:
        return ""

    try:
        query_vector = await plugin.embedding_provider.get_embedding(query)
        if not query_vector:
            return ""

        loop = asyncio.get_event_loop()
        from .security_utils import safe_build_milvus_expression
        session_filter = safe_build_milvus_expression("session_id", session_id, "==")

        search_results = await loop.run_in_executor(
            None,
            lambda: plugin.milvus_manager.search(
                collection_name=plugin.collection_name,
                query_vectors=[query_vector],
                vector_field=VECTOR_FIELD_NAME,
                search_params=plugin.search_params,
                limit=1,
                expression=f"memory_id > 0 and {session_filter}",
                output_fields=["memory_id", "content", "create_time", "personality_id", "session_id"],
            ),
        )
        if not search_results or not search_results[0] or len(search_results[0]) == 0:
            return ""

        hit = search_results[0][0]
        entity = hit.entity.to_dict().get("entity", {})
        old_id = entity.get("memory_id")
        old_content = entity.get("content", "")
        old_ctime = entity.get("create_time", int(time.time()))
        old_persona = entity.get("personality_id", "")
        old_session = entity.get("session_id", session_id)

        pure, meta = split_memory_content_meta(old_content)
        if meta.get("resolved"):
            return pure[:80]  # 已经标记过了

        meta["resolved"] = True
        meta["resolved_at"] = int(time.time())
        new_content = pack_memory_content(pure, meta)

        new_vector = await plugin.embedding_provider.get_embedding(pure)
        if not new_vector:
            return ""

        def _reinsert():
            plugin.milvus_manager.insert(
                collection_name=plugin.collection_name,
                data=[{
                    "personality_id": old_persona,
                    "session_id": old_session,
                    "content": new_content,
                    VECTOR_FIELD_NAME: new_vector,
                    "create_time": old_ctime,  # 保留原时间
                }],
            )
            plugin.milvus_manager.delete(
                plugin.collection_name,
                f"memory_id == {old_id}",
            )
            plugin.milvus_manager.flush([plugin.collection_name])

        await loop.run_in_executor(None, _reinsert)
        logger.info(f"[active_memory] resolved: {pure[:60]}")
        return pure[:80]

    except Exception as e:
        logger.error(f"[active_memory] resolve 失败: {e}", exc_info=True)
        return ""


async def recall_memory(plugin: "Mnemosyne", event, query: str, top_k: int = 3) -> list[str]:
    """主动检索记忆，返回最相关的几条正文。"""
    session_id = event.unified_msg_origin
    if not session_id or not plugin.milvus_manager or not plugin.embedding_provider:
        return []

    try:
        query_vector = await plugin.embedding_provider.get_embedding(query)
        if not query_vector:
            return []

        loop = asyncio.get_event_loop()
        from .security_utils import safe_build_milvus_expression
        session_filter = safe_build_milvus_expression("session_id", session_id, "==")

        search_results = await loop.run_in_executor(
            None,
            lambda: plugin.milvus_manager.search(
                collection_name=plugin.collection_name,
                query_vectors=[query_vector],
                vector_field=VECTOR_FIELD_NAME,
                search_params=plugin.search_params,
                limit=max(1, min(top_k, 10)),
                expression=f"memory_id > 0 and {session_filter}",
                output_fields=["content", "create_time"],
            ),
        )
        if not search_results or not search_results[0]:
            return []

        results = []
        from datetime import datetime
        for hit in search_results[0]:
            entity = hit.entity.to_dict().get("entity", {})
            pure, _ = split_memory_content_meta(entity.get("content", ""))
            ts = entity.get("create_time")
            try:
                time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
            except (TypeError, ValueError):
                time_str = ""
            results.append(f"[{time_str}] {pure}" if time_str else pure)
        return results

    except Exception as e:
        logger.error(f"[active_memory] recall 失败: {e}", exc_info=True)
        return []
