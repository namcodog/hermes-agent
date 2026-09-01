"""Small, deterministic safety net for research replies without a result.

This is deliberately not a content judge: it recognizes only the known
internal-progress boilerplate that is not useful as a user-facing answer.  The
research Profile remains responsible for turning successful tool evidence into
actual findings.  This helper merely makes a malformed terminal reply disclose
what ran, what failed, and what can be done next.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


_INTERNAL_PROGRESS = (
    "本轮研究尚未完成",
    "研究尚未完成",
    "执行步骤",
    "适用性决定",
)
_USER_FACING_FINDING = ("结论", "发现", "依据", "建议", "推荐", "未取得")


def is_internal_progress_only(response: object) -> bool:
    """Identify the narrow known case of a status message posing as a reply."""
    if not isinstance(response, str):
        return False
    text = response.strip()
    if not text or len(text) > 280 or any(word in text for word in _USER_FACING_FINDING):
        return False
    return any(marker in text for marker in _INTERNAL_PROGRESS)


def _tool_statuses(messages: Iterable[object]) -> tuple[list[str], list[str]]:
    succeeded: list[str] = []
    failed: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        name = str(message.get("tool_name") or message.get("name") or "未命名工具")
        content = str(message.get("content") or "")
        is_failure = bool(re.search(r"\b(error|failed|exception)\b|错误|失败", content, re.I))
        bucket = failed if is_failure else succeeded
        if name not in bucket:
            bucket.append(name)
    return succeeded[:6], failed[:6]


def format_internal_progress_footer(messages: Iterable[object]) -> str:
    """Give a truthful recovery note without inventing research findings."""
    succeeded, failed = _tool_statuses(messages)
    lines = ["【当前可交付进展】这不是完整研究结论，不能只以内部步骤状态交付。"]
    if succeeded:
        lines.append("本轮已完成取数：" + "、".join(succeeded) + "。")
    else:
        lines.append("本轮没有可确认的已完成取数，因此不对市场或个股下结论。")
    if failed:
        lines.append("未完成的取数：" + "、".join(failed) + "（工具返回失败）。")
    else:
        lines.append("未完成的部分未形成可核验研究结论；请继续时优先把已取到的数据整理为结论、依据、限制和下一步。")
    return "\n".join(lines)
