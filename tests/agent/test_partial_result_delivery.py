"""Tests for the bounded user-result fallback (#partial-result-delivery)."""

from agent.partial_result_delivery import (
    format_internal_progress_footer,
    is_internal_progress_only,
)


def test_detects_only_known_internal_progress_boilerplate():
    assert is_internal_progress_only("本轮研究尚未完成。（仍有 9 个执行步骤、0 个适用性决定未完成。）")
    assert not is_internal_progress_only("结论：研究尚未完成，当前不建议据此行动。")
    assert not is_internal_progress_only("正常的简短回答。")


def test_footer_reports_tool_progress_without_inventing_findings():
    footer = format_internal_progress_footer(
        [
            {"role": "tool", "tool_name": "market_scan", "content": "ok"},
            {"role": "tool", "tool_name": "sector_scan", "content": "错误: timeout"},
        ]
    )
    assert "market_scan" in footer
    assert "sector_scan" in footer
    assert "不对市场或个股下结论" not in footer
    assert "完整研究结论" in footer
