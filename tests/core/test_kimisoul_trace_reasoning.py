"""Trace capture of the model's reasoning (ThinkPart).

The reasoning models (kimi_k2) emit their chain-of-thought as a separate
``ThinkPart`` that ``Message.extract_text()`` (TextPart-only) silently drops.
Before the fix this meant the agent's "why" never reached
``.agent_trace.jsonl`` — starving the orchestrator's narrator and the Opus
judge of reasoning even though we paid for the reasoning tokens. These tests
pin the contract that ``_dump_agent_status`` archives the ThinkPart verbatim
in a dedicated ``reasoning`` field while keeping ``content`` = prose.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from kosong.message import Message, TextPart, ThinkPart, ToolCall

from kimi_cli.soul.kimisoul import KimiSoul


def _bare_soul(work_dir):
    """A KimiSoul stub carrying only what ``_dump_agent_status`` touches."""
    soul = KimiSoul.__new__(KimiSoul)
    soul._status_interval = 5
    soul._status_seq = 0
    soul._status_event_buffer = []
    soul._runtime = SimpleNamespace(session=SimpleNamespace(work_dir=str(work_dir)))
    return soul


def _read_trace(work_dir):
    path = work_dir / ".agent_trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_thinkpart_reasoning_is_captured_in_trace(tmp_path):
    soul = _bare_soul(tmp_path)
    msg = Message(
        role="assistant",
        content=[
            ThinkPart(think="The tile size is too small; widen it to cut decode latency."),
            TextPart(text="Patching the kernel."),
        ],
        tool_calls=[
            ToolCall(
                id="1",
                function=ToolCall.FunctionBody(
                    name="Shell", arguments=json.dumps({"command": "echo hi"})
                ),
            )
        ],
    )
    soul._status_event_buffer.append((5, msg))
    soul._dump_agent_status(5)

    records = _read_trace(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["reasoning"] == "The tile size is too small; widen it to cut decode latency."
    assert rec["content"] == "Patching the kernel."
    assert rec["tool_calls"][0]["name"] == "Shell"


def test_multiple_thinkparts_are_joined(tmp_path):
    soul = _bare_soul(tmp_path)
    msg = Message(
        role="assistant",
        content=[
            ThinkPart(think="First, profile the hot loop."),
            ThinkPart(think="Then fuse the two kernels."),
            TextPart(text="Proceeding."),
        ],
    )
    soul._status_event_buffer.append((2, msg))
    soul._dump_agent_status(2)

    rec = _read_trace(tmp_path)[0]
    assert "First, profile the hot loop." in rec["reasoning"]
    assert "Then fuse the two kernels." in rec["reasoning"]


def test_no_thinkpart_omits_reasoning_field(tmp_path):
    soul = _bare_soul(tmp_path)
    msg = Message(role="assistant", content=[TextPart(text="No thinking here.")])
    soul._status_event_buffer.append((3, msg))
    soul._dump_agent_status(3)

    rec = _read_trace(tmp_path)[0]
    assert "reasoning" not in rec
    assert rec["content"] == "No thinking here."
