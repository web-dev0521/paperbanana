"""Tests for planner agent formatting behavior."""

from __future__ import annotations

from paperbanana.agents.planner import PlannerAgent
from paperbanana.core.types import ReferenceExample


class _MockVLM:
    name = "mock-vlm"
    model_name = "mock-model"

    async def generate(self, *args, **kwargs):
        return "ok"


def test_format_examples_includes_structure_hints():
    agent = PlannerAgent(_MockVLM())
    text = agent._format_examples(
        [
            ReferenceExample(
                id="ref_001",
                source_context="context",
                caption="caption",
                image_path="",
                structure_hints={"nodes": ["A"], "edges": ["A->B"]},
            )
        ]
    )

    assert "Structure Hints" in text
    assert "nodes" in text
