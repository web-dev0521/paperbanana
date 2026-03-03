"""Planner Agent: Generates detailed textual descriptions from source context."""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType, ReferenceExample
from paperbanana.core.utils import load_image
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class PlannerAgent(BaseAgent):
    """Generates a comprehensive textual description for diagram generation.

    Uses in-context learning from retrieved reference examples (including
    their images) to create a detailed description that the Visualizer
    can render. Matches paper equation 4: P = VLM_plan(S, C, {(S_i, C_i, I_i)}).
    """

    def __init__(self, vlm_provider: VLMProvider, prompt_dir: str = "prompts"):
        super().__init__(vlm_provider, prompt_dir)

    @property
    def agent_name(self) -> str:
        return "planner"

    async def run(
        self,
        source_context: str,
        caption: str,
        examples: list[ReferenceExample],
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
        supported_ratios: list[str] | None = None,
    ) -> tuple[str, str | None]:
        """Generate a detailed textual description of the target diagram.

        Args:
            source_context: Methodology text from the paper.
            caption: Communicative intent / figure caption.
            examples: Retrieved reference examples for in-context learning.
            diagram_type: Type of diagram being generated.
            supported_ratios: Aspect ratios the image provider supports.

        Returns:
            Tuple of (description, recommended_ratio).
            recommended_ratio is a string like '16:9' or None if not provided.
        """
        # Format examples for in-context learning
        examples_text = self._format_examples(examples)

        # Load reference images for visual in-context learning
        example_images = self._load_example_images(examples)

        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        # Inject supported ratios into the prompt template
        ratios_str = ", ".join(supported_ratios) if supported_ratios else "1:1, 16:9"
        prompt = self.format_prompt(
            template,
            source_context=source_context,
            caption=caption,
            examples=examples_text,
            supported_ratios=ratios_str,
        )

        logger.info(
            "Running planner agent",
            num_examples=len(examples),
            num_images=len(example_images),
            context_length=len(source_context),
        )

        raw_output = await self.vlm.generate(
            prompt=prompt,
            images=example_images if example_images else None,
            temperature=0.7,
            max_tokens=4096,
        )

        description, ratio = self._parse_ratio(raw_output)
        logger.info(
            "Planner generated description",
            length=len(description),
            recommended_ratio=ratio,
        )
        return description, ratio

    def _format_examples(self, examples: list[ReferenceExample]) -> str:
        """Format reference examples for the planner prompt.

        Each example includes its text metadata and a reference to the
        corresponding image (passed separately as visual input).
        """
        if not examples:
            return "(No reference examples available. Generate based on source context alone.)"

        lines = []
        img_index = 0
        for i, ex in enumerate(examples, 1):
            has_image = self._has_valid_image(ex)
            image_ref = ""
            if has_image:
                img_index += 1
                image_ref = f"\n**Diagram**: [See reference image {img_index} above]"

            ratio_info = ""
            if ex.aspect_ratio:
                ratio_info = f"\n**Aspect Ratio**: {ex.aspect_ratio:.2f}"

            structure_info = ""
            if ex.structure_hints:
                hints_text = str(ex.structure_hints)
                structure_info = f"\n**Structure Hints**: {hints_text[:240]}"

            lines.append(
                f"### Example {i}\n"
                f"**Caption**: {ex.caption}\n"
                f"**Source Context**: {ex.source_context[:500]}"
                f"{ratio_info}"
                f"{structure_info}"
                f"{image_ref}\n"
            )
        return "\n".join(lines)

    def _has_valid_image(self, example: ReferenceExample) -> bool:
        """Check if a reference example has a valid image file."""
        if not example.image_path:
            return False
        return Path(example.image_path).exists()

    def _load_example_images(self, examples: list[ReferenceExample]) -> list:
        """Load reference images from disk for in-context learning.

        Returns a list of PIL Image objects for examples that have valid images.
        """
        images = []
        for ex in examples:
            if not self._has_valid_image(ex):
                continue
            try:
                img = load_image(ex.image_path)
                images.append(img)
            except Exception as e:
                logger.warning(
                    "Failed to load reference image",
                    image_path=ex.image_path,
                    error=str(e),
                )
        return images

    _VALID_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"}

    @classmethod
    def _parse_ratio(cls, text: str) -> tuple[str, str | None]:
        """Extract RECOMMENDED_RATIO from planner output and return clean description."""
        match = re.search(r"RECOMMENDED_RATIO:\s*([\d:]+)", text)
        if match:
            ratio = match.group(1).strip()
            if ratio in cls._VALID_RATIOS:
                # Remove the ratio line (and surrounding markdown fences) from description
                clean = re.sub(
                    r"\n*```\n*RECOMMENDED_RATIO:.*?\n*```\n*",
                    "",
                    text,
                ).strip()
                # Also handle case without fences
                clean = re.sub(r"\n*RECOMMENDED_RATIO:.*", "", clean).strip()
                return clean, ratio
            logger.warning("Planner returned invalid ratio", ratio=ratio)
        return text.strip(), None
