"""Prompt formatting helpers built on top of Hugging Face tokenizers."""

from __future__ import annotations

from dataclasses import dataclass

from tinycodescaling.benchmarks.prompt_templates import render_user_prompt
from tinycodescaling.utils.hashing import sha256_text


@dataclass
class ChatPromptFormatter:
    """Apply the benchmark prompt template and optional model chat template."""
    model_name: str
    system_prompt: str
    use_chat_template: bool = True
    revision: str | None = None

    def __post_init__(self) -> None:
        """Load the tokenizer once so prompt formatting stays consistent per run."""
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for chat-template prompt formatting."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
        )

    def format_problem_prompt(self, problem_prompt: str) -> str:
        """Convert one benchmark problem into the exact string sent to the model."""
        user_prompt = render_user_prompt(problem_prompt)
        return self.format_user_text(user_prompt)

    def format_user_text(self, user_text: str) -> str:
        """Wrap arbitrary user text in the same chat-template path as benchmark prompts."""
        if not self.use_chat_template:
            return user_text

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @property
    def prompt_template_hash(self) -> str:
        """Hash the effective prompt formatting configuration for reproducibility."""
        chat_template = getattr(self.tokenizer, "chat_template", "") or ""
        payload = "\n".join(
            [
                f"model={self.model_name}",
                f"revision={self.revision or ''}",
                f"system={self.system_prompt}",
                f"use_chat_template={self.use_chat_template}",
                chat_template,
            ]
        )
        return f"sha256:{sha256_text(payload)}"

    @property
    def model_revision(self) -> str | None:
        """Expose the resolved tokenizer revision or commit hash when available."""
        init_kwargs = getattr(self.tokenizer, "init_kwargs", {}) or {}
        return init_kwargs.get("_commit_hash")
