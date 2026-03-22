#!/usr/bin/env python3
"""
Model registry; curated metadata for known LLMs.

Sources:
  - Official provider announcements and technical reports
  - HuggingFace model cards (open models)
  - Verified community research (estimates marked explicitly)

Fields per entry:
  creator         : Organisation that trained the model
  family          : Model family / series name
  parameters      : Parameter count string (or None if undisclosed)
  parameters_note : Confidence level / source note
  release_year    : Year of public release
  open_weights    : True if weights are publicly downloadable
  architecture    : Known architecture details
  license         : License (open models) or "Proprietary"
  hf_url          : HuggingFace model card URL (open models)
  notes           : Any extra context worth surfacing
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelMeta:
    creator: str
    family: str
    parameters: Optional[str]
    parameters_note: str
    release_year: int
    open_weights: bool
    architecture: str
    license: str
    hf_url: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry: keyed by the LiteLLM model string (or prefix)
# ---------------------------------------------------------------------------
REGISTRY: dict[str, ModelMeta] = {

    # ── OpenAI ────────────────────────────────────────────────────────────
    "gpt-4o": ModelMeta(
        creator="OpenAI",
        family="GPT-4o",
        parameters=None,
        parameters_note="Not disclosed by OpenAI",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (multimodal: text, image, audio)",
        license="Proprietary",
        notes="Omni model; natively handles text, vision, and audio in one model.",
    ),
    "gpt-4o-mini": ModelMeta(
        creator="OpenAI",
        family="GPT-4o",
        parameters="~8B (community estimate)",
        parameters_note="Estimated: OpenAI has not confirmed",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (multimodal)",
        license="Proprietary",
        notes="Small/fast variant of GPT-4o; cost-optimised.",
    ),
    "gpt-4-turbo": ModelMeta(
        creator="OpenAI",
        family="GPT-4",
        parameters=None,
        parameters_note="Not disclosed by OpenAI",
        release_year=2023,
        open_weights=False,
        architecture="Transformer",
        license="Proprietary",
        notes="128k context window; knowledge cutoff April 2023.",
    ),
    "gpt-3.5-turbo": ModelMeta(
        creator="OpenAI",
        family="GPT-3.5",
        parameters="~175B (community estimate)",
        parameters_note="Estimated based on GPT-3 lineage; not confirmed",
        release_year=2022,
        open_weights=False,
        architecture="Transformer (decoder-only)",
        license="Proprietary",
    ),
    "o1": ModelMeta(
        creator="OpenAI",
        family="OpenAI o-series",
        parameters=None,
        parameters_note="Not disclosed by OpenAI",
        release_year=2024,
        open_weights=False,
        architecture="Transformer with extended chain-of-thought reasoning",
        license="Proprietary",
        notes="Trained with reinforcement learning to reason before answering.",
    ),
    "o3-mini": ModelMeta(
        creator="OpenAI",
        family="OpenAI o-series",
        parameters=None,
        parameters_note="Not disclosed by OpenAI",
        release_year=2025,
        open_weights=False,
        architecture="Transformer with chain-of-thought reasoning",
        license="Proprietary",
        notes="Small reasoning model; faster and cheaper than o1.",
    ),

    # ── Anthropic ─────────────────────────────────────────────────────────
    "claude-opus-4-6": ModelMeta(
        creator="Anthropic",
        family="Claude 4",
        parameters=None,
        parameters_note="Not disclosed by Anthropic",
        release_year=2025,
        open_weights=False,
        architecture="Transformer (Constitutional AI trained)",
        license="Proprietary",
        notes="Most capable Claude 4 model.",
    ),
    "claude-sonnet-4-6": ModelMeta(
        creator="Anthropic",
        family="Claude 4",
        parameters=None,
        parameters_note="Not disclosed by Anthropic",
        release_year=2025,
        open_weights=False,
        architecture="Transformer (Constitutional AI trained)",
        license="Proprietary",
        notes="Balanced performance and speed in the Claude 4 family.",
    ),
    "claude-haiku-4-5-20251001": ModelMeta(
        creator="Anthropic",
        family="Claude Haiku",
        parameters=None,
        parameters_note="Not disclosed by Anthropic",
        release_year=2025,
        open_weights=False,
        architecture="Transformer (Constitutional AI trained)",
        license="Proprietary",
        notes="Fastest and most compact Claude model.",
    ),
    "claude-3-5-sonnet-20241022": ModelMeta(
        creator="Anthropic",
        family="Claude 3.5",
        parameters=None,
        parameters_note="Not disclosed by Anthropic",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (Constitutional AI trained)",
        license="Proprietary",
    ),
    "claude-3-opus-20240229": ModelMeta(
        creator="Anthropic",
        family="Claude 3",
        parameters=None,
        parameters_note="Not disclosed by Anthropic",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (Constitutional AI trained)",
        license="Proprietary",
    ),

    # ── Google ────────────────────────────────────────────────────────────
    "gemini/gemini-1.5-pro": ModelMeta(
        creator="Google DeepMind",
        family="Gemini 1.5",
        parameters=None,
        parameters_note="Not disclosed by Google",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (multimodal, MoE suspected)",
        license="Proprietary",
        notes="1M token context window; supports text, image, audio, video.",
    ),
    "gemini/gemini-1.5-flash": ModelMeta(
        creator="Google DeepMind",
        family="Gemini 1.5",
        parameters=None,
        parameters_note="Not disclosed by Google",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (multimodal)",
        license="Proprietary",
        notes="Fast, lightweight variant of Gemini 1.5 Pro.",
    ),
    "gemini/gemini-2.0-flash": ModelMeta(
        creator="Google DeepMind",
        family="Gemini 2.0",
        parameters=None,
        parameters_note="Not disclosed by Google",
        release_year=2025,
        open_weights=False,
        architecture="Transformer (multimodal)",
        license="Proprietary",
        notes="Next-gen Gemini; improved speed and multimodal capabilities.",
    ),

    # ── Meta (Llama) ──────────────────────────────────────────────────────
    "ollama/llama3": ModelMeta(
        creator="Meta AI",
        family="Llama 3",
        parameters="8B",
        parameters_note="Confirmed by Meta (default 8B variant)",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only, GQA, RoPE)",
        license="Llama 3 Community License",
        hf_url="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "ollama/llama3:70b": ModelMeta(
        creator="Meta AI",
        family="Llama 3",
        parameters="70B",
        parameters_note="Confirmed by Meta",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only, GQA, RoPE)",
        license="Llama 3 Community License",
        hf_url="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    "ollama/llama3.1": ModelMeta(
        creator="Meta AI",
        family="Llama 3.1",
        parameters="8B",
        parameters_note="Confirmed by Meta",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only, 128k context)",
        license="Llama 3.1 Community License",
        hf_url="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    ),
    "ollama/llama3.1:70b": ModelMeta(
        creator="Meta AI",
        family="Llama 3.1",
        parameters="70B",
        parameters_note="Confirmed by Meta",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only, 128k context)",
        license="Llama 3.1 Community License",
        hf_url="https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
    ),
    "ollama/llama3.1:405b": ModelMeta(
        creator="Meta AI",
        family="Llama 3.1",
        parameters="405B",
        parameters_note="Confirmed by Meta",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only, 128k context)",
        license="Llama 3.1 Community License",
        hf_url="https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct",
        notes="Largest open-weights model from Meta; rivals closed frontier models.",
    ),
    "ollama/llama3.2": ModelMeta(
        creator="Meta AI",
        family="Llama 3.2",
        parameters="3B",
        parameters_note="Confirmed by Meta",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (multimodal — text + vision)",
        license="Llama 3.2 Community License",
        hf_url="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
    ),

    # ── Mistral AI ────────────────────────────────────────────────────────
    "ollama/mistral": ModelMeta(
        creator="Mistral AI",
        family="Mistral",
        parameters="7B",
        parameters_note="Confirmed by Mistral AI",
        release_year=2023,
        open_weights=True,
        architecture="Transformer (GQA, sliding window attention)",
        license="Apache 2.0",
        hf_url="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "mistral/mistral-large-latest": ModelMeta(
        creator="Mistral AI",
        family="Mistral Large",
        parameters=None,
        parameters_note="Not disclosed by Mistral AI",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (MoE suspected)",
        license="Proprietary (via API)",
        notes="Mistral's flagship closed model.",
    ),
    "mistral/mistral-small-latest": ModelMeta(
        creator="Mistral AI",
        family="Mistral Small",
        parameters=None,
        parameters_note="Not disclosed by Mistral AI",
        release_year=2024,
        open_weights=False,
        architecture="Transformer",
        license="Proprietary (via API)",
    ),
    "ollama/mixtral": ModelMeta(
        creator="Mistral AI",
        family="Mixtral",
        parameters="46.7B total / 12.9B active (MoE 8x7B)",
        parameters_note="Confirmed by Mistral AI — Mixture of Experts",
        release_year=2023,
        open_weights=True,
        architecture="Sparse Mixture of Experts (8 experts, 2 active per token)",
        license="Apache 2.0",
        hf_url="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
        notes="MoE architecture: 8 expert networks, 2 activated per token.",
    ),
    "ollama/mixtral:8x22b": ModelMeta(
        creator="Mistral AI",
        family="Mixtral",
        parameters="141B total / 39B active (MoE 8x22B)",
        parameters_note="Confirmed by Mistral AI",
        release_year=2024,
        open_weights=True,
        architecture="Sparse Mixture of Experts (8 experts, 2 active per token)",
        license="Apache 2.0",
        hf_url="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1",
    ),

    # ── Deepseek ──────────────────────────────────────────────────────────
    "ollama/deepseek-r1": ModelMeta(
        creator="DeepSeek AI",
        family="DeepSeek-R1",
        parameters="7B (default Ollama variant)",
        parameters_note="Confirmed — multiple sizes available (1.5B–671B)",
        release_year=2025,
        open_weights=True,
        architecture="Transformer with chain-of-thought reasoning (MoE at larger sizes)",
        license="MIT",
        hf_url="https://huggingface.co/deepseek-ai/DeepSeek-R1",
        notes="Reasoning model trained with RL; rivals o1 on benchmarks. Full 671B MoE available.",
    ),
    "ollama/deepseek-r1:671b": ModelMeta(
        creator="DeepSeek AI",
        family="DeepSeek-R1",
        parameters="671B total (MoE)",
        parameters_note="Confirmed by DeepSeek",
        release_year=2025,
        open_weights=True,
        architecture="Mixture of Experts Transformer",
        license="MIT",
        hf_url="https://huggingface.co/deepseek-ai/DeepSeek-R1",
    ),
    "deepseek/deepseek-chat": ModelMeta(
        creator="DeepSeek AI",
        family="DeepSeek-V3",
        parameters="671B total / ~37B active (MoE)",
        parameters_note="Confirmed by DeepSeek in technical report",
        release_year=2024,
        open_weights=True,
        architecture="Mixture of Experts Transformer (Multi-head Latent Attention)",
        license="MIT",
        hf_url="https://huggingface.co/deepseek-ai/DeepSeek-V3",
        notes="Trained on 14.8T tokens. MoE means only ~37B params active per forward pass.",
    ),

    # ── Google (Gemma) ────────────────────────────────────────────────────
    "ollama/gemma2": ModelMeta(
        creator="Google DeepMind",
        family="Gemma 2",
        parameters="9B",
        parameters_note="Confirmed by Google",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only, grouped-query attention)",
        license="Gemma Terms of Use",
        hf_url="https://huggingface.co/google/gemma-2-9b-it",
    ),
    "ollama/gemma2:27b": ModelMeta(
        creator="Google DeepMind",
        family="Gemma 2",
        parameters="27B",
        parameters_note="Confirmed by Google",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only)",
        license="Gemma Terms of Use",
        hf_url="https://huggingface.co/google/gemma-2-27b-it",
    ),

    # ── Qwen (Alibaba) ────────────────────────────────────────────────────
    "ollama/qwen2.5": ModelMeta(
        creator="Alibaba Cloud (Qwen Team)",
        family="Qwen 2.5",
        parameters="7B (default Ollama variant)",
        parameters_note="Confirmed — range from 0.5B to 72B",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only, GQA, RoPE)",
        license="Apache 2.0 (≤72B); Qwen License (larger)",
        hf_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    ),
    "ollama/qwen2.5:72b": ModelMeta(
        creator="Alibaba Cloud (Qwen Team)",
        family="Qwen 2.5",
        parameters="72B",
        parameters_note="Confirmed by Alibaba",
        release_year=2024,
        open_weights=True,
        architecture="Transformer (decoder-only)",
        license="Qwen License",
        hf_url="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
    ),

    # ── xAI ───────────────────────────────────────────────────────────────
    "xai/grok-2": ModelMeta(
        creator="xAI",
        family="Grok 2",
        parameters=None,
        parameters_note="Not disclosed by xAI",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (MoE suspected)",
        license="Proprietary",
        notes="Developed by Elon Musk's xAI; has real-time X (Twitter) data access.",
    ),

    # ── Cohere ────────────────────────────────────────────────────────────
    "cohere/command-r-plus": ModelMeta(
        creator="Cohere",
        family="Command R",
        parameters="104B",
        parameters_note="Confirmed by Cohere",
        release_year=2024,
        open_weights=False,
        architecture="Transformer (optimised for RAG and tool use)",
        license="Proprietary",
        notes="Designed for enterprise RAG pipelines.",
    ),
    "cohere/command-r": ModelMeta(
        creator="Cohere",
        family="Command R",
        parameters="35B",
        parameters_note="Confirmed by Cohere",
        release_year=2024,
        open_weights=False,
        architecture="Transformer",
        license="Proprietary",
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def lookup(model: str) -> Optional[ModelMeta]:
    """
    Look up metadata for a model string.
    Tries exact match first, then prefix match for versioned/tagged names.
    """
    if model in REGISTRY:
        return REGISTRY[model]

    # Prefix match — e.g. "gpt-4o-2024-11-20" → "gpt-4o"
    for key in REGISTRY:
        if model.startswith(key) or key.startswith(model.split(":")[0]):
            return REGISTRY[key]

    return None


def all_models() -> list[tuple[str, ModelMeta]]:
    return sorted(REGISTRY.items(), key=lambda x: (x[1].creator, x[0]))
if __name__ == "__main__":
