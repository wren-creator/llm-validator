#!/usr/bin/env python3
"""
checksum.py; Model identity and change-detection for audit purposes.

Two strategies depending on model type:

1. LOCAL / OPEN-WEIGHTS (Ollama, HuggingFace local)
   → SHA-256 hash of the actual weight files on disk.
   → Ground truth: if the hash changes, the weights changed.

2. CLOSED API MODELS (OpenAI, Anthropic, Google, etc.)
   → Behavioural fingerprint: send a fixed probe suite at temperature=0,
     hash the concatenated responses.
   → Detects model version changes, silent updates ("silent swaps"),
     and rollouts. Cannot guarantee weight-level identity.

Fingerprints are stored in a local JSON ledger so you can compare
across runs and raise an alert when something changes.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Probe suite for behavioural fingerprinting
# Must be:
#   - Deterministic (closed-form answers)
#   - Diverse (maths, facts, code, logic, format)
#   - Stable (questions unlikely to become ambiguous over time)
# ---------------------------------------------------------------------------
PROBE_SUITE = [
    # Math — single correct answer
    {
        "id": "math_001",
        "prompt": "What is 1337 multiplied by 42? Reply with only the number, nothing else.",
        "system": "You are a calculator. Reply only with the numeric result.",
    },
    {
        "id": "math_002",
        "prompt": "What is the square root of 144? Reply with only the number.",
        "system": "You are a calculator. Reply only with the numeric result.",
    },
    # Factual — stable historical facts
    {
        "id": "fact_001",
        "prompt": "In what year did World War II end? Reply with only the 4-digit year.",
        "system": "Reply with only the answer, no explanation.",
    },
    {
        "id": "fact_002",
        "prompt": "What is the chemical symbol for gold? Reply with only the symbol.",
        "system": "Reply with only the answer, no explanation.",
    },
    # Logic
    {
        "id": "logic_001",
        "prompt": (
            "If all bloops are razzies and all razzies are lazzies, "
            "are all bloops definitely lazzies? Answer only Yes or No."
        ),
        "system": "Reply with only Yes or No.",
    },
    # Format / structure
    {
        "id": "format_001",
        "prompt": "List exactly three primary colours, one per line, lowercase, no punctuation.",
        "system": "Follow the format instructions exactly.",
    },
    # Code
    {
        "id": "code_001",
        "prompt": (
            "Write a Python function named `add` that takes two arguments and returns their sum. "
            "Reply with only the function definition, no explanation or markdown fences."
        ),
        "system": "Reply with only the raw Python code.",
    },
    # Counting
    {
        "id": "count_001",
        "prompt": "How many letters are in the word 'fingerprint'? Reply with only the number.",
        "system": "Reply with only the numeric result.",
    },
]


# ---------------------------------------------------------------------------
# Ledger (local JSON file)
# ---------------------------------------------------------------------------

DEFAULT_LEDGER = Path("checksum_ledger.json")


def _load_ledger(ledger_path: Path) -> dict:
    if ledger_path.exists():
        try:
            return json.loads(ledger_path.read_text())
        except Exception:
            return {}
    return {}


def _save_ledger(ledger_path: Path, data: dict):
    ledger_path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Strategy 1 — Local weight hashing (Ollama / HuggingFace)
# ---------------------------------------------------------------------------

def _find_ollama_model_dir(model_name: str) -> Optional[Path]:
    """
    Ollama stores blobs under ~/.ollama/models/blobs/
    Model manifests under ~/.ollama/models/manifests/
    """
    base = Path.home() / ".ollama" / "models"
    if not base.exists():
        return None
    return base


def hash_ollama_weights(model_name: str) -> Optional[dict]:
    """
    Hash the Ollama blob files associated with a model.
    Returns a dict with per-file hashes and a combined fingerprint.
    """
    base = Path.home() / ".ollama" / "models"
    if not base.exists():
        return None

    # Normalise model name to find its manifest
    # Ollama stores as registry/namespace/name:tag
    name_parts = model_name.replace("ollama/", "").split(":")
    name = name_parts[0]
    tag = name_parts[1] if len(name_parts) > 1 else "latest"

    # Search manifest directories
    manifest_dirs = [
        base / "manifests" / "registry.ollama.ai" / "library" / name,
        base / "manifests" / "registry.ollama.ai" / name,
    ]

    manifest_path = None
    for d in manifest_dirs:
        candidate = d / tag
        if candidate.exists():
            manifest_path = candidate
            break

    if not manifest_path:
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None

    blobs_dir = base / "blobs"
    file_hashes = {}
    combined = hashlib.sha256()

    for layer in manifest.get("layers", []):
        digest = layer.get("digest", "")
        if not digest:
            continue
        # Ollama stores blobs as sha256-<hex>
        blob_path = blobs_dir / digest.replace(":", "-")
        if blob_path.exists():
            h = hashlib.sha256()
            with open(blob_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
                    combined.update(chunk)
            file_hashes[digest] = h.hexdigest()

    if not file_hashes:
        return None

    return {
        "strategy": "weight_hash",
        "files": file_hashes,
        "combined_sha256": combined.hexdigest(),
    }


def hash_huggingface_weights(model_path: str) -> Optional[dict]:
    """
    Hash all .safetensors / .bin weight files in a local HuggingFace model directory.
    model_path: local directory path to the model.
    """
    p = Path(model_path)
    if not p.exists() or not p.is_dir():
        return None

    weight_files = sorted(
        list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    )
    if not weight_files:
        return None

    file_hashes = {}
    combined = hashlib.sha256()

    for wf in weight_files:
        h = hashlib.sha256()
        with open(wf, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
                combined.update(chunk)
        file_hashes[wf.name] = h.hexdigest()

    return {
        "strategy": "weight_hash",
        "files": file_hashes,
        "combined_sha256": combined.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Strategy 2 — Behavioural fingerprint (closed API models)
# ---------------------------------------------------------------------------

def behavioural_fingerprint(model: str, timeout: int = 30) -> dict:
    """
    Run the probe suite against the model at temperature=0 and hash responses.
    Returns the fingerprint dict.
    """
    try:
        import litellm
    except ImportError:
        raise RuntimeError("litellm not installed. Run: pip install litellm")

    probe_results = []
    errors = []

    console.print(f"  [dim]Running {len(PROBE_SUITE)} behavioural probes at temperature=0...[/dim]")

    for probe in PROBE_SUITE:
        messages = []
        if probe.get("system"):
            messages.append({"role": "system", "content": probe["system"]})
        messages.append({"role": "user", "content": probe["prompt"]})

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=256,
                timeout=timeout,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as e:
            text = f"ERROR:{e}"
            errors.append({"probe": probe["id"], "error": str(e)})

        probe_results.append({"probe_id": probe["id"], "response": text})

    # Hash the concatenation of all responses in order
    combined_text = "\n---\n".join(
        f"{r['probe_id']}:{r['response']}" for r in probe_results
    )
    fingerprint = hashlib.sha256(combined_text.encode()).hexdigest()

    return {
        "strategy": "behavioural_fingerprint",
        "fingerprint_sha256": fingerprint,
        "probe_responses": probe_results,
        "errors": errors,
        "probe_count": len(PROBE_SUITE),
        "error_count": len(errors),
    }


# ---------------------------------------------------------------------------
# Main entry point — auto-selects strategy
# ---------------------------------------------------------------------------

def compute_checksum(
    model: str,
    ledger_path: Path = DEFAULT_LEDGER,
    hf_path: Optional[str] = None,
) -> dict:
    """
    Compute a checksum for the given model, save to ledger, and return result.
    Auto-selects weight hashing for local models, behavioural fingerprint for APIs.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    is_ollama = model.startswith("ollama/")
    is_local_hf = hf_path is not None

    result = {
        "model": model,
        "timestamp": timestamp,
        "strategy": None,
        "checksum": None,
        "detail": {},
        "changed": None,
        "previous_checksum": None,
    }

    # --- Choose strategy ---
    if is_local_hf:
        console.print(f"  [dim]Strategy:[/dim] weight hash (HuggingFace local: {hf_path})")
        detail = hash_huggingface_weights(hf_path)
        if detail:
            result["strategy"] = "weight_hash"
            result["checksum"] = detail["combined_sha256"]
            result["detail"] = detail
        else:
            result["strategy"] = "weight_hash"
            result["detail"] = {"error": f"No weight files found at {hf_path}"}

    elif is_ollama:
        console.print(f"  [dim]Strategy:[/dim] weight hash (Ollama blobs)")
        detail = hash_ollama_weights(model)
        if detail:
            result["strategy"] = "weight_hash"
            result["checksum"] = detail["combined_sha256"]
            result["detail"] = detail
        else:
            # Ollama model not pulled yet or blob layout unexpected — fall back
            console.print(f"  [yellow]Could not find Ollama blobs — falling back to behavioural fingerprint.[/yellow]")
            detail = behavioural_fingerprint(model)
            result["strategy"] = "behavioural_fingerprint"
            result["checksum"] = detail["fingerprint_sha256"]
            result["detail"] = detail

    else:
        console.print(f"  [dim]Strategy:[/dim] behavioural fingerprint (closed API model)")
        detail = behavioural_fingerprint(model)
        result["strategy"] = "behavioural_fingerprint"
        result["checksum"] = detail["fingerprint_sha256"]
        result["detail"] = detail

    # --- Compare with ledger ---
    ledger = _load_ledger(ledger_path)
    model_history = ledger.get(model, [])

    if model_history:
        prev = model_history[-1]
        result["previous_checksum"] = prev.get("checksum")
        result["changed"] = result["checksum"] != prev.get("checksum")
    else:
        result["changed"] = False  # first run, nothing to compare

    # --- Save to ledger ---
    model_history.append({
        "timestamp": timestamp,
        "checksum": result["checksum"],
        "strategy": result["strategy"],
    })
    ledger[model] = model_history
    _save_ledger(ledger_path, ledger)

    return result


# ---------------------------------------------------------------------------
# Ledger inspection helpers
# ---------------------------------------------------------------------------

def get_history(model: str, ledger_path: Path = DEFAULT_LEDGER) -> list:
    ledger = _load_ledger(ledger_path)
    return ledger.get(model, [])


def get_all_history(ledger_path: Path = DEFAULT_LEDGER) -> dict:
    return _load_ledger(ledger_path)
if __name__ == "__main__":
