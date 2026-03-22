#!/usr/bin/env python3

"""Schema definitions and loader for test suite YAML/JSON files."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Evaluator configs
# ---------------------------------------------------------------------------

class AccuracyEvaluator(BaseModel):
    type: str = "accuracy"
    contains: Optional[List[str]] = None
    not_contains: Optional[List[str]] = None
    exact_match: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class SafetyEvaluator(BaseModel):
    type: str = "safety"
    expect_refusal: bool = False
    forbidden_patterns: Optional[List[str]] = None


class ConsistencyEvaluator(BaseModel):
    type: str = "consistency"
    runs: int = Field(default=3, ge=2, le=20)
    max_variance: float = Field(default=0.3, ge=0.0, le=1.0)


EvaluatorConfig = Union[AccuracyEvaluator, SafetyEvaluator, ConsistencyEvaluator]


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    id: str
    prompt: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    evaluators: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    def parsed_evaluators(self) -> List[EvaluatorConfig]:
        result = []
        for ev in self.evaluators:
            t = ev.get("type")
            if t == "accuracy":
                result.append(AccuracyEvaluator(**ev))
            elif t == "safety":
                result.append(SafetyEvaluator(**ev))
            elif t == "consistency":
                result.append(ConsistencyEvaluator(**ev))
            else:
                raise ValueError(f"Unknown evaluator type: '{t}'")
        return result


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

class TestSuite(BaseModel):
    suite: str
    model: str
    description: Optional[str] = None
    default_system_prompt: Optional[str] = None
    tests: List[TestCase]

    @field_validator("tests")
    @classmethod
    def tests_not_empty(cls, v):
        if not v:
            raise ValueError("Test suite must contain at least one test.")
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Test IDs must be unique within a suite.")
        return v


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_and_validate_suite(path: Path) -> TestSuite:
    """Load and validate a YAML or JSON test suite file."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    elif path.suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .yaml, .yml, or .json")

    return TestSuite(**raw)
if __name__ == "__main__":
