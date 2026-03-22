#!/usr/bin/env python3
"""Detailed logger; records per-test results and writes to JSON."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from schemas.loader import TestCase


class DetailedLogger:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.started_at = datetime.now(timezone.utc).isoformat()

    def record(
        self,
        test: TestCase,
        response: Union[str, List[str]],
        results: List[Dict[str, Any]],
        elapsed: float,
        error: Optional[str],
    ):
        all_passed = all(r["passed"] for r in results)
        self.entries.append({
            "test_id": test.id,
            "description": test.description,
            "tags": test.tags,
            "prompt": test.prompt,
            "response": response,
            "elapsed_seconds": round(elapsed, 3),
            "error": error,
            "passed": all_passed,
            "evaluator_results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def write(self, path: Path):
        passed = sum(1 for e in self.entries if e["passed"])
        failed = len(self.entries) - passed

        report = {
            "meta": {
                "started_at": self.started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "total": len(self.entries),
                "passed": passed,
                "failed": failed,
            },
            "tests": self.entries,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
  if __name__ == "__main__":    
