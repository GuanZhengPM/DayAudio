#!/usr/bin/env python3
"""Deterministic JSONL ASR fixture used by the public end-to-end test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    events = (
        {
            "kind": "commit",
            "segment_id": "fixture-segment",
            "revision": 1,
            "start": 0.2,
            "end": 1.2,
            "text": "这是公开生成的端到端测试转录。",
            "confidence": 0.99,
            "language": "zh",
        },
        {"kind": "end"},
    )
    for event in events:
        print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
