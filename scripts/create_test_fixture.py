#!/usr/bin/env python3
"""Create deterministic, non-speech WAV fixtures for local/core tests."""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rate = 16_000
    frames = []
    for index in range(round(args.seconds * rate)):
        second = index / rate
        if second < 1 or 3 <= second < 4:
            value = 0
        else:
            value = round(4000 * math.sin(2 * math.pi * (440 if second < 3 else 660) * second))
        frames.append(struct.pack("<h", value))
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"".join(frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
