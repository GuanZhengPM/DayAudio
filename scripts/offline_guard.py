#!/usr/bin/env python3
"""Run the DayAudio CLI while failing every attempted network connection."""

from __future__ import annotations

import socket
import sys


class NetworkAccessBlocked(RuntimeError):
    pass


def _blocked(*_args, **_kwargs):
    raise NetworkAccessBlocked("network access attempted during offline acceptance")


socket.create_connection = _blocked  # type: ignore[assignment]
socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.socket.connect_ex = _blocked  # type: ignore[method-assign]

from dayaudio.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
