"""Local inference server the game connects to.

Protocol: newline-delimited JSON over TCP (see docs/protocol.md).
The model calls are stubs until the classifier and buyer GPT are trained.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

PROTOCOL_VERSION = 1


def handle_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Map one request to one response. Pure function, so it is easy to test."""
    kind = msg.get("type")

    if kind == "ping":
        return {"type": "pong", "protocol": PROTOCOL_VERSION}

    if kind == "classify":
        strokes = msg.get("strokes")
        if not isinstance(strokes, list) or not strokes:
            return _error("classify needs a non-empty 'strokes' list")
        # TODO: run the sketch classifier.
        return {"type": "classify_result", "top_k": [{"label": "apple", "prob": 1.0}]}

    if kind == "chat":
        if not isinstance(msg.get("text"), str):
            return _error("chat needs a 'text' string")
        # TODO: run the buyer GPT.
        return {"type": "chat_reply", "text": "How old is it?", "offer": None}

    return _error(f"unknown message type: {kind!r}")


def _error(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


async def _serve_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while line := await reader.readline():
            try:
                msg = json.loads(line)
                reply = handle_message(msg) if isinstance(msg, dict) else _error("expected an object")
            except json.JSONDecodeError as e:
                reply = _error(f"invalid JSON: {e.msg}")
            writer.write((json.dumps(reply, ensure_ascii=False) + "\n").encode())
            await writer.drain()
    finally:
        writer.close()


async def serve(host: str, port: int) -> None:
    server = await asyncio.start_server(_serve_client, host, port)
    print(f"sellanything-ml listening on {host}:{port}")
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
