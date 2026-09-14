"""``python -m regulars serve`` (uvicorn) and ``python -m regulars tick`` (one cron tick).

Both read the system from the environment (``DATABASE_URL``, Stripe/BlueBubbles settings) via
``regulars.wiring.System.from_env``. ``serve`` needs a real ``Transport``; ``tick`` only needs
one because ``BotRuntime.tick`` sends messages through it (deposit requests, drop DMs).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence

import httpx
import uvicorn

from regulars.bot.transport import BlueBubblesTransport, Transport
from regulars.wiring import System


def _bluebubbles_transport() -> Transport:
    base_url = os.environ.get("BLUEBUBBLES_URL", "")
    password = os.environ.get("BLUEBUBBLES_PASSWORD", "")
    if not base_url or not password:
        raise RuntimeError("BLUEBUBBLES_URL and BLUEBUBBLES_PASSWORD must be set")
    return BlueBubblesTransport(base_url, password, httpx.AsyncClient())


def _serve() -> None:
    system = System.from_env(transport=_bluebubbles_transport())
    uvicorn.run(system.app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))  # noqa: S104


async def _tick() -> None:
    system = System.from_env(transport=_bluebubbles_transport())
    backfills = await system.runtime.tick()
    for b in backfills:
        print(f"backfill requested: group={b.group_id} plan={b.plan_id} seats={b.seats}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="regulars")
    parser.add_argument("command", choices=("serve", "tick"))
    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve()
    else:
        asyncio.run(_tick())


if __name__ == "__main__":
    main(sys.argv[1:])
