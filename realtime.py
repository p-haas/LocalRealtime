from __future__ import annotations

import asyncio

from src.core.config import config_from_args
from src.orchestrator import RealtimeOrchestrator
from src.ui.terminal_ui import TerminalUI


async def _main() -> None:
    config = config_from_args()
    ui = TerminalUI()
    orchestrator = RealtimeOrchestrator(config, ui)
    await orchestrator.run()


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)


if __name__ == "__main__":
    main()
