from __future__ import annotations

import shutil
import subprocess


class ConsoleNotifier:
    """Audible best-effort notification with a terminal fallback."""

    def announce(self, message: str) -> None:
        print(f"\a[RECAP] {message}", flush=True)
        speaker = shutil.which("spd-say") or shutil.which("espeak")
        if speaker:
            try:
                subprocess.Popen(
                    [speaker, message],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass
