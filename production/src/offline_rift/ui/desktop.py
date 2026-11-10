"""Launch a native overlay; Windows binaries also run through WSL interoperability."""

import os
import platform
import subprocess
from pathlib import Path


def launch_desktop(port, *, executable=None, position="top-right"):
    if position not in {"top-right", "top-left"}:
        raise ValueError("Invalid HUD position")
    if executable is None:
        repository = Path(__file__).resolve().parents[4]
        if os.name == "nt" or "microsoft" in platform.release().lower():
            executable = (
                repository
                / "production/frontend/macro-hud/release/OfflineRiftHUD-win32-x64/OfflineRiftHUD.exe"
            )
        else:
            executable = (
                repository
                / "production/frontend/macro-hud/release/OfflineRiftHUD-linux-x64/OfflineRiftHUD"
            )
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise ValueError(
            "Native HUD is not packaged. Run npm run package:win in "
            "production/frontend/macro-hud, or use --desktop PATH. "
            "Use --serve-only for the browser HUD."
        )
    environment = {key: value for key, value in os.environ.items() if key != "RIOT_API_KEY"}
    return subprocess.Popen(
        [str(executable), "--backend", f"http://127.0.0.1:{port}", "--position", position],
        env=environment,
    )
