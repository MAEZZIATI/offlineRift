"""Launch a native overlay; Windows binaries also run through WSL interoperability."""

import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

from rift_common.artifacts import atomic_json, file_digest, read_json


def _windows_local_executable(executable, environment):
    """Stage WSL-hosted Electron packages onto Windows storage before starting children."""
    windows_path = subprocess.run(
        ["wslpath", "-w", str(executable)],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        env=environment,
    ).stdout.strip()
    if not windows_path.startswith("\\\\"):
        return executable
    local_app_data = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[Console]::Write($env:LOCALAPPDATA)",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        env=environment,
    ).stdout.strip()
    if not local_app_data or local_app_data.startswith("\\\\"):
        raise ValueError("The native HUD requires a Windows-local LOCALAPPDATA directory")
    mounted = subprocess.run(
        ["wslpath", "-u", local_app_data],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        env=environment,
    ).stdout.strip()
    if not mounted:
        raise ValueError("Could not resolve Windows LOCALAPPDATA from WSL")
    source = executable.parent
    files = [path for path in sorted(source.rglob("*")) if path.is_file()]
    stamps = [
        [path.relative_to(source).as_posix(), path.stat().st_size, path.stat().st_mtime_ns]
        for path in files
    ]
    cache = Path(mounted) / "OfflineRift" / "HUD"
    source_id = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    index = cache / f"source-{source_id}.json"
    try:
        previous = read_json(index)
    except (OSError, ValueError):
        previous = {}
    if previous.get("files") == stamps:
        digest = previous["sha256"]
    else:
        signature = hashlib.sha256()
        for path in files:
            signature.update(path.relative_to(source).as_posix().encode())
            signature.update(file_digest(path).encode())
        digest = signature.hexdigest()
        atomic_json(index, {"files": stamps, "sha256": digest})
    destination = cache / digest[:16]
    marker = destination / ".complete"
    try:
        complete = read_json(marker).get("sha256") == digest
    except (OSError, ValueError):
        complete = False
    if not complete:
        print(f"Preparing Windows-local HUD: {destination}", flush=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        atomic_json(marker, {"sha256": digest})
    return destination / executable.name


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
            "The desktop package is required to run the HUD."
        )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"RIOT_API_KEY", "ELECTRON_RUN_AS_NODE"}
    }
    if "microsoft" in platform.release().lower() and executable.suffix.lower() == ".exe":
        try:
            executable = _windows_local_executable(executable, environment)
        except subprocess.SubprocessError as error:
            raise RuntimeError(
                "Could not prepare the Windows-local HUD. Check Windows PowerShell "
                "and WSL interoperability."
            ) from error
    return subprocess.Popen(
        [
            str(executable),
            f"--backend=http://127.0.0.1:{port}",
            f"--position={position}",
            "--enable-logging=stderr",
        ],
        env=environment,
        cwd=executable.parent,
    )
