"""Repeatable CPU CLI smoke test in a fresh temporary output directory."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Interrupt and resume the exact production entry point."""
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.json"
    with TemporaryDirectory(prefix="offlinerift-smoke-") as directory:
        output = Path(directory)
        command = [
            sys.executable,
            "-m",
            "offlinerift.training.run",
            "--config",
            config,
            "--output",
            str(output),
        ]
        subprocess.run([*command, "--stop-after", "4"], check=True)
        subprocess.run([*command, "--resume", str(output / "checkpoint.json")], check=True)


if __name__ == "__main__":
    main()
