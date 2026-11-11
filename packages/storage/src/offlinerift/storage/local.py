"""Local atomic JSON artifact adapter, restricted to direct children."""

from pathlib import Path
from tempfile import NamedTemporaryFile


class LocalArtifactStore:
    """A per-run store; contains no credentials or identity-map support."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Artifact name must be a direct child")
        path = self.root / name
        if path.is_symlink():
            raise ValueError("Artifact must not be a symlink")
        return path

    def write_json(self, name: str, payload: str) -> None:
        path = self._path(name)
        self.root.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root, delete=False) as out:
            temporary = Path(out.name)
            out.write(payload)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def read_json(self, name: str) -> str:
        return self._path(name).read_text(encoding="utf-8")
