"""Keep application/research dependencies out of the common contract and runtime."""

import ast
from pathlib import Path


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_dependency_direction():
    root = Path(__file__).parents[1]
    for path in (root / "common/src").rglob("*.py"):
        assert not any(
            name.startswith(("offline_rift", "rift_research")) for name in imports(path)
        ), path
    for path in (root / "production/src").rglob("*.py"):
        assert not any(name.startswith("rift_research") for name in imports(path)), path
    for path in (root / "production/src/offline_rift/inference").rglob("*.py"):
        assert not any(name.startswith("offline_rift.training") for name in imports(path)), path
