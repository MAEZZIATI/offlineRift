"""Keep domain free of I/O and reject outward production dependencies."""

import ast
from pathlib import Path


def test_domain_has_no_io_dependencies() -> None:
    root = Path("packages/domain/src/offlinerift/domain")
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name in {"typing", "pydantic"} for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module in {"typing", "pydantic"}
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"open", "eval", "exec", "__import__"}


def test_package_dependencies_point_inward() -> None:
    allowed = {"training": {"domain", "storage"}}
    for path in Path("packages").glob("*/src/offlinerift/**/*.py"):
        package = path.parts[1]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("offlinerift."):
                dependency = (node.module or "").split(".")[1]
                assert dependency == package or dependency in allowed.get(package, {"domain"})
