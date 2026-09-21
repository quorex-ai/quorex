"""Garde-fous d'architecture (ADR-001, CONTRIBUTING.md).

Aucun module hors de quorex.storage n'importe SQLAlchemy, psycopg ou pgvector.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "quorex"
FORBIDDEN = {"sqlalchemy", "psycopg", "pgvector", "alembic"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_sql_only_in_storage() -> None:
    offenders = []
    for py in SRC.rglob("*.py"):
        if "storage" in py.relative_to(SRC).parts:
            continue
        bad = _imports(py) & FORBIDDEN
        if bad:
            offenders.append(f"{py.relative_to(SRC)} importe {sorted(bad)}")
    assert not offenders, "SQL hors de storage/ :\n" + "\n".join(offenders)