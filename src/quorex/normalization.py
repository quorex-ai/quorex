from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

_NOM_ALNUM = re.compile(r"[^a-z0-9]+")

def canonical_form(raw: str) -> str:
    """Minuscule, sans accents, snake_case. Sans lookup de synonymes."""
    s = unicodedata.normalize("NFKD", raw)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("'", " ").replace("’", " ")
    s = _NOM_ALNUM.sub("_", s)
    return s.strip("_")

class Normalizer:
    """canonical_form puis table de synonymes (variante -> canonique)"""
    def __init__(self, synonyms: Mapping[str, str]) -> None:
        self._synonyms = {canonical_form(k): v for k, v in synonyms.items()}

    def normalize(self, raw: str) -> str:
        base = canonical_form(raw)
        return self._synonyms.get(base, base)

    def __len__(self) -> int:
        return len(self._synonyms)

def normalizer_from_yaml(path: str) -> Normalizer:
    """Charge data/synonyms.yaml (format canonique: [variante])"""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    table: dict[str, str] = {}
    for canonical, variants in data.items():
        for v in variants or []:
            table[v] = canonical
    return Normalizer(table)