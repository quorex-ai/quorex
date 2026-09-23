"""S12. Normalisation des attributs (tests/acceptance/SCENARIOS.md)."""
from pathlib import Path

import pytest

from quorex.normalization import Normalizer, canonical_form, normalizer_from_yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def norm() -> Normalizer:
    return normalizer_from_yaml(str(ROOT / "data" / "synonyms.yaml"))


@pytest.mark.parametrize("raw, expected", [
    ("Couleur préférée", "couleur_preferee"),
    ("couleur  préférée ", "couleur_preferee"),
    ("couleur-préférée", "couleur_preferee"),
    ("COULEUR_PREFEREE", "couleur_preferee"),
    ("couleur favorite", "couleur_preferee"),
    ("favorite color", "couleur_preferee"),
    ("Ville de résidence", "ville_de_residence"),
    ("l'âge", "age"),
    ("âge", "age"),
    ("mon manager", "mon_manager"),
])
def test_s12_table(norm: Normalizer, raw: str, expected: str) -> None:
    assert norm.normalize(raw) == expected


def test_s12_attributs_distincts_restent_distincts(norm: Normalizer) -> None:
    assert norm.normalize("couleur préférée") != norm.normalize("couleur de la voiture")


def test_canonical_form_sans_synonymes() -> None:
    assert canonical_form("  Élève -- Modèle ! ") == "eleve_modele"
    assert canonical_form("x") == "x"


def test_table_yaml_sans_doublon() -> None:
    import yaml

    data = yaml.safe_load((ROOT / "data" / "synonyms.yaml").read_text(encoding="utf-8"))
    seen: dict[str, str] = {}
    for canonical, variants in data.items():
        for v in variants or []:
            assert v not in seen, f"'{v}' mappé vers {seen[v]} et {canonical}"
            seen[v] = canonical