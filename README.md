# QUOREX

Mémoire de faits pour agents IA. Un agent envoie ce que dit son utilisateur, QUOREX en extrait des faits structurés, détecte ceux qui contredisent un fait existant, clôt l'ancien et enregistre le nouveau. L'agent ne relit que ce qui est vrai maintenant, et peut demander ce qui était vrai à une date passée.

Chaque fait est bitemporel : quand il est vrai dans la vie de l'utilisateur, et quand le système l'a appris. Rien n'est supprimé, tout est rejouable.

Code propriétaire, voir `LICENSE`.

## Ce que fait la v1

Quatre opérations HTTP :

| Opération | Ce qu'elle fait |
|---|---|
| `remember` | Texte libre (extraction par LLM) ou fait structuré (`attribute`, `value`). Remplace ce qu'il contredit. |
| `recall` | Faits actifs d'un utilisateur. Avec `as_of`, l'état de la mémoire à une date passée. |
| `diff` | Faits ajoutés, remplacés et invalidés entre deux dates. |
| `forget` | Clôt un fait ou un attribut. Ne supprime jamais. |

Pas de SDK, pas de MCP, extraction synchrone. Voir `docs/architecture.md` pour tout le reste.

## Stack

Python 3.12, FastAPI, PostgreSQL 16 + pgvector, Alembic, `sentence-transformers` (multilingual-e5) en local, LLM derrière une interface (Gemini en v1), `uv` pour les dépendances.

## Démarrer

Prérequis : Docker, `uv` (https://docs.astral.sh/uv/).

```bash
git clone <url> && cd quorex
cp .env.example .env            # remplir les variables, voir plus bas
docker compose up -d            # Postgres + pgvector sur localhost:5432
uv sync                         # dépendances, dont le modèle d'embedding au premier lancement
uv run alembic upgrade head     # schéma + table de synonymes
uv run quorex serve             # API sur http://localhost:8000
```

Vérifier : `curl http://localhost:8000/health` doit répondre `{"status":"ok"}`.

Créer un tenant et une clé :

```bash
uv run quorex tenant create --name demo
# affiche l'id du tenant et une clé qx_... à copier maintenant, elle n'est jamais réaffichée
```

Premier fait :

```bash
curl -X POST http://localhost:8000/v1/remember \
  -H "Authorization: Bearer qx_..." \
  -H "Content-Type: application/json" \
  -d '{"user": "alice", "attribute": "couleur préférée", "value": "bleu"}'

curl -X POST http://localhost:8000/v1/recall \
  -H "Authorization: Bearer qx_..." \
  -H "Content-Type: application/json" \
  -d '{"user": "alice"}'
```

## Variables d'environnement

| Variable | Obligatoire | Défaut | Rôle |
|---|---|---|---|
| `QUOREX_DATABASE_URL` | oui | | `postgresql://user:pass@localhost:5432/quorex` |
| `QUOREX_EMBEDDING_MODEL` | non | `intfloat/multilingual-e5-base` | Modèle local |
| `QUOREX_LLM_PROVIDER` | oui | | `gemini` ou `fake` |
| `QUOREX_GEMINI_API_KEY` | si gemini | | Clé API Gemini |
| `QUOREX_GEMINI_MODEL` | si gemini | | Nom du modèle, à vérifier sur la console |
| `QUOREX_SIM_MATCH` | non | `0.85` | Seuil de remplacement direct (niveau b) |
| `QUOREX_SIM_AMBIGUOUS` | non | `0.70` | Seuil bas de la zone d'arbitrage LLM (niveau c) |
| `QUOREX_MIN_CONFIDENCE` | non | `0.5` | Sous ce seuil, un fait extrait est rejeté |
| `QUOREX_LOG_LEVEL` | non | `INFO` | |

Le service refuse de démarrer si une variable obligatoire manque ou si la dimension du modèle d'embedding ne correspond pas à celle enregistrée en base.

## Tests

```bash
uv run pytest                              # unitaires + acceptation avec le FakeLLMProvider
uv run pytest tests/acceptance -v          # les scénarios de tests/acceptance/SCENARIOS.md
uv run pytest -m gemini                    # scénarios rejoués contre le vrai LLM, nécessite la clé
```

Les scénarios d'acceptation sont la définition de « terminé » pour chaque jalon. Un jalon n'est pas fermé tant que ses scénarios ne passent pas.

## Mesurer les seuils

```bash
uv run python scripts/measure_thresholds.py data/attribute_pairs.csv
```

Produit précision et rappel du niveau (b) pour chaque seuil et la part de paires en zone (c). Résultats à consigner dans `docs/decisions/ADR-005-seuils-mesures.md`. À rejouer à chaque changement de modèle d'embedding.

## Structure

```
src/quorex/
  api/            routes FastAPI, auth, grille d'erreurs
  domain/         modèle de fait, interfaces FactStore / LLMProvider / EmbeddingProvider
  extraction/     prompt, schéma, parsing de la sortie LLM
  contradiction/  niveaux (a) exact, (b) embedding, (c) arbitrage
  storage/        seule couche qui connaît SQL ; PostgresFactStore
  embeddings/     sentence-transformers
  llm/            gemini, fake
alembic/          migrations
data/             synonyms.yaml, attribute_pairs.csv
scripts/          measure_thresholds.py, reembed
tests/
  unit/
  acceptance/     SCENARIOS.md et leur implémentation
docs/
  architecture.md
  extraction-examples.md
  decisions/      ADR-001 à ADR-007
  legacy-audit/   audit de la v0, justification des ADR
```

Règle : aucun module hors de `storage/` n'importe SQLAlchemy ou psycopg. Un test le vérifie.

## Contribuer

Voir `CONTRIBUTING.md`. En résumé : `main` est stable et n'est jamais poussée directement, tout le travail passe par `dev` puis une PR relue par l'autre.

## Documents

- `docs/architecture.md` : schéma, flux, interfaces, grille d'erreurs, concurrence.
- `docs/decisions/` : une fiche par décision, avec le contexte et les alternatives écartées.
- `docs/extraction-examples.md` : entrées/sorties du LLM, fixtures des tests.
- `tests/acceptance/SCENARIOS.md` : les douze scénarios qui ferment les jalons.
- `docs/legacy-audit/` : l'audit de la v0, pour se souvenir pourquoi.