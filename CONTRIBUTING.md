# Contribuer à QUOREX

Deux personnes, un dépôt, pas de protection technique des branches. Ces règles remplacent la protection. Elles sont courtes pour être suivies.

## Branches

- `main` : toujours déployable. Personne n'y pousse directement, jamais. Elle avance uniquement par merge de `dev` quand un jalon est fermé et validé à deux.
- `dev` : branche par défaut, où le travail courant atterrit. Toujours descendante de `main`.
- Branches de travail : créées depuis `dev`, nommées `feat/<sujet>`, `fix/<sujet>`, `docs/<sujet>`. Durée de vie : quelques jours. Supprimées après merge.

Pas de branche `staging`, `release` ou personnelle de longue durée. Une branche qui vit plus d'une semaine à côté de `dev`, on en parle au point hebdo.

## Pull requests

- Une PR par branche de travail, vers `dev`. L'autre relit et approuve avant merge. On ne merge pas sa propre PR sans relecture, même petite.
- Merge par squash, message clair, suppression de la branche.
- Une PR ouverte depuis trois jours sans relecture est un sujet du point hebdo.
- De `dev` vers `main` : merge normal (pas de squash, pas de rebase) pour garder l'historique, uniquement quand les scénarios d'acceptation du jalon passent.

## Interdits

- `git push --force` sur `main` ou `dev`.
- Rebase d'une branche déjà poussée et vue par l'autre.
- Commit d'un secret, même dans `.env.example`. Si ça arrive : la clé est révoquée dans l'heure, pas seulement retirée du fichier.
- Code SQL, import de SQLAlchemy ou psycopg hors de `src/quorex/storage/`.

## Commits

Format : `type: description courte`, en français. Types : `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.

Exemples : `feat: FactStore.replace atomique avec retry sur conflit`, `docs: ADR-005 résultats de la mesure des seuils`.

## Définition de « terminé »

Un jalon est fermé quand ses scénarios de `tests/acceptance/SCENARIOS.md` passent en test automatisé, que la PR est relue, et que le statut est mis à jour dans le dashboard interne. Pas avant.

## Décisions

Tout choix d'architecture qui contraint le reste du code fait l'objet d'une fiche dans `docs/decisions/` (contexte, décision, alternatives écartées, conséquences), relue par l'autre. Ce qui n'est pas écrit n'a pas été décidé.

## Hook local

Pour bloquer le geste réflexe de pousser sur `main`, une fois par machine :

```bash
git config core.hooksPath .githooks
```

`.githooks/pre-push` refuse tout push vers `main`. Il se contourne avec `--no-verify`, ce qui est précisément le geste qu'on ne fait pas.