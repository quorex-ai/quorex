# ADR-007 : Multi-tenant dès le schéma, clés API haschées

Statut : acceptée, septembre 2026
Décideurs : Clément, Samy

## Contexte

La v0 avait une clé statique unique comparée avec `!=`, sans notion de client : quiconque possédait la clé lisait et purgeait tous les `user_id`. Ajouter l'isolation après coup est une migration de toutes les tables et de toutes les requêtes.

## Décision

- Toute table portant des données a `tenant_id`. Toute requête de lecture ou d'écriture filtre dessus. `FactStore` reçoit le `tenant_id` du contexte de requête et ne peut pas l'omettre.
- Une clé API appartient à un tenant. Format `qx_` + prefix de 8 caractères + 32 caractères aléatoires. Seul le hash argon2id est stocké ; le prefix sert à retrouver la ligne. Révocation par `revoked_at`.
- Un utilisateur final est identifié par (`tenant_id`, `external_id`), créé à la volée au premier `remember`.
- Aucun secret dans le dépôt. `.env.example` ne contient que des noms de variables. Le service refuse de démarrer si une variable obligatoire manque.

## Alternative écartées

- Un tenant implicite en v1, multi-tenant plus tard : c'est la migration douloureuse qu'on veut éviter.
- Row-level security PostgreSQL dès la v1 : bonne garantie, mais ajoute de la complexité de connexion (un rôle par requête). Le schéma la permet ; elle viendra quand un client l'exigera.
- JWT : inutile pour du serveur à serveur, une clé opaque hachée suffit.

## Conséquences

- Un test d'acceptation vérifie qu'une clé du tenant A ne lit aucun fait du tenant B, même avec un `external_id` identique.
- La rotation de clé est une opération standard : créer, migrer, révoquer.
- La table `tenants` porte aussi le modèle d'embedding et se dimension, ce qui prépare un modèle par tenant sans le faire en v1.
` 