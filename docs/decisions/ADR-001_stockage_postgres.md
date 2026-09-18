# ADR 001 : Stockage PostgreSQL + pgvector derrière une interface FactStore

Statut : acceptée, septembre 2026
Décideurs : Clément, Samy

## Contexte 

Le moteur v0 (quorex-legacy) reposait sur un stockage maison : index HNSW en numpy, WAL snapshots écrits à la main, consolidation automatique. L'audit a relevé un checkpoint non atomique, un WAL empoisonné au premier échec d'écriture, une consolidation qui a remplacé 163 souvenirs par 50 centroïdes, un verrou globale 3 secondes et aucune isolation entre clients.

Le problème de QUOREX est relationnel avant d'être vectoriel : un fait a un utilisateur, un tenant, un attribut, un prédécesseur, deux axes de temps. Le recall filtre sur ces champs. La recherche par similarité intervient à un seul endroit, la contradiction, sur quelques dizaines de faits par utilisateur.

## Décision

PostegreSQL avec l'extension pgvector est le stockage de la v1. Toute la logique SQL vit `storage/`, derrière une interface `FactStore` définie dans `domain/`. Aucun module hors de `storage/` n'importe SQLAlchemy, psycopg ou n'écrit de SQL. Un est d'import le vérifie.

La cohérence est portée par la base : index unique partiel sur (`user_id`, `subject`, `attribute`) où `valid_to IS NULL` contraintes `CHECK` liant `valid_to`, `end_reason` et `replaced_by`, transactions sur `replace`.

## Alternatives écartées

- Base vectorielle dédiée (Qdrant, Weavite, Milvus, Pinecone) : excellente pour des millions de vecteurs, faible sur les relations, l'historique et les transactions. On coderait la cohérence à la main, et la synchronisation faits/vecteurs entre deux système est le bug classique.
- SQLite + sqlite-vec : viable pour démarrer, pénible en multi-tenant et en concurrence, migration vers Postgres ensuite. Autant partir sur Postgres avec docker-compose.
- Moteur maison : voir l'audit.

## Conséquences

- Un seul système à héberger, sauvegarder, migrer.
- Le bitemporel s'exprime en SQL pur, sans logique applicative.
- pgvector suffit jusqu'à quelques millions de vecteurs par tenant. Au-delà, on ajoutera un index externe derrière la même interface.
- Un moteur de stockage maison reste possible plus tard, derrière `FactStore`, sans toucher au reste. C'est un projet de R&D financé par des clients, pas un prérequis.