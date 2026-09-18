# ADR-002 : Modèle de fait bitemporel

Statut : acceptée, septembre 2026
Décideurs : Clément, Samy

## Contexte : 

Les mémoires épisodiques (mem0, Zep, v0) stockent des textes et les retrouvent par similarité. Quand un utilisateur change d'avis, les deux version coexistent et l'agent choisit au hasard. Le différenciateur annoncé de QUOREX est le remplacement des faits périmés. Pour que ce remplacement soit auditable et ce qui est vrai vraiment, mais ce qui était vrai à une date, et ce que le système savait à une date.

## Décision :

Chaque fait porte deux axes de temps :

- Temps de validité : `valid_from`, `valid_to` (null si actif). Quand le fait est vrai dans la vie de l'utilisateur.
- Temps de connaissance : `recorded_at`, immuable. Quand le système l'a appris.

Un fait n'est jamais supprimé physiquement. Il est clos : `valid_to` renseigné, `end_reason` parmi `replaced`, `invalidated`, `expired`, `forgotten`, et `replaced_by` pointant sur le successeur quand il existe.

`recall` accepte `as_of` : renvoie les faits tels que `recorded_at <= as_of AND valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of).` `diff` calcule les faits ajoutés, remplacés et invalidés, entre deux dates.

## Alternatives écartées :

- Un seul axe de temps (validité seule) : ne permet pas de distinguer "le fait était vrai le 5" et "le système le savait le 5". Un fait appris rétroactivement fausse l'historique.
- Suppression physique avec table d'archive : deux tables à maintenir en cohérence, pas de `replaced_by` naturel.
- Versionnage par numéro : ne donne pas de sémantique temporelle, impossible d'exprimer une invalidation sans remplaçant.

## Conséquences

- Le time travel et le diff sont des requêtes SQL, pas de la logique applicative. Ce sont les deux fonctions de la démo.
- La table `facts` ne fait que croître. Acceptable en v1 ; une politique d'archivage des faits clos depuis plus N mois viendra avec le volume.
- `forget` clôt avec `forgotten` et ne supprime rien. Une demande RGPD d'effacement complet nécessitera une opération dédiée hors API publique, à documenter avant le premier client européen sérieux. 