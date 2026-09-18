# ADR-005 : Seuils de similarité fixés par mesure, pas par intuition

Statut : acceptée, septembre 2026
Décideurs : Clément, Samy

## Contexte 

La v0 avait des seuils codés en dur sans justification. L'audit a montré qu'ils ne correspondaient à rien de mesuré. Les niveaux (b) et (c) de la contradiction dépendant de deux seuils dont la valeur change directement le comportement : trop haut, on créer des doublons ; trop bas, on écrase des faits distincts.

## Décision

Les seuils `QUOREX_SIM_MATCH` (défaut 0,85) et `QUOREX_SIM_AMBIGUOUS` (défaut 0,70) sont des valeurs de départ. Ils sont validés ou corrigés par une mesure reproductible :

1. Un fichier `data/attribute_pairs.csv` d'au moins 200 paires d'attributs étiquetées à la main par les deux cofondateurs : `same` (même attributs), `different`, `ambiguous` (un humain hésite).
2. Un script `scripts/measure_tresholds.py` calcule, pour le modèle d'embedding configuré, la similarité de chaque paire et produit précision et rappel du niveau (b) pour chaque seuil de 0,50 à 0,95 par pas de 0,01, ainsi que la proportion de faires qui tomberaient en zone (c).
3. Les seuils retenus sont ceux qui maximisent la précision du niveau (b) au-dessus de 0,95 tout en gardant la zone (c) sous 15% des paires. Le résultat et la date de la mesure son consignés en bas de cette fiche.
4. Tout changement de modèle d'embedding impose de rejouer la mesure.

## Alternatives écartées

- Seuils fixes "raisonnables" : c'est la v0.
- Apprentissage automatique des seuils sur les données de production : pas de production, et biais d'auto-confirmation.

## Conséquences 

- Le CSV est le premier jeu de données du produit et un actif : il grossit avec les cas rencontrés en production.
- Les seuils défendables devant un client technique : on montre la courbe.
- Le script est aussi le test de non-régression d'un changement de modèle d'embedding.

## Mesures
| Date | Modèle | SIM_MATCH |
SIM_AMBIGUOUS | Précision (b) | Zone (c)
| Paires |
| --- | --- | --- | --- | --- | --- | --- |
| à compléter | | | | | | |