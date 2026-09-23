# Scénarios d'acceptation QUOREX v1

Ce fichier est la définition de « terminé ». Un jalon n'est fermé que quand ses scénarios passent en test automatisé, avec le `FakeLLMProvider` pour les scénarios qui impliquent une extraction ou un arbitrage.

Convention : un utilisateur `u1` du tenant `T1`, les dates sont en jours relatifs (J0 = premier appel). Les attributs sont écrits sous forme normalisée. « Actif » signifie `valid_to IS NULL`.

Chaque scénario indique le jalon qui doit le faire passer.

---

## S1. Remplacement simple (jalon 2)

Le cas de la démo.

État initial : aucun fait pour `u1`.

Actions :
1. J0 : `remember` structuré `couleur_preferee = bleu`.
2. J12 : `remember` structuré `couleur_preferee = rouge`.

État attendu :
- Un fait actif : `couleur_preferee = rouge`, `valid_from = J12`, `recorded_at = J12`.
- Un fait clos : `couleur_preferee = bleu`, `valid_from = J0`, `valid_to = J12`, `end_reason = replaced`, `replaced_by` = id du fait rouge.
- La réponse du second appel contient `action: replaced` et `replaced_fact_id`.
- `recall()` renvoie uniquement `rouge`.
- Résolution au niveau (a), aucun appel LLM.

---

## S2. Transitoire ne remplace pas un durable (jalon 6)

État initial : aucun fait.

Actions :
1. J0 : `remember` texte « j'habite à Paris ». Le fake LLM renvoie `ville_residence = Paris`, `durability = durable`.
2. J5 : `remember` texte « je suis à Paris pour le week-end ». Le fake LLM renvoie `localisation_actuelle = Paris`, `durability = transient`, `expires_at = J7`.

État attendu :
- Deux faits actifs, attributs distincts.
- Aucun `replaced_by`.
- À J8, `recall()` ne renvoie plus `localisation_actuelle` (expiré) mais toujours `ville_residence`.

Variante S2b : le transitoire est ignoré avec l'avertissement transient_ignored_durable_exists, le durable reste seul actif

---

## S3. Invalidation sans remplaçant (jalon 6)

État initial : `manager = Julie`, actif depuis J0.

Actions :
1. J20 : `remember` texte « Julie est partie ». Le fake LLM renvoie `{attribute: manager, value: null, invalidates: manager}`.

État attendu :
- Le fait `manager = Julie` est clos : `valid_to = J20`, `end_reason = invalidated`, `replaced_by = null`.
- Aucun fait actif pour `manager`.
- `recall()` ne renvoie rien pour `manager`.
- `recall(as_of = J10)` renvoie `manager = Julie`.

---

## S4. Attributs proches mais distincts (jalon 5)

État initial : `couleur_preferee = bleu`, actif.

Actions :
1. `remember` structuré `couleur_voiture = rouge`.

État attendu :
- Deux faits actifs : `couleur_preferee = bleu` et `couleur_voiture = rouge`.
- Le niveau (a) ne matche pas. La similarité entre les deux attributs est mesurée : si elle est sous 0,70, création directe ; si elle est entre 0,70 et 0,85, le fake LLM est appelé et répond `different`, création. Dans les deux cas, aucun remplacement.
- Ce scénario est aussi une ligne du CSV de mesure des seuils (ADR-005) avec l'étiquette `different`.

---

## S5. Même attribut en français et en anglais (jalon 5)

État initial : `couleur_preferee = bleu`, actif.

Actions :
1. `remember` structuré `favorite_color = red`.

État attendu, selon la table de synonymes :
- Si `favorite_color` est une variante de `couleur_preferee` dans `synonyms.yaml` : résolution au niveau (a), remplacement, un seul fait actif `couleur_preferee = red`.
- Sinon : résolution au niveau (b) ou (c), remplacement attendu, et le cas est ajouté à `synonyms.yaml` en sortie de test (le test échoue avec un message explicite si ni (a) ni (b) ni (c) ne remplacent).

---

## S6. Time travel avec as_of (jalon 3)

État initial : le résultat de S1 (bleu clos à J12, rouge actif).

Actions :
1. `recall(as_of = J5)`.
2. `recall(as_of = J12)`.
3. `recall(as_of = J20)`.
4. `recall(as_of = J-1)` (avant tout enregistrement).

État attendu :
1. `couleur_preferee = bleu` uniquement.
2. `couleur_preferee = rouge` uniquement (le fait rouge est valide à partir de J12 inclus, le bleu jusqu'à J12 exclu).
3. `couleur_preferee = rouge`.
4. Liste vide.

Variante S6b, connaissance rétroactive : à J30, `remember` structuré `ville_naissance = Lyon` avec `valid_from = J-3650` (un fait vrai depuis toujours, appris tard). Attendu : `recall(as_of = J10)` ne renvoie pas `ville_naissance` (le système ne le savait pas), `recall(as_of = J31)` le renvoie. C'est ce qui distingue le bitemporel d'un simple axe de validité.

---

## S7. Diff entre deux dates (jalon 3)

État initial : le résultat de S1 et de S3 (bleu remplacé par rouge à J12, Julie invalidée à J20), plus `langue = fr` enregistré à J15.

Actions :
1. `diff(from = J10, to = J25)`.

État attendu :
- `added` : `couleur_preferee = rouge` (J12), `langue = fr` (J15).
- `replaced` : `couleur_preferee = bleu` avec `replaced_by` = rouge.
- `invalidated` : `manager = Julie`.
- `diff(from = J0, to = J5)` : tout vide sauf `added` contenant `couleur_preferee = bleu` et `manager = Julie`.
- `diff(from = J25, to = J10)` : erreur 422 `invalid_time_range`.

---

## S8. Deux remember concurrents sur le même attribut (jalon 2)

État initial : aucun fait.

Actions :
1. 20 appels `remember` structuré `couleur_preferee = valeur_i` (i de 1 à 20) lancés en parallèle.

État attendu :
- Exactement un fait actif pour `couleur_preferee`.
- 19 faits clos avec `end_reason = replaced`, formant une chaîne via `replaced_by` sans cycle ni trou : en partant du fait actif et en remontant les `replaced_by` inverses, on retrouve les 20 faits.
- Aucune erreur 500. Au plus quelques 409 `concurrent_write` acceptables si le retry unique échoue, mais le total de faits (actifs + clos) est égal au nombre d'appels ayant renvoyé 200.
- L'index `facts_active_unique` n'a jamais été contourné.

---

## S9. Isolation entre tenants (jalon 1)

État initial : deux tenants `T1` et `T2`, chacun avec un utilisateur d'`external_id = "alice"`.

Actions :
1. Avec la clé de `T1` : `remember` `couleur_preferee = bleu` pour `alice`.
2. Avec la clé de `T2` : `recall()` pour `alice`.
3. Avec la clé de `T2` : `forget` par id du fait créé en 1.
4. Avec la clé de `T2` : `remember` `couleur_preferee = vert` pour `alice`.
5. Avec la clé de `T1` : `recall()` pour `alice`.

État attendu :
2. Liste vide (ou 404 `user_not_found` si `alice` de T2 n'existe pas encore).
3. 404 `fact_not_found`.
4. 200, création dans T2.
5. `couleur_preferee = bleu` uniquement. Le fait de T2 est invisible.
- Une clé révoquée (`revoked_at` renseigné) renvoie 401 `invalid_api_key` sur toute route.

---

## S10. Extraction : LLM indisponible ou sortie invalide (jalon 6)

Actions :
1. `remember` texte avec le fake LLM configuré pour lever `LLMUnavailable`.
2. `remember` texte avec le fake LLM configuré pour renvoyer un JSON non conforme au schéma.
3. `remember` texte avec le fake LLM configuré pour renvoyer un fait avec `confidence = 0.3`.

État attendu :
1. 503 `llm_unavailable`. Le message est stocké dans `messages`. Aucun fait écrit. Le `request_id` est dans la réponse.
2. 200, `facts: []`, un avertissement `extraction_invalid_output`. Le message et la sortie brute sont stockés.
3. 200, `facts: []`, un avertissement `fact_rejected` avec `reason = low_confidence`.

Variante S10b, arbitrage indisponible : niveau (c) atteint et fake LLM lève `LLMUnavailable`. Attendu : création sans remplacement, avertissement `arbitration_unavailable`, deux faits actifs.

---

## S11. forget (jalon 3)

État initial : `couleur_preferee = rouge` et `langue = fr`, actifs.

Actions :
1. `forget(attribute = couleur_preferee)`.
2. `forget(fact_id = <id de langue>)`.
3. `forget(fact_id = <id inconnu>)`.

État attendu :
1. Le fait est clos, `end_reason = forgotten`, `replaced_by = null`. Toujours présent en base. `recall(as_of = avant)` le renvoie encore.
2. Idem pour `langue`.
3. 404 `fact_not_found`.
- Aucun `DELETE` n'a été exécuté (vérifiable par le compte de lignes de `facts`).

---

## S12. Normalisation des attributs (jalon 2)

Table de cas pour `normalize()`, tous attendus égaux à la forme canonique :

| Entrée | Sortie |
|---|---|
| `Couleur préférée` | `couleur_preferee` |
| `couleur  préférée ` | `couleur_preferee` |
| `couleur-préférée` | `couleur_preferee` |
| `COULEUR_PREFEREE` | `couleur_preferee` |
| `couleur favorite` | `couleur_preferee` (via synonymes) |
| `favorite color` | `couleur_preferee` (via synonymes) |
| `Ville de résidence` | `ville_residence` |
| `l'âge` | `age` |
| `âge` | `age` |

Et deux cas qui ne doivent pas être confondus : `couleur_preferee` et `couleur_voiture` restent distincts après normalisation.

---

## Correspondance jalons / scénarios

| Jalon | Scénarios à faire passer |
|---|---|
| 1. Stockage et modèle | S9 |
| 2. remember structuré, niveau (a) | S1, S8, S12 |
| 3. recall as_of, diff, forget | S6, S6b, S7, S11 |
| 4. Démo enregistrée | S1 + S6 + S7 joués à l'écran |
| 5. Embeddings, niveau (b) | S4, S5 |
| 6. Extraction LLM, niveau (c) | S2, S2b, S3, S10, S10b |
| 7. API complète | tous, plus les codes d'erreur de la grille vérifiés un par un |