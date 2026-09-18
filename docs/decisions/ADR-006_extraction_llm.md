# ADR-006 : Extraction par LLM derrière une interface remplaçable Gemini en 1

Statut : acceptée, septembre 2026
Décideurs : Clément, Samy

## Contexte

La v0 n'avait pas d'extraction : un message était stocké tel quel. Le modèle de fait de la v1 exige de transformer un texte libre en faits structurés (sujet, attribut, valeur, confiance, durabilité). Un petit modèle local sur CPU est trop lent et peu fiable sur les cas ambigu ; une API payante coûte une infrastructure de centime par message mais on crée une dépendance.

L'audit a montré ce qu'un composant figé sur un fournisseur produit : l'encodeur v0 verrouillé sur 26 mots anglais a rendu le produit inutilisable en français.

## Décision

Toute interaction avec un LLM passe par `LLMProvider.complete_json(prompt, schema)`. Le moteur n'importe aucun SDK de fournisseur. L'implémentation est choisir par `QUOREX_LLM_PROVIDER`.

En v1 : Gemini Flash, version la plus récente disponible sur le tier gratuit au moment du développement, avec sortie JSON contrainte par schéma. Un `FakeLLMProvider` sert les tests d'acceptation avec des sorties fixées.

L'extraction est synchrone. Le schéma de sortie inclut `durability`, `expires_at` et `invalidates` (pour un message qui rend un fait faux sans le remplacer).

Erreurs : `LLMUnavailable` remonte en 503 à l'extraction (le message est stocké, rejouable) ; au niveau (c) de la contradiction, on crée sans remplacer et on avertit. `LLMInvalidOutput` après une tentative de réparation : zéro fait extrat, avertissement.

## Alternatives écartées

- Modèle local (Qwen, Llama via Ollama) : plusieurs secondes par message sur CPU, fiabilité insuffisante sur l'ambigu. Reste une option pour un client qui exige que rien ne sorte de chez lui.
- API payante dès le départ : inutile sans client ; la bascule se fait par variable d'environnement.
- Extraction asynchrone : plus complexe à tester et à démontrer ; viendra quand un client se plaindra de la latence.

## Conséquences

- Le coût LLM par message est mesurable dans les logs (`llm_calls`, `llm_ms`) et sera refacturé dans le prix.
- Les limites du tier gratuit suffisent pour développer et démontrer, pas pour un client en production : la bascule vers un petit modèle payant est prévue avant le premier client.
- Les exemples d'entrée/sortie des cinq scénarios de l'audit sont des fixtures : changer de fournisseur, c'est vérifier qu'elles passent encore.