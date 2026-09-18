# ADRE-003 : Règle de contradiction en trois niveaux

Statut : acceptée, septembre 2026
Décideurs : Clément, Samy

## Contexte

L'audit a exécuté les cinq scénarios de contradiction contre le `ConflictResolver` de la v0 : avec l'encodeur de production, les cinq donnaient "renforcement" (le second message n'était jamais écrit). Avec un encodeur idéal, deux sur cinq étaient corrects. Une détection fondée uniquement sur la similarité de texte confond "couleur préférée" et "couleur de ma voiture", et ne distingue pas "j'habite à Paris" de "je suis à Partis ce week-end".

Appeler un LLM à chaque écriture pour trancher coûte et ralentit, et n'est pas nécessaire dans la majorité des cas.

## Décision

Un nouveau fait est rapproché des faits actifs du même utilisateur dans cet odre strict, du moins cher au plus cher:
- (a) Correspondance exacte sur (`subject`, `attribute` normalisé). Si trouvée : remplacement. Jamais d'appel LLM à ce niveau.
- (b) Similarité cosinus entre l'embedding de l'attribut nouveau et ceux des attributs actifs. Si >= `QUOREX_SIM_MATCH` (0,85 au départ) = remplacement.
- (c) Si la meilleure similarité est dans [`QUOREX_SIM_AMBIGUOUS`, `QUOREX_SIM_MATCH` [ (0,70 à 0,85 au départ) : le LLM arbritre entre `same` et `different`. Sinon : création.

Règle de transverses : 

- Un fait `transient` ne remplace jamais un `durable` ou `permanent`.
- Si le LLM est indisponible au niveau (c) : création, avec un avertissement dans la réponse. Mieux vaut un doublon qu'une perte.
- Un fait structuré direct (`attribute`, `value`) passe par (a) et, à défaut, (b) et (c) comme les autres.

## Alternatives écartées

- LLM systématique : coûteux, lent, non déterministe pour la démo.
- Similarité seule sans niveau exacte : c'est la v0.
- Comparaison sur la valeur plutôt que l'attribut : deux valeurs différentes du même attribut sont justement le cas à détecter.

## Conséquences

- Le chemin structuré est gratuit et déterministe : c'est celui de la démo.
- Les seuils sont des points de départ, voir ADR-005.
- Le niveau (b) exige un embedding par attribut, stocké dans `attribute_embedding`, et un index HNSW pgvector.
- Les logs comptent les résolutions par niveau, pour mesurer combien de fois le LLM est réellement appelé.