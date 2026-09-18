# Exemples d'extraction

Entrées et sorties attendues du `LLMProvider` pour l'extraction de faits et l'arbitrage de contradiction. Ces exemples ont trois usages :

1. Fixtures du `FakeLLMProvider` dans les tests d'acceptation (le fake renvoie la sortie associée à l'entrée exacte).
2. Exemples few-shot dans le prompt d'extraction envoyé au vrai LLM.
3. Test de non-régression quand on change de fournisseur : le vrai LLM doit produire des sorties équivalentes (mêmes attributs après normalisation, mêmes valeurs, même durabilité).

Convention : `subject` vaut `user` sauf mention. Les attributs sont écrits en langue naturelle, la normalisation (ADR-004) les canonise ensuite. `confidence` est la confiance du modèle dans le fait extrait, entre 0 et 1.

## Schéma de sortie de l'extraction

```json
{
  "type": "object",
  "required": ["facts"],
  "properties": {
    "facts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["subject", "attribute", "value", "confidence", "durability"],
        "properties": {
          "subject":     { "type": "string" },
          "attribute":   { "type": "string" },
          "value":       { "type": ["string", "null"] },
          "confidence":  { "type": "number", "minimum": 0, "maximum": 1 },
          "durability":  { "type": "string", "enum": ["permanent", "durable", "transient"] },
          "expires_at":  { "type": ["string", "null"], "description": "ISO 8601, obligatoire si transient" },
          "invalidates": { "type": ["string", "null"], "description": "attribut rendu faux sans remplaçant" }
        }
      }
    }
  }
}
```

Règles données au modèle dans le prompt :

- N'extraire que des faits sur l'utilisateur ou sur des personnes et entités de son entourage direct, énoncés ou clairement impliqués. Pas d'inférence au-delà du texte.
- `permanent` : ne change jamais ou presque (date de naissance, ville de naissance, langue maternelle). `durable` : vrai jusqu'à nouvel ordre (ville de résidence, employeur, préférence). `transient` : borné dans le temps, avec une fin estimable (déplacement, humeur, événement à venir).
- Une préférence exprimée par « finalement », « en fait », « plutôt » remplace la précédente : c'est un fait normal, la contradiction est gérée par le moteur, pas par le modèle.
- Un message qui annonce la fin d'un état sans le remplacer (« a quitté », « n'est plus », « c'est fini ») produit un fait avec `value: null` et `invalidates` renseigné.
- Une question, une hypothèse, une négation sans alternative ou un propos sur un tiers sans lien avec l'utilisateur ne produisent aucun fait.
- Toujours renvoyer un objet JSON conforme au schéma, même vide : `{"facts": []}`.

## S1. Remplacement simple

Message 1 (J0) :

```
je préfère le bleu
```

```json
{"facts": [{"subject": "user", "attribute": "couleur préférée", "value": "bleu", "confidence": 0.85, "durability": "durable", "expires_at": null, "invalidates": null}]}
```

Message 2 (J12) :

```
finalement le rouge
```

```json
{"facts": [{"subject": "user", "attribute": "couleur préférée", "value": "rouge", "confidence": 0.8, "durability": "durable", "expires_at": null, "invalidates": null}]}
```

Note : le second message n'a de sens qu'avec le contexte du premier. En v1 l'extraction est sans mémoire de conversation, donc ce cas passe par le chemin structuré dans la démo. Le fake renvoie néanmoins cette sortie pour tester le remplacement au niveau (a) via texte libre.

## S2. Durable vs transitoire

Message 1 :

```
j'habite à Paris
```

```json
{"facts": [{"subject": "user", "attribute": "ville de résidence", "value": "Paris", "confidence": 0.95, "durability": "durable", "expires_at": null, "invalidates": null}]}
```

Message 2 (reçu un vendredi, J5) :

```
je suis à Paris pour le week-end
```

```json
{"facts": [{"subject": "user", "attribute": "localisation actuelle", "value": "Paris", "confidence": 0.9, "durability": "transient", "expires_at": "J7T23:59:59", "invalidates": null}]}
```

Variante S2b, sortie volontairement dégradée pour tester la règle « un transitoire ne remplace pas un durable » :

```json
{"facts": [{"subject": "user", "attribute": "ville de résidence", "value": "Paris", "confidence": 0.6, "durability": "transient", "expires_at": "J7T23:59:59", "invalidates": null}]}
```

## S3. Invalidation sans remplaçant

Message 1 (J0) :

```
mon manager c'est Julie
```

```json
{"facts": [{"subject": "user", "attribute": "manager", "value": "Julie", "confidence": 0.95, "durability": "durable", "expires_at": null, "invalidates": null}]}
```

Message 2 (J20) :

```
Julie est partie de la boîte
```

```json
{"facts": [{"subject": "user", "attribute": "manager", "value": null, "confidence": 0.85, "durability": "durable", "expires_at": null, "invalidates": "manager"}]}
```

Contre-exemple, à ne pas produire : `{"attribute": "manager", "value": "inconnu"}`. Une valeur inventée est une erreur d'extraction, pas une invalidation.

## S4. Attributs proches mais distincts

Message :

```
ma voiture est rouge
```

```json
{"facts": [{"subject": "user", "attribute": "couleur de la voiture", "value": "rouge", "confidence": 0.9, "durability": "durable", "expires_at": null, "invalidates": null}]}
```

Attendu côté moteur : aucun remplacement de `couleur préférée`. Si la similarité entre les deux attributs tombe dans la zone (c), l'arbitrage (voir plus bas) doit répondre `different`.

## S5. Même attribut en anglais

Message :

```
my favorite color is red
```

```json
{"facts": [{"subject": "user", "attribute": "favorite color", "value": "red", "confidence": 0.9, "durability": "durable", "expires_at": null, "invalidates": null}]}
```

Attendu côté moteur : `favorite color` est une variante de `couleur_preferee` dans `synonyms.yaml`, résolution au niveau (a). La valeur n'est pas traduite : l'agent du client décide de la langue.

## Plusieurs faits dans un message

Message :

```
Je m'appelle Marc, j'ai 34 ans, je bosse chez Doctolib depuis mars et je déménage à Lyon le mois prochain
```

```json
{"facts": [
  {"subject": "user", "attribute": "prénom", "value": "Marc", "confidence": 0.95, "durability": "permanent", "expires_at": null, "invalidates": null},
  {"subject": "user", "attribute": "âge", "value": "34", "confidence": 0.9, "durability": "durable", "expires_at": null, "invalidates": null},
  {"subject": "user", "attribute": "employeur", "value": "Doctolib", "confidence": 0.9, "durability": "durable", "expires_at": null, "invalidates": null},
  {"subject": "user", "attribute": "date de début employeur", "value": "mars", "confidence": 0.7, "durability": "durable", "expires_at": null, "invalidates": null},
  {"subject": "user", "attribute": "ville de résidence future", "value": "Lyon", "confidence": 0.8, "durability": "durable", "expires_at": null, "invalidates": null}
]}
```

Notes : l'âge est `durable` et non `permanent` (il change). Le déménagement futur n'est pas encore une ville de résidence ; l'attribut distinct évite d'écraser la ville actuelle avant le fait. Le mois sans année garde une confiance plus basse.

## Sujet autre que l'utilisateur

Message :

```
ma femme s'appelle Léa et elle est allergique aux arachides
```

```json
{"facts": [
  {"subject": "conjoint", "attribute": "prénom", "value": "Léa", "confidence": 0.95, "durability": "permanent", "expires_at": null, "invalidates": null},
  {"subject": "conjoint", "attribute": "allergie", "value": "arachides", "confidence": 0.9, "durability": "durable", "expires_at": null, "invalidates": null}
]}
```

Le `subject` est un rôle relatif à l'utilisateur, pas le prénom, pour que « ma femme » et « Léa » désignent la même entité dans les messages suivants.

## Aucun fait

Messages et sortie attendue identique `{"facts": []}` :

```
tu peux me rappeler ce que j'ai dit hier ?
```

```
je me demande si je devrais déménager
```

```
mon collègue Paul aime le vert
```

Le troisième est un piège : un fait sur un tiers sans lien durable avec l'utilisateur. On l'ignore en v1 plutôt que de créer un sujet `collegue_paul` ; la gestion de tiers arbitraires est hors périmètre.

## S10. Sorties invalides à tolérer ou rejeter

Sortie avec texte autour, à réparer (extraire le premier objet JSON) :

```
Voici les faits : {"facts": [{"subject": "user", "attribute": "langue", "value": "français", "confidence": 0.9, "durability": "durable"}]}
```

Sortie avec champs manquants tolérés : `expires_at` et `invalidates` absents valent `null`. Un `durability` absent est une erreur.

Sortie à rejeter (`LLMInvalidOutput` après une tentative de réparation) :

```
{"facts": [{"attribute": "langue", "value": "français"}]}
```

```
Je ne peux pas extraire de faits de ce message.
```

Sortie à rejeter par le moteur, pas par le provider (`fact_rejected`, `low_confidence`) :

```json
{"facts": [{"subject": "user", "attribute": "plat préféré", "value": "sushi", "confidence": 0.3, "durability": "durable", "expires_at": null, "invalidates": null}]}
```

## Arbitrage de contradiction (niveau c)

Schéma de sortie :

```json
{"type": "object", "required": ["verdict"], "properties": {"verdict": {"type": "string", "enum": ["same", "different"]}, "reason": {"type": "string"}}}
```

Le prompt donne les deux attributs bruts, les deux valeurs, et demande si les deux faits portent sur la même chose dans la vie de l'utilisateur, de sorte que le second remplace le premier.

Cas S4 :

```
Fait existant : "couleur préférée" = "bleu"
Nouveau fait : "couleur de la voiture" = "rouge"
```

```json
{"verdict": "different", "reason": "Une préférence de couleur et la couleur d'un objet possédé sont deux attributs indépendants."}
```

Cas synonymes non couverts par la table :

```
Fait existant : "ville de résidence" = "Paris"
Nouveau fait : "ville où j'habite" = "Lyon"
```

```json
{"verdict": "same", "reason": "Les deux désignent le lieu de résidence actuel."}
```

Cas piège, à ne pas confondre :

```
Fait existant : "employeur" = "Doctolib"
Nouveau fait : "employeur précédent" = "Alan"
```

```json
{"verdict": "different", "reason": "L'employeur actuel et l'employeur précédent coexistent."}
```

Chaque paire ci-dessus est aussi une ligne de `data/attribute_pairs.csv` avec l'étiquette correspondante.