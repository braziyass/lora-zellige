# Règles de contribution

## Issues
- Toute tâche = une issue (modèle `.github/ISSUE_TEMPLATE/tache.md`), avec un responsable et une étiquette :
  `données`, `entraînement`, `évaluation`, `démo`, `rapport`, `éthique`.
- On ferme une issue par un commit ou une PR : `closes #7`.

## Branches
- `main` reste toujours exécutable.
- Une branche par issue : `feat/12-sweep-rangs`, `data/5-legendes`, `docs/20-model-card`.
- Pull request relue par un autre membre avant fusion.

## Messages de commit
Format : `<type>: <ce qui change>` — au présent, précis.

| Type | Exemple |
|---|---|
| `feat` | `feat: ajoute le calcul de diversité entre graines` |
| `fix` | `fix: retire le jeton zllg avant le CLIP-score` |
| `data` | `data: ajoute 12 légendes (portes de Fès) dans sources.csv` |
| `exp` | `exp: résultats balayage rangs 4-32, 1500 étapes` |
| `docs` | `docs: complète la fiche du modèle` |
| `refactor` | `refactor: déplace le chargement du pipeline dans generate.py` |

❌ `update`, `fix bug`, `wip`, `final version 2`

## Ce qu'on ne versionne pas
Images, poids (`*.safetensors`), générations brutes : voir `.gitignore`.
On versionne les **figures retenues** et les **tableaux CSV** de `results/`.

## Notebooks
Numérotés, exécutables de haut en bas sur Colab ; la logique va dans `src/`, le notebook l'appelle.
Vider les sorties lourdes avant de commiter.
