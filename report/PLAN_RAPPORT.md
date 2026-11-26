# Plan du rapport (6 à 8 pages) et de la soutenance (10 min)

## Rapport

| § | Section | Pages | Contenu clé | Figures / tableaux |
|---|---|---|---|---|
| 1 | Introduction | 0,5 | Problème : spécialiser un grand modèle avec 20-50 images ; pourquoi le zellige | 1 image base vs LoRA |
| 2 | Notions | 1 | Diffusion latente en 5 lignes ; LoRA : W + (α/r)·B·A, ce qu'est le rang ; rôle du jeton rare | Schéma LoRA |
| 3 | Données | 1 | Sources, licences, tri, légendes, split train/test, limites | Tableau du corpus, mosaïque de 12 images |
| 4 | Méthode | 1 | SD 1.5, couches ciblées, hyperparamètres, contraintes 12 Go, protocole (prompts, graines) | Tableau des hyperparamètres |
| 5 | Résultats | 2 | Rang × étapes sur les 4 mesures ; compromis domaine / texte ; coût ; intensité | `curves_*.png`, `tradeoff.png`, grilles visuelles, tableau principal |
| 6 | Discussion | 1 | Quand surapprend-on ? Quel rang suffit ? Limites des métriques (CLIP ne voit pas la géométrie) | Exemples d'échecs |
| 7 | Éthique | 0,5 | Licences, patrimoine, mémorisation, transparence | – |
| 8 | Conclusion | 0,5 | Configuration recommandée, pistes (SDXL, DoRA, encodeur de texte) | – |
| A | Annexes | hors limite | Attributions des images, contributions de chaque membre | – |

## Soutenance (10 min)
1. Problème et démo « avant / après » en direct (1 min 30)
2. Données et licences (1 min 30)
3. LoRA en une diapositive (1 min)
4. Résultats : une courbe, une grille, le tableau principal (3 min)
5. Surapprentissage : un exemple de quasi-copie et comment on le détecte (1 min 30)
6. Limites, éthique, conclusion (1 min 30)
