# Résultats

| Fichier | Produit par |
|---|---|
| `metrics_summary.csv` | `src/evaluate.py` — moyennes par configuration et famille de prompts, jointes aux coûts |
| `metrics_per_image.csv` | `src/evaluate.py` — une ligne par image générée |
| `metrics_reference.json` | niveau naturel de similarité test ↔ train, seuil de copie |
| `figures/curves_<catégorie>.png` | courbes rang × étapes pour les 4 mesures |
| `figures/tradeoff.png` | compromis fidélité au domaine / respect du texte |
| `figures/grid_<prompt>_s<graine>.png` | grilles visuelles (lignes = rangs, colonnes = étapes) |

Les générations brutes (`generations/`) ne sont pas versionnées.
