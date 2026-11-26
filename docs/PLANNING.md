# Planning hebdomadaire (8 semaines — à ajuster au calendrier du cours)

Chaque ligne = une ou plusieurs **issues** GitHub (`tools/create_issues.sh` les crée d'un coup).
Rôles proposés : **A** = données & éthique · **B** = entraînement & coût · **C** = évaluation & démo.
Tout le monde écrit une partie du rapport.

| Sem. | Objectif | Livrable dans le dépôt | Resp. |
|---|---|---|---|
| 1 | Choix du domaine, lecture LoRA, Colab opérationnel | `README` initial, `docs/JOURNAL.md`, issues créées | Tous |
| 2 | Collecte 40-80 candidates, tri, licences | `data/sources.csv` rempli, `notebooks/01_donnees` | A |
| 3 | Légendes manuelles, préparation 512², split | `data/processed/report.json` (non versionné) + fiche des données | A, C |
| 3 | Baseline : modèle de base sur les prompts d'évaluation | `notebooks/02_baseline`, `results/figures/baseline_*.png` | C |
| 4 | Première LoRA (rang 8, 1500 étapes) | `run_info.json` résumé dans `results/`, premières images | B |
| 5 | Balayage rangs 4/8/16/32 × 250-1500 étapes | `results/costs.csv`, `notebooks/03_entrainement_lora` | B |
| 6 | Évaluation chiffrée + grilles visuelles | `results/metrics_summary.csv`, `results/figures/` | C |
| 6 | Option : variation de l'intensité (0,5 / 0,8 / 1,0 / 1,2) | `results/figures/scale_*.png` | C |
| 7 | Démo Gradio + vidéo (≤ 2 min) | `demo/app.py`, `demo/README.md` (lien vidéo) | C, B |
| 7 | Fiche du modèle, éthique, licences | `docs/` | A |
| 8 | Rapport 6-8 pages, répétition soutenance 10 min | `report/rapport.pdf`, slides | Tous |

## Rituel hebdomadaire
1. Mettre à jour `docs/JOURNAL.md` (fait / problèmes / prochaine étape, par membre).
2. Fermer les issues terminées en citant le commit (`closes #12`).
3. Pousser au moins un commit par membre.
