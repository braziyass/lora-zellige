#!/usr/bin/env bash
# Crée les étiquettes et les issues du planning (docs/PLANNING.md).
# Prérequis : GitHub CLI connecté (gh auth login), exécuter à la racine du dépôt.
set -euo pipefail

for l in données entraînement évaluation démo rapport éthique; do
  gh label create "$l" --force >/dev/null
done

issue () { gh issue create --title "$1" --label "$2" --body "$3"; }

issue "[sem. 1] Configurer Colab et lancer un premier test SD 1.5" "entraînement" "Vérifier VRAM, temps d'une image, installation requirements."
issue "[sem. 2] Collecter 40-80 images candidates sous licence libre" "données" "data/download_commons.py + photos équipe. Remplir data/sources.csv."
issue "[sem. 2] Vérifier les licences et exclure NC/ND" "éthique" "Colonne license de sources.csv, docs/LICENCES.md."
issue "[sem. 3] Écrire les légendes et préparer le jeu 512x512" "données" "Colonne caption, src/prepare_data.py, 5-10 images de test."
issue "[sem. 3] Baseline : modèle de base sur prompts d'évaluation" "évaluation" "python -m src.generate --base_only ; notebook 02."
issue "[sem. 4] Première LoRA rang 8 / 1500 étapes" "entraînement" "src.train_lora, noter temps et VRAM."
issue "[sem. 5] Balayage rangs 4/8/16/32" "entraînement" "src.sweep, résumer les coûts dans results/."
issue "[sem. 6] Évaluation chiffrée et figures" "évaluation" "src.evaluate : CLIP-score, domaine, mémorisation, diversité."
issue "[sem. 6] Étude de l'intensité de la LoRA" "évaluation" "Échelles 0.5 / 0.8 / 1.0 / 1.2 sur la meilleure config."
issue "[sem. 7] Démo Gradio + vidéo" "démo" "demo/app.py, vidéo ≤ 2 min, lien dans demo/README.md."
issue "[sem. 7] Fiche du modèle et réflexion éthique" "éthique" "docs/MODEL_CARD.md, docs/ETHIQUE.md."
issue "[sem. 8] Rapport 6-8 pages" "rapport" "Suivre report/PLAN_RAPPORT.md."
issue "[sem. 8] Préparer la soutenance (10 min)" "rapport" "Slides + répétition chronométrée."
echo "Issues créées."
