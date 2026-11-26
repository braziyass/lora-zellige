# Réflexion éthique

## 1. Données et consentement
- Seules des images à licence compatible sont utilisées ; chaque image est tracée dans `data/sources.csv`.
- Aucun nom d'artiste vivant dans les légendes ni dans les prompts.
- Photos prises par l'équipe : pas de personnes identifiables, accord des propriétaires des lieux privés.

## 2. Patrimoine culturel
Le zellige est un savoir-faire artisanal marocain, transmis par des maâlems.
- **Attribution :** la démo et le rapport nomment l'origine du savoir-faire.
- **Représentation :** on vérifie que le modèle ne mélange pas le zellige avec des clichés
  « orientalistes » (prompts de contrôle, inspection visuelle).
- **Impact économique :** l'outil sert à l'exploration et à l'enseignement, pas à remplacer la
  production artisanale ; on le dit explicitement dans la fiche du modèle.

## 3. Transparence
- Les images générées sont marquées « générées par IA » dans la démo et le rapport.
- Les limites (géométrie imparfaite, biais du corpus) sont documentées.

## 4. Mémorisation
Avec 20-50 images, le risque de recopier une image d'entraînement est réel.
On le **mesure** (similarité DINOv2 au plus proche voisin, `src/evaluate.py`) et on choisit
une configuration qui reste au niveau naturel des images de test.

## 5. Coût environnemental
On rapporte le temps GPU total du projet (somme des `run_info.json` + générations)
et on évite les balayages inutiles (un entraînement par rang, checkpoints réutilisés).

## 6. Usage abusif
La licence OpenRAIL-M de SD 1.5 s'applique ; la LoRA est publiée avec les mêmes restrictions.
