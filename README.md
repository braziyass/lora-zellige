# LoRA Zellige — spécialiser un modèle texte-image avec peu de données

> **Projet 5 — IA générative visuelle.** On adapte Stable Diffusion 1.5 au **zellige marocain**
> avec une LoRA entraînée sur 20 à 50 images sous licence libre, sans réentraîner le modèle.
> On compare plusieurs **rangs** et **nombres d'étapes**, puis on mesure le **respect du texte**,
> le **coût** et le **surapprentissage**.

| | |
|---|---|
| Modèle de base | `stable-diffusion-v1-5/stable-diffusion-v1-5` (512 × 512) |
| Matériel | Google Colab / Kaggle, GPU T4 ou P100, **≤ 12 Go de VRAM** (pic mesuré ≈ 6-7 Go) |
| Jeton de concept | `zllg` → prompts du type « `zllg zellige, …` » |
| Équipe | Membre A · Membre B · Membre C *(à compléter)* |

---

## 1. Installation (Colab / Kaggle)

```bash
git clone https://github.com/braziyass/lora-zellige.git
cd lora-zellige
pip install -q -r requirements.txt
```

Ou ouvrez directement les notebooks de `notebooks/` (la première cellule fait l'installation).

## 2. Reproduire toutes les expériences

```bash
# 1) Données : téléchargement sous licence libre puis légendage MANUEL dans data/sources.csv
python data/download_commons.py --category "Zellij in Morocco" --limit 80
#    → ouvrir data/sources.csv, mettre keep=1/0 et écrire une légende courte par image
python -m src.prepare_data --config configs/base.yaml

# 2) Première LoRA simple (rang 8, 1500 étapes, ~15 min sur T4)
python -m src.train_lora --config configs/base.yaml

# 3) Balayage rangs 4/8/16/32 × checkpoints 250/500/1000/1500
python -m src.sweep --sweep configs/sweep.yaml

# 4) Génération (base + tous les checkpoints, mêmes prompts et graines)
python -m src.generate --config configs/base.yaml --include_base

# 5) Évaluation chiffrée + figures
python -m src.evaluate --config configs/base.yaml

# 6) Démo
python demo/app.py --share
```

## 3. Organisation du dépôt

| Dossier | Contenu |
|---|---|
| `notebooks/` | `01_donnees` · `02_baseline` · `03_entrainement_lora` · `04_evaluation` · `05_demo` |
| `src/` | `prepare_data` · `dataset` · `train_lora` · `sweep` · `generate` · `evaluate` · `plots` · `utils` |
| `configs/` | `base.yaml` (une expérience) · `sweep.yaml` (balayage) |
| `data/` | script de téléchargement, `sources.csv` (traçabilité des licences), fiche des données — **sans images** |
| `prompts/` | prompts d'évaluation (domaine + composition) |
| `results/` | tableaux de mesures et figures retenues |
| `demo/` | application Gradio + lien vers la vidéo |
| `report/` | rapport final (6-8 pages) |
| `docs/` | licences, fiche du modèle, éthique, planning, journal |

Les images, poids LoRA (`outputs/`) et générations brutes sont exclus de Git (`.gitignore`).
Les LoRA finales sont publiées sur Hugging Face Hub *(lien à ajouter)*.

## 4. Protocole d'évaluation

| Question | Mesure | Outil |
|---|---|---|
| Le modèle respecte-t-il le texte ? | **CLIP-score** (jeton `zllg` retiré du texte) sur prompts de *composition* | CLIP ViT-L/14 |
| A-t-il appris le domaine ? | **Similarité au domaine** : cosinus CLIP moyen avec les images de **test** réelles | CLIP ViT-L/14 |
| Surapprend-il / copie-t-il ? | **Similarité au plus proche voisin d'entraînement**, ratio vs le niveau naturel des images de test, taux de quasi-copies | DINOv2-small |
| Les sorties s'effondrent-elles ? | **Diversité** entre graines pour un même prompt | DINOv2-small |
| Combien ça coûte ? | Temps d'entraînement, pic de VRAM, paramètres entraînés, taille du fichier LoRA | `run_info.json` |

Trois familles de prompts (`prompts/eval_prompts.csv`) :
- **in_domain** : proches des légendes d'entraînement ;
- **held_out** : légendes des images de test, jamais vues ;
- **composition** : le concept dans un contexte nouveau (« une tasse décorée de zellige… »).
  Une LoRA qui surapprend produit du zellige **mais oublie la tasse** → son CLIP-score chute.

## 5. Résultats principaux

Run complet sur Colab (GPU T4) : 4 rangs × 4 checkpoints + modèle de base, 15 prompts × 3 graines = 45 images par configuration.
Tableau complet : [`results/metrics_overview.csv`](results/metrics_overview.csv). Rang 0 = modèle de base sans LoRA.

| Rang | Étapes | CLIP composition ↑ | Sim. domaine (test) ↑ | Ratio mémorisation ↓ | Diversité ↑ | Taille (Mo) |
|---|---|---|---|---|---|---|
| base | – | 28.59 | 0.708 | 0.56 | **0.437** | – |
| 4 | 1000 | 28.87 | 0.760 | 0.60 | 0.384 | **3.2** |
| 8 | 1500 | 28.22 | 0.791 | 0.67 | 0.363 | 6.4 |
| **16** | **1000** | **29.04** | 0.780 | 0.66 | 0.399 | 12.8 |
| 32 | 1000 | 29.09 | 0.778 | 0.69 | 0.303 | 25.6 |
| 32 | 1500 | **29.58** | **0.805** | 0.74 | 0.312 | 25.6 |

**Coût :** ~13 min d'entraînement par rang (1 500 étapes), quel que soit le rang ; pic de VRAM 2,6 Go (fp16 + gradient checkpointing) ; le fichier LoRA grandit linéairement avec le rang (3,2 → 25,6 Mo).

**Ce que montrent les chiffres :**
- **Le domaine est appris** : la similarité aux images de test réelles passe de 0,708 (base) à 0,76–0,80, et monte avec le nombre d'étapes et le rang.
- **Pas de copie** : taux de quasi-copies = 0 partout et ratio de mémorisation < 1 (référence test réel ↔ train = 0,715), mais il **augmente** avec rang × étapes (0,56 → 0,74) : la LoRA se rapproche des images d'entraînement.
- **La diversité baisse** avec l'entraînement, surtout au rang 32 (0,437 → 0,30) : premier signe de surapprentissage (les sorties se ressemblent).
- **Le respect du texte reste stable** : le CLIP-score sur les prompts de composition varie entre 27,7 et 29,6 autour de la base (28,6). Avec 45 images par configuration, ces écarts (~±1) sont proches du bruit.
- **Le rang coûte en taille, pas en temps** : à 1 000 étapes, le rang 16 obtient presque le même résultat que le rang 32 avec un fichier deux fois plus petit et plus de diversité.

**Configuration retenue : rang 16, 1 000 étapes** — bon compromis domaine (0,780) / texte (29,0) / diversité (0,40), sans copie, 12,8 Mo.
Alternative légère : rang 4, 1 000 étapes (3,2 Mo). Le rang 32 à 1 500 étapes maximise la fidélité au style mais perd en diversité.

**Limites :** petits échantillons (6 prompts de composition × 3 graines) ; le filtre de sécurité de Stable Diffusion a remplacé quelques images par du noir (faux positifs sur les motifs), ce qui abaisse légèrement certains scores ; CLIP ne juge pas la justesse géométrique des étoiles — à compléter par l'inspection visuelle des grilles (`results/figures/grid_*.png`).

## 6. Éthique et licences

Voir [`docs/ETHIQUE.md`](docs/ETHIQUE.md), [`docs/LICENCES.md`](docs/LICENCES.md) et
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). Chaque image d'entraînement est tracée
(auteur, licence, URL) dans `data/sources.csv`.

## 7. Contribuer

Voir [`CONTRIBUTING.md`](CONTRIBUTING.md) (issues, branches, messages de commit) et
le planning hebdomadaire dans [`docs/PLANNING.md`](docs/PLANNING.md).
