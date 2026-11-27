# Fiche du modèle — LoRA Zellige *(à compléter en fin de projet)*

## Résumé
- **Type :** adaptateur LoRA pour le UNet de Stable Diffusion 1.5 (couches d'attention `to_q/k/v/out`)
- **Rang retenu :** 16 · **Étapes :** 1 000 · **Taille :** 12,8 Mo · **Paramètres entraînés :** ~3,2 M (rang 32 : 6,38 M = 0,74 % du UNet)
- **Jeton :** `zllg` — utilisation : `zllg zellige, <description>`
- **Intensité conseillée :** 0,7 – 1,0

## Usage prévu
Exploration créative et pédagogique de motifs de zellige ; illustration ; étude du transfert de style avec peu de données.

## Usages hors périmètre
- Présenter les images comme des photographies réelles ou comme des œuvres d'artisans.
- Reproduire des œuvres protégées ou l'œuvre d'un·e artisan identifiable.
- Concurrencer ou dévaloriser le travail des maâlems (artisans) : citer la source du savoir-faire.

## Données d'entraînement
21 images (16 entraînement / 5 test), Wikimedia Commons, licences libres (voir `data/README.md`).

## Évaluation
| Mesure | Base | LoRA retenue |
|---|---|---|
| CLIP-score (composition) | 28,59 | 29,04 |
| Similarité au domaine (images test) | 0,708 | 0,780 |
| Ratio de mémorisation | 0,56 | 0,66 |
| Taux de quasi-copies | 0 | 0 |
| Diversité entre graines | 0,437 | 0,399 |

## Limites et biais
- Corpus réduit et géographiquement concentré ; motifs peu représentés mal rendus.
- Géométrie souvent imparfaite (symétries, étoiles) : SD 1.5 ne « comprend » pas la construction géométrique.
- Le modèle de base peut produire des clichés « orientalistes » ; vérifier les prompts hors domaine.

## Coût d'entraînement
GPU T4 (Colab), ~13 min pour 1 500 étapes, pic de VRAM 2,6 Go.
