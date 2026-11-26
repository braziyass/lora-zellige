# Fiche du modèle — LoRA Zellige *(à compléter en fin de projet)*

## Résumé
- **Type :** adaptateur LoRA pour le UNet de Stable Diffusion 1.5 (couches d'attention `to_q/k/v/out`)
- **Rang retenu :** … · **Étapes :** … · **Taille :** … Mo · **Paramètres entraînés :** … (… % du UNet)
- **Jeton :** `zllg` — utilisation : `zllg zellige, <description>`
- **Intensité conseillée :** 0,7 – 1,0

## Usage prévu
Exploration créative et pédagogique de motifs de zellige ; illustration ; étude du transfert de style avec peu de données.

## Usages hors périmètre
- Présenter les images comme des photographies réelles ou comme des œuvres d'artisans.
- Reproduire des œuvres protégées ou l'œuvre d'un·e artisan identifiable.
- Concurrencer ou dévaloriser le travail des maâlems (artisans) : citer la source du savoir-faire.

## Données d'entraînement
… images (voir `data/README.md`), licences libres, légendes manuelles.

## Évaluation
| Mesure | Base | LoRA retenue |
|---|---|---|
| CLIP-score (composition) | | |
| Similarité au domaine | | |
| Ratio de mémorisation | | |
| Taux de quasi-copies | | |

## Limites et biais
- Corpus réduit et géographiquement concentré ; motifs peu représentés mal rendus.
- Géométrie souvent imparfaite (symétries, étoiles) : SD 1.5 ne « comprend » pas la construction géométrique.
- Le modèle de base peut produire des clichés « orientalistes » ; vérifier les prompts hors domaine.

## Coût d'entraînement
GPU …, … min, pic de VRAM … Go.
