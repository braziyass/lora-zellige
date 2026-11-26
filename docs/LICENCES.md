# Licences

| Élément | Licence | Conséquence pour le projet |
|---|---|---|
| Code de ce dépôt | MIT (`LICENSE`) | Réutilisation libre avec mention |
| Stable Diffusion 1.5 | CreativeML OpenRAIL-M | Usage autorisé, avec restrictions d'usage (pas de contenu illégal, trompeur, diffamatoire…). Ces restrictions **se transmettent** à notre LoRA. |
| CLIP (OpenAI), DINOv2 (Meta) | MIT / Apache 2.0 | Utilisés uniquement pour l'évaluation |
| Images d'entraînement | CC0 / domaine public / CC BY / CC BY-SA | Attribution obligatoire : voir `data/sources.csv`. Les CC BY-SA imposent de partager les œuvres dérivées sous la même licence → par prudence, la LoRA est publiée en **CC BY-SA 4.0** si une image CC BY-SA est utilisée. |
| Photos de l'équipe | CC BY 4.0 (consentement écrit des auteurs) | Aucune personne identifiable sur les photos |
| LoRA entraînée | OpenRAIL-M + CC BY-SA 4.0 *(à confirmer)* | Fiche du modèle : `docs/MODEL_CARD.md` |

## Attributions

La liste complète (auteur, licence, URL) est générée depuis `data/sources.csv`
et reproduite en annexe du rapport.

## Exclusions

- Licences NC (non commercial) et ND (pas de modification) : **exclues** (l'entraînement est une modification).
- Images sans licence explicite, captures de sites commerciaux, Pinterest, Instagram : **exclues**.
