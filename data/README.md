# Fiche des données

> Aucune image n'est versionnée dans Git. Ce dossier contient uniquement
> le script de téléchargement et la traçabilité des sources.

## Composition

| Élément | Valeur *(à compléter)* |
|---|---|
| Domaine | Zellige marocain (mosaïque de carreaux émaillés taillés) — **un seul concept** |
| Nombre d'images | … au total (… entraînement / … test), cible 20-50 |
| Résolution | 512 × 512 (recadrage centré, `src/prepare_data.py`) |
| Sources | Wikimedia Commons (catégories : `Zellige`, …), Openverse, photos de l'équipe |
| Licences acceptées | CC0, domaine public, CC BY, CC BY-SA (pas de NC / ND) |
| Légendes | Écrites à la main, en anglais (langue du modèle), 8-20 mots, format `zllg zellige, <description>` |

## Pipeline

1. `python data/download_commons.py --category "Zellige" --limit 80` → `data/raw/` + `data/sources.csv`
2. **Tri manuel** dans `sources.csv` : colonne `keep` = 1 / 0. Retirer :
   filigranes, texte incrusté, personnes identifiables, vues trop larges où le zellige est minoritaire,
   autres styles (bois sculpté, plâtre ciselé…) pour ne **pas mélanger les concepts**.
3. **Légende** dans la colonne `caption` : décrire ce qu'on voit (motif, couleurs, support, cadrage).
   - ✅ `eight-pointed star pattern in green, white and black tiles, close-up`
   - ❌ `beautiful zellige` (trop vague) · ❌ nom d'artiste vivant
4. `python -m src.prepare_data` → redimensionnement 512, suppression des quasi-doublons (dHash),
   séparation train / test (5-10 images de test), rapport `data/processed/report.json`.

## Format de `sources.csv`

`file, source_url, page_url, author, license, license_url, commons_description, caption, keep, notes`

Pour une photo de l'équipe : `author` = nom du membre, `license` = `CC BY 4.0`
(ou `own work team`), `source_url` vide, `notes` = lieu et date.

## Limites connues

- Biais de sélection : Commons sur-représente certains monuments (Fès, Marrakech, Alhambra…).
- Le zellige andalou (Espagne) et marocain sont proches ; préciser le choix dans `notes`.
- Petit jeu → risque élevé de mémorisation, mesuré dans `src/evaluate.py`.
