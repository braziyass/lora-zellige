# Démo

```bash
python demo/app.py --share      # sur Colab : ouvre un lien public temporaire
```

Fonctions : prompt libre (`{trigger}` = jeton du concept), choix du checkpoint (rang / étapes),
curseur d'intensité 0 → 1,5, comparaison avant / après avec la même graine.

**Vidéo (≤ 2 min) :** *lien à ajouter* — scénario conseillé :
1. Prompt du domaine, intensité 0 puis 1,0 ;
2. Prompt de composition (tasse, salle de bain) : montrer que le texte est respecté ;
3. Checkpoint surappris (rang 32, 1500 étapes) : montrer qu'il « oublie » la tasse.
