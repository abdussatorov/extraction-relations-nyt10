# Interface — Projet 10 : Extraction de relations NYT10

## Ce que fait l'application
- Saisir un texte anglais ou choisir un exemple.
- Détecter les entités reconnues par spaCy (NER).
- Prédire les relations entre les entités avec le modèle entraîné dans le notebook.
- Ajuster le seuil de confiance.
- Afficher les triplets et le graphe orienté.
- Exporter les triplets en CSV.

## Étape 1 — Installer les dépendances (PowerShell)
Dans le dossier du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Étape 2 — Préparer le modèle
1. Ouvre le notebook `Projet10_Extraction_Relations_NYT10_Final (1).ipynb`.
2. Exécute les cellules dans l'ordre jusqu'à la cellule d'entraînement final.
3. Exécute la cellule ajoutée à la fin du notebook qui exporte `final` vers `artifacts/relation_model.joblib`.
4. Garde `app.py` et `relation_extraction_core.py` dans le même dossier que le notebook.

## Étape 3 — Lancer l'interface
```powershell
streamlit run app.py
```
Streamlit affiche une adresse locale à ouvrir dans le navigateur.

## Important
- L'interface utilise le modèle réellement entraîné par le notebook. Elle ne contient pas de modèle fictif.
- Le texte de démonstration est en anglais car le modèle et spaCy `en_core_web_sm` sont conçus ici pour l'anglais.
- Si le modèle ne retourne aucun triplet, baisse le seuil ou essaie une phrase où la relation est explicite.
- Les performances dépendent du modèle entraîné et du bruit du corpus NYT10.
