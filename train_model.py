"""Entraîne le modèle à partir de features.pkl et l'exporte vers artifacts/relation_model.joblib
Usage (dans le dossier du projet) :  python train_model.py
"""
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

if not os.path.exists("features.pkl"):
    raise SystemExit("features.pkl introuvable : lance ce script dans le dossier du projet.")

print("Lecture de features.pkl ...")
feat = pd.read_pickle("features.pkl")
print(f"{len(feat)} exemples, {feat['relation'].nunique()} classes")

prep = ColumnTransformer([
    ("between", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "between"),
    ("path", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=False), "path"),
    ("types", OneHotEncoder(handle_unknown="ignore"), ["h_type", "t_type", "type_pair", "order"]),
    ("num", MaxAbsScaler(), ["dist", "path_len"]),
])
model = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(C=8, max_iter=3000, class_weight="balanced")),
])

print("Entraînement (1 à 3 minutes) ...")
model.fit(feat, feat["relation"].values)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/relation_model.joblib")
print("OK ! Modèle exporté vers artifacts/relation_model.joblib")
print("Classes :", list(model.classes_))
