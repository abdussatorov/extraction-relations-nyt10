import os
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import joblib
import spacy
from relation_extraction_core import extract_triples

st.set_page_config(page_title="Relation Extraction | NYT10", page_icon="🔗", layout="wide")

@st.cache_resource
def load_resources():
    if not os.path.exists("artifacts/relation_model.joblib"):
        raise FileNotFoundError(
            "Modèle absent. Exécute d'abord les cellules du notebook jusqu'à l'entraînement final "
            "et la cellule d'export du modèle."
        )
    model = joblib.load("artifacts/relation_model.joblib")
    nlp = spacy.load("en_core_web_sm")
    return model, nlp

st.title("🔗 Extraction de relations entre entités")
st.caption("Projet 10 — NYT10 • NER + dépendances syntaxiques + classifieur • Graphe de connaissances")

with st.sidebar:
    st.header("⚙️ Paramètres")
    threshold = st.slider("Seuil de confiance", 0.10, 0.95, 0.50, 0.05)
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("Texte → spaCy / NER → caractéristiques syntaxiques → modèle → triplets → graphe")
    st.info("Le modèle traite du texte anglais, comme le corpus NYT10.")

examples = {
    "Steve Jobs / Apple": "Steve Jobs founded Apple in Cupertino.",
    "Business relations": "Tim Cook works for Apple. Jeff Bezos founded Amazon in Seattle.",
    "Geography": "Paris is the capital of France. Rabat is the capital of Morocco."
}
choice = st.selectbox("Exemple rapide", ["Écrire mon texte…"] + list(examples.keys()))
default_text = "" if choice == "Écrire mon texte…" else examples[choice]
text_input = st.text_area(
    "Texte à analyser (en anglais)", value=default_text, height=150,
    placeholder="Ex. Steve Jobs founded Apple in Cupertino."
)

run = st.button("🚀 Extraire les relations", type="primary", use_container_width=False)

if run:
    if not text_input.strip():
        st.warning("Saisis d'abord un texte.")
    else:
        try:
            model, nlp = load_resources()
            with st.spinner("Analyse du texte et prédiction des relations…"):
                triples, doc = extract_triples(text_input.strip(), nlp, model, threshold)
            left, right = st.columns([1, 1])
            with left:
                st.subheader("1. Entités détectées (NER)")
                entities = [{"Entité": e.text, "Type": e.label_} for e in doc.ents]
                if entities:
                    st.dataframe(pd.DataFrame(entities), use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune entité reconnue par spaCy dans ce texte.")
            with right:
                st.subheader("2. Résultat de l'extraction")
                if triples:
                    result_df = pd.DataFrame(triples)[["h", "relation", "t", "proba", "phrase"]]
                    result_df.columns = ["Entité 1", "Relation", "Entité 2", "Confiance", "Phrase source"]
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Télécharger les triplets (CSV)",
                        result_df.to_csv(index=False).encode("utf-8"),
                        file_name="triplets_extraits.csv", mime="text/csv"
                    )
                else:
                    st.warning("Aucune relation n'a dépassé ce seuil. Essaie un seuil plus bas ou un texte différent.")

            if triples:
                st.subheader("3. Graphe de connaissances")
                graph = nx.MultiDiGraph()
                for triple in triples:
                    graph.add_edge(triple["h"], triple["t"],
                                   relation=triple["relation"], proba=triple["proba"])
                fig, ax = plt.subplots(figsize=(12, 6))
                pos = nx.spring_layout(graph, seed=42)
                nx.draw_networkx_nodes(graph, pos, node_size=2200, ax=ax)
                nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
                nx.draw_networkx_edges(
                    graph, pos, arrows=True, arrowsize=18, node_size=2200,
                    connectionstyle="arc3,rad=0.12", ax=ax
                )
                labels = {(u, v): d["relation"] for u, v, d in graph.edges(data=True)}
                nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8, ax=ax)
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.metric("Triplets extraits", len(triples))
                st.metric("Entités dans le graphe", graph.number_of_nodes())
                st.metric("Relations dans le graphe", graph.number_of_edges())
        except FileNotFoundError as e:
            st.error(str(e))
        except OSError as e:
            st.error(f"Impossible de charger spaCy ou le modèle : {e}")
        except Exception as e:
            st.error(f"Une erreur est survenue pendant l'analyse : {e}")
            st.caption("Vérifie que le notebook a été exécuté jusqu'à l'entraînement et à l'export du modèle.")

st.markdown("---")
st.caption("Projet académique — les relations prédites sont des estimations du modèle, pas des faits garantis.")
