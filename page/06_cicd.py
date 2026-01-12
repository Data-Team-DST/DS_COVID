# 08_cicd_pipeline.py — Présentation CI/CD pédagogique

import streamlit as st
from streamlit_extras.colored_header import colored_header
import os, sys, joblib
from sklearn.pipeline import Pipeline

def run():
    # Header
    colored_header(
        label="CI/CD et qualité logicielle",
        description=(
            "Présentation du pipeline CI réel : objectifs, outils, limites pédagogiques, "
            "et positionnement académique."
        ),
        color_name="blue-70"
    )
    st.divider()

    # --- Section 1 : Pourquoi CI/CD ---
    st.markdown("## 1. Pourquoi un pipeline CI/CD ?")
    st.markdown(
        "Même sans déploiement industriel, nous avons mis en place des pratiques de **qualité, "
        "reproductibilité et maintenabilité**. L’objectif est d’éviter le code fragile, "
        "les régressions et la dette technique."
    )
    st.divider()

    # --- Section 2 : Pipeline CI implémenté ---
    st.markdown("## 2. Pipeline CI")
    st.markdown(
        "Exécuté automatiquement via GitHub Actions à chaque push/PR :"
    )
    st.markdown(
        "- Linting Python avec pylint (score ≥ 8)\n"
        "- Tests unitaires avec pytest\n"
        "- Rapports de couverture\n"
        "- Analyse statique SonarCloud"
    )
    st.divider()

    # --- Section 3 : Philosophie qualité ---
    st.markdown("## 3. Philosophie qualité & tests")
    st.markdown(
        "Approche pragmatique : priorité à la lisibilité, robustesse et détection précoce des régressions. "
        "Tests ciblés sur les composants critiques, couverture volontairement modeste mais contrôlée."
    )
    st.divider()

    # --- Section 4 : Artefacts et traçabilité ---
    st.markdown("## 4. Artefacts produits")
    st.markdown(
        "- Rapports pytest (coverage.xml)\n"
        "- Analyses SonarCloud\n"
        "- Logs GitHub Actions\n"
        "- Historique des exécutions CI"
    )
    st.divider()

    # --- Section 5 : Limites volontaires ---
    st.markdown("## 5. Limites assumées")
    st.markdown(
        "Certains éléments classiques ne sont pas implémentés par choix pédagogique :\n"
        "- Pas de build Docker\n"
        "- Pas de CD / déploiement continu\n"
        "- Pas d’orchestration Kubernetes\n"
        "- Pas de monitoring temps réel\n\n"
        "→ Cohérent avec le périmètre académique et les ressources disponibles."
    )
    st.divider()

    # --- Section 6 : Perspectives ---
    st.markdown("## 6. Perspectives")
    st.markdown(
        "- Introduction progressive de Docker\n"
        "- Séparation CI / CD\n"
        "- Déploiement contrôlé en test\n"
        "- Monitoring basique performances/dérives"
    )
    st.divider()

    # --- Section 7 : Positionnement final ---
    st.markdown("## 7. Positionnement final")
    st.markdown(
        "- Pipeline CI réel, orienté qualité\n"
        "- Au-delà des exigences minimales\n"
        "- Adapté à un projet académique avancé\n"
        "- Base crédible pour industrialisation future"
    )
    st.divider()

    # --- Section 8 : Pipeline Sklearn ---
    st.subheader("Exploration des pipelines Sklearn")
    st.markdown(
        "Cette partie permet de charger un pipeline existant ou d’en créer un nouveau. "
        "Aucune modification du pipeline n’est nécessaire pour la présentation."
    )

    # Configuration des chemins
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path: sys.path.insert(0, project_root)
    save_dir_paths = os.path.join(project_root, "models")
    data_dir = os.path.join(project_root, "data", "raw", "COVID-19_Radiography_Dataset", "COVID-19_Radiography_Dataset")
    os.makedirs(save_dir_paths, exist_ok=True)

    # Vérification des données
    if not os.path.exists(data_dir):
        st.error("Répertoire de données introuvable")
        st.stop()
    else:
        labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d, "images"))]
        st.info(f"Labels disponibles : {', '.join(labels)}")

    # Mode de travail
    mode = st.radio("Mode :", ["Charger un pipeline existant", "Créer un nouveau pipeline"], index=0)

    if mode == "Charger un pipeline existant":
        pkl_files = [f for f in os.listdir(save_dir_paths) if f.endswith(".pkl")]
        if not pkl_files:
            st.warning("Aucun pipeline trouvé.")
            st.stop()
        selected_pipeline = st.selectbox("Pipeline enregistré :", pkl_files, index=(pkl_files.index("Pipeline_ML_PCA_Complete.pkl") if "Pipeline_ML_PCA_Complete.pkl" in pkl_files else 0))
        if st.button("Charger et analyser"):
            pipeline_path = os.path.join(save_dir_paths, selected_pipeline)
            try:
                loaded_pipeline = joblib.load(pipeline_path)
                st.session_state.loaded_pipeline = loaded_pipeline
                st.session_state.pipeline_name = selected_pipeline
                st.success("Pipeline chargé avec succès")
                st.markdown("**Étapes du pipeline :**")
                for i, (name, transformer) in enumerate(loaded_pipeline.steps, 1):
                    st.code(f"{i}. {name} : {transformer.__class__.__name__}")
            except Exception as e:
                st.error(f"Erreur de chargement : {e}")
    else:
        st.info("Création d’un pipeline personnalisé disponible dans la suite du module.")
