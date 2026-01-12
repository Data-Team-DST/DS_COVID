# page/08_cicd_pipeline.py
# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: CI pipeline presentation —
#   quality-focused, reproducible, deliberately non-industrial.

import streamlit as st
from streamlit_extras.colored_header import colored_header

# Imports Pipeline Sklearn
import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def run():
    # Header / hero
    colored_header(
        label="CI/CD et qualité logicielle",
        description=(
            "Présentation du pipeline CI réellement implémenté : "
            "objectifs, outils, limites assumées et positionnement pédagogique."
        ),
        color_name="blue-70"
    )
    st.divider()

    cicd_container = st.container(border=True)
    show_pipe_cont = st.container(border=True)
    with show_pipe_cont:
        st.subheader("Organisation du code et amélioration")
        st.info(
            "Pour résoudre les problématiques d’organisation, un système de pipelines "
            "Sklearn / Streamlit a été développé."
        )
        USE_PIPELINE_PAGE = st.checkbox(
            "Afficher la section Pipeline Sklearn",
            value=False
        )

    if USE_PIPELINE_PAGE:
        pipeline_container = st.container(border=True)

    with cicd_container:
        # 1. Why CI/CD here?
        st.markdown(
            "## 1. Pourquoi un pipeline CI/CD dans ce projet ?\n\n"
            "Même en l’absence de mise en production, un effort volontaire a été réalisé "
            "sur la **qualité du code**, la **reproductibilité** et la **maintenabilité**.\n\n"
            "L’objectif n’est pas l’industrialisation, mais d’éviter le code fragile, "
            "les régressions silencieuses et la dette technique."
        )
        st.divider()

        # 2. Implemented CI pipeline
        st.markdown(
            "## 2. Pipeline CI implémenté\n\n"
            "Le pipeline est exécuté automatiquement via GitHub Actions "
            "à chaque push et pull request."
        )
        st.text_area(
            "CI en place",
            value=(
                "- Linting Python avec pylint (seuil bloquant : score ≥ 8)\n"
                "- Tests unitaires avec pytest\n"
                "- Génération de rapports de couverture\n"
                "- Analyse statique centralisée via SonarCloud"
            ),
            height="stretch",
            key="cicd_core"
        )
        st.divider()

        # 3. Code quality & testing philosophy
        st.markdown(
            "## 3. Philosophie qualité et tests\n\n"
            "Les choix CI/CD reflètent une approche pragmatique et réaliste."
        )
        st.text_area(
            "Approche qualité",
            value=(
                "- Priorité à la lisibilité et à la robustesse du code\n"
                "- Tests unitaires ciblés sur les composants critiques\n"
                "- Couverture volontairement modeste mais contrôlée\n"
                "- Détection précoce des régressions"
            ),
            height="stretch",
            key="cicd_quality"
        )
        st.divider()

        # 4. Artefacts & traçabilité
        st.markdown(
            "## 4. Artefacts et traçabilité\n\n"
            "Le pipeline produit des éléments exploitables pour l’audit et le suivi."
        )
        st.text_area(
            "Artefacts générés",
            value=(
                "- Rapports de couverture pytest (coverage.xml)\n"
                "- Analyses SonarCloud\n"
                "- Logs GitHub Actions\n"
                "- Historique des exécutions CI"
            ),
            height="stretch",
            key="cicd_artifacts"
        )
        st.divider()

        # 5. What is intentionally missing
        st.markdown(
            "## 5. Ce qui est volontairement absent\n\n"
            "Certains éléments classiques de CI/CD ne sont pas implémentés, par choix."
        )
        st.text_area(
            "Limites assumées",
            value=(
                "- Pas de build Docker\n"
                "- Pas de déploiement continu (CD)\n"
                "- Pas d’orchestration Kubernetes\n"
                "- Pas de monitoring en temps réel\n\n"
                "Ces choix sont cohérents avec le périmètre académique, "
                "les contraintes de ressources et les objectifs pédagogiques."
            ),
            height="stretch",
            key="cicd_limits"
        )
        st.divider()

        # 6. Perspective & alignment
        st.markdown(
            "## 6. Perspectives et alignement avec la formation\n\n"
            "Le pipeline constitue une base saine pour des évolutions futures."
        )
        st.text_area(
            "Évolutions possibles",
            value=(
                "- Introduction progressive de Docker\n"
                "- Séparation claire entre CI et CD\n"
                "- Déploiement contrôlé en environnement de test\n"
                "- Monitoring basique des performances et des dérives"
            ),
            height="stretch",
            key="cicd_future"
        )
        st.divider()

        # 7. Final positioning
        st.markdown(
            "## 7. Positionnement final\n\n"
            "- Pipeline CI réel et fonctionnel, orienté qualité\n"
            "- Implémenté au-delà des exigences minimales\n"
            "- Adapté à un projet académique avancé\n"
            "- Base crédible pour une industrialisation future, sans sur-promesse"
        )

    if USE_PIPELINE_PAGE:
        st.divider()
        with pipeline_container:

            st.title("Exploration et exécution de pipelines Sklearn")

            # Ajouter le répertoire racine du projet au chemin Python
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Import des transformateurs
            # IMPORTANT : importer avant joblib pour que les classes soient disponibles lors du unpickling
            from src.features.St_Pipeline.Transformateurs import (
                ImagePathLoader,
                TupleToDataFrame,
                ImageResizer,
                ImageAugmenter,
                ImageNormalizer,
                ImageMasker,
                ImageFlattener,
                ImageRandomCropper,
                ImageStandardScaler,
                RGB_to_L,
                ImageAnalyser,
                ImagePCA,
                ImageHistogram,
                SaveTransformer,
                VisualizeTransformer,
                TrainTestSplitter,
            )

            # Importer joblib après les transformateurs
            import joblib

            # Configuration des chemins
            save_dir_paths = os.path.join(project_root, "models")
            data_dir = os.path.join(
                project_root,
                "data",
                "raw",
                "COVID-19_Radiography_Dataset",
                "COVID-19_Radiography_Dataset"
            )

            # Créer le dossier models s’il n’existe pas
            os.makedirs(save_dir_paths, exist_ok=True)
            st.header("Configuration")

            # Vérification du répertoire de données
            if os.path.exists(data_dir):
                st.success("Données trouvées")
                labels = [
                    d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d, "images"))
                ]
                st.info(f"Labels disponibles : {', '.join(labels)}")
            else:
                st.error("Répertoire de données introuvable")
                st.stop()

            # Mode de sélection
            mode = st.radio(
                "Mode de travail :",
                ["Charger un pipeline existant", "Créer un nouveau pipeline"],
                index=0
            )

            # MODE 1 : CHARGER UN PIPELINE EXISTANT
            if mode == "Charger un pipeline existant":
                pkl_files = [
                    f for f in os.listdir(save_dir_paths)
                    if f.endswith(".pkl")
                ]

                if not pkl_files:
                    st.warning(
                        f"Aucun pipeline trouvé dans {save_dir_paths}"
                    )
                    st.info(
                        "Créez un nouveau pipeline ou utilisez le notebook pour en générer."
                    )
                    st.stop()

                container_selection = st.container(border=True)
                with container_selection:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown("### Sélection")
                        selected_pipeline = st.selectbox(
                            "Pipeline enregistré :",
                            pkl_files,
                            index=(
                                pkl_files.index("Pipeline_ML_PCA_Complete.pkl")
                                if "Pipeline_ML_PCA_Complete.pkl" in pkl_files
                                else 0
                            )
                        )

                        st.success(selected_pipeline)
                        load_button = st.button(
                            "Charger et analyser",
                            width="stretch"
                        )

                    with col2:
                        st.markdown("### Détails du pipeline")

                        if load_button:
                            pipeline_path = os.path.join(
                                save_dir_paths,
                                selected_pipeline
                            )

                            try:
                                with st.spinner("Chargement du pipeline en cours"):
                                    loaded_pipeline = joblib.load(pipeline_path)

                                    def enable_streamlit_recursive(pipeline_obj):
                                        if isinstance(pipeline_obj, Pipeline):
                                            for _, transformer in pipeline_obj.steps:
                                                if isinstance(transformer, Pipeline):
                                                    enable_streamlit_recursive(transformer)
                                                elif hasattr(transformer, "use_streamlit"):
                                                    transformer.use_streamlit = True
                                        elif hasattr(pipeline_obj, "use_streamlit"):
                                            pipeline_obj.use_streamlit = True

                                    enable_streamlit_recursive(loaded_pipeline)

                                    st.session_state.loaded_pipeline = loaded_pipeline
                                    st.session_state.pipeline_name = selected_pipeline

                                st.success("Pipeline chargé avec succès")

                                st.markdown("**Étapes du pipeline :**")
                                for i, (name, transformer) in enumerate(
                                    loaded_pipeline.steps, 1
                                ):
                                    st.code(
                                        f"{i}. {name} : {transformer.__class__.__name__}"
                                    )

                            except Exception as e:
                                st.error(f"Erreur de chargement : {e}")
                                st.session_state.loaded_pipeline = None

            else:
                st.info(
                    "La création de pipeline personnalisé est disponible dans la suite du module."
                )
