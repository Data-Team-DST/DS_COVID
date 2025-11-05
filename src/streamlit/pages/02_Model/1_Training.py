import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))  # Ajouter le dossier streamlit

# Calcul correct du project_root
current_file = Path(__file__)  # pages/02_Model/1_Training.py
current_dir = current_file.parent  # pages/02_Model/
streamlit_dir = current_dir.parent.parent  # streamlit/
src_dir = streamlit_dir.parent  # src/
project_root = src_dir.parent  # DS_COVID/

sys.path.append(str(project_root))

try:
    from src.features.Pipelines.Pipeline_Sklearn import PipelineManager
except ImportError:
    st.error("❌ Impossible d'importer PipelineManager. Vérifiez le chemin d'accès.")

st.title("🏋️ Entraînement du Modèle")

st.markdown(
    """
Cette page permet d'entraîner différents modèles de classification pour la détection COVID-19 
en utilisant des pipelines sklearn configurables.
"""
)

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Configuration")

    # Sélection du type de données
    data_source = st.selectbox(
        "Source des données",
        ["Données simulées (test)", "Dataset COVID-19", "Upload personnalisé"],
    )

    # Paramètres généraux
    test_size = st.slider("Taille du jeu de test (%)", 10, 50, 20) / 100
    random_state = st.number_input("Seed aléatoire", 1, 9999, 42)

# Interface principale
tab1, tab2, tab3 = st.tabs(["� Configuration", "🚀 Entraînement", "📊 Résultats"])

with tab1:
    st.header("Configuration des Pipelines")

    # Charger le gestionnaire de pipelines
    try:
        # Construire le chemin vers le fichier de configuration
        config_path = (
            project_root
            / "src"
            / "features"
            / "Pipelines"
            / "Pipeline_Sklearn_config.json"
        )

        if not config_path.exists():
            st.error(f"❌ Fichier de configuration non trouvé: {config_path}")
            st.info(
                "🔧 Vérifiez que le fichier Pipeline_Sklearn_config.json existe dans src/features/Pipelines/"
            )
            manager = None
        else:
            manager = PipelineManager(str(config_path))

        if manager is None:
            st.error("❌ Impossible de charger le gestionnaire de pipelines")
            selected_configs = []
            configs = []
        else:
            # Afficher les configurations disponibles
            st.subheader("📋 Pipelines Disponibles")
            configs = manager.get_available_configs()

            config_df = pd.DataFrame(configs)
            st.dataframe(config_df, use_container_width=True)

            # Sélection des pipelines à entraîner
            st.subheader("🎯 Sélection des Pipelines")
            selected_configs = st.multiselect(
                "Choisissez les pipelines à entraîner:",
                options=[config["name"] for config in configs],
                default=["basic_rf", "fast_prototype"] if configs else [],
                help="Sélectionnez un ou plusieurs pipelines pour comparaison",
            )

        if selected_configs and configs:
            st.success(f"✅ {len(selected_configs)} pipeline(s) sélectionné(s)")

            # Afficher les détails des pipelines sélectionnés
            with st.expander("🔍 Détails des pipelines sélectionnés"):
                for config_name in selected_configs:
                    config_details = next(
                        (c for c in configs if c["name"] == config_name), None
                    )
                    if config_details:
                        st.write(f"**{config_name}**: {config_details['description']}")
                        st.write(
                            f"- GridSearch: {'✅' if config_details['grid_search'] else '❌'}"
                        )
                        st.write(f"- CV Folds: {config_details['cv_folds']}")
        elif selected_configs and not configs:
            st.warning("⚠️ Pipelines sélectionnés mais configuration non chargée")

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la configuration: {e}")
        selected_configs = []

with tab2:
    st.header("Entraînement des Modèles")

    if not selected_configs:
        st.warning(
            "⚠️ Veuillez sélectionner au moins un pipeline dans l'onglet Configuration"
        )
    else:
        # Paramètres d'entraînement
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Données")
            if data_source == "Données simulées (test)":
                n_samples = st.slider("Nombre d'échantillons", 100, 5000, 1000)
                n_features = st.selectbox(
                    "Taille des features", [1024, 4096, 16384], index=2
                )
                n_classes = st.selectbox("Nombre de classes", [2, 4], index=1)

        with col2:
            st.subheader("🎛️ Paramètres")
            enable_comparison = st.checkbox("Activer la comparaison", True)
            save_results = st.checkbox("Sauvegarder les résultats", True)

        # Bouton d'entraînement
        if st.button("🚀 Lancer l'Entraînement", type="primary"):
            # Générer les données selon la source
            if data_source == "Données simulées (test)":
                with st.spinner("🔄 Génération des données simulées..."):
                    np.random.seed(random_state)
                    X = np.random.rand(n_samples, n_features)
                    y = np.random.randint(0, n_classes, n_samples)

                    # Division train/test
                    split_idx = int((1 - test_size) * n_samples)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]

                    st.success(
                        f"✅ Données générées: {X_train.shape[0]} train, {X_test.shape[0]} test"
                    )

                # Entraînement des pipelines
                results_container = st.container()
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = {}

                for i, config_name in enumerate(selected_configs):
                    status_text.text(f"🏋️ Entraînement de {config_name}...")
                    progress_bar.progress((i) / len(selected_configs))

                    # Container pour les résultats de ce pipeline
                    with results_container:
                        with st.expander(
                            f"📈 Résultats - {config_name}", expanded=True
                        ):
                            result_placeholder = st.empty()

                            try:
                                # Capture des logs avec redirection
                                start_time = time.time()

                                # Entraînement
                                result = manager.train_pipeline(
                                    config_name, X_train, y_train, X_test, y_test
                                )

                                end_time = time.time()
                                training_duration = end_time - start_time

                                results[config_name] = result

                                # Affichage des résultats
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(
                                        "Précision Test",
                                        f"{result.get('test_accuracy', 0):.3f}",
                                        delta=None,
                                    )

                                with col2:
                                    st.metric(
                                        "F1-Score",
                                        f"{result.get('test_f1', 0):.3f}",
                                        delta=None,
                                    )

                                with col3:
                                    st.metric(
                                        "Score CV",
                                        f"{result.get('cv_mean', 0):.3f}",
                                        delta=f"±{result.get('cv_std', 0):.3f}",
                                    )

                                # Détails supplémentaires
                                if "best_params" in result:
                                    st.write("🔧 **Meilleurs paramètres:**")
                                    st.json(result["best_params"])

                                st.success(
                                    f"✅ {config_name} terminé en {training_duration:.1f}s"
                                )

                            except Exception as e:
                                st.error(f"❌ Erreur avec {config_name}: {str(e)}")

                # Finalisation
                progress_bar.progress(1.0)
                status_text.text("✅ Entraînement terminé!")

                # Sauvegarde des résultats dans la session
                st.session_state["training_results"] = results
                st.session_state["comparison_data"] = {
                    "configs": selected_configs,
                    "data_info": {
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_classes": n_classes,
                        "test_size": test_size,
                    },
                }

            else:
                st.info("🚧 Chargement de données réelles non encore implémenté")

with tab3:
    st.header("Comparaison des Résultats")

    if "training_results" in st.session_state:
        results = st.session_state["training_results"]

        if results:
            # Tableau de comparaison
            st.subheader("📊 Tableau Comparatif")

            comparison_data = []
            for config_name, result in results.items():
                row = {
                    "Pipeline": config_name,
                    "Précision Test": f"{result.get('test_accuracy', 0):.4f}",
                    "F1-Score": f"{result.get('test_f1', 0):.4f}",
                    "Score CV": f"{result.get('cv_mean', 0):.4f}",
                    "Écart-type CV": f"{result.get('cv_std', 0):.4f}",
                    "Temps (s)": f"{result['training_time'].total_seconds():.1f}",
                }

                if "best_score" in result:
                    row["Meilleur GridSearch"] = f"{result['best_score']:.4f}"

                comparison_data.append(row)

            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)

            # Meilleur modèle
            best_model = max(
                results.items(), key=lambda x: x[1].get("test_accuracy", 0)
            )
            st.success(
                f"🏆 **Meilleur modèle**: {best_model[0]} avec une précision de {best_model[1].get('test_accuracy', 0):.4f}"
            )

            # Graphiques de comparaison
            st.subheader("📈 Visualisations")

            col1, col2 = st.columns(2)

            with col1:
                # Graphique en barres des précisions
                metrics_df = pd.DataFrame(
                    {
                        "Pipeline": [r["Pipeline"] for r in comparison_data],
                        "Précision": [
                            float(r["Précision Test"]) for r in comparison_data
                        ],
                    }
                )
                st.bar_chart(metrics_df.set_index("Pipeline"))

            with col2:
                # Temps d'entraînement
                time_df = pd.DataFrame(
                    {
                        "Pipeline": [r["Pipeline"] for r in comparison_data],
                        "Temps": [float(r["Temps (s)"]) for r in comparison_data],
                    }
                )
                st.bar_chart(time_df.set_index("Pipeline"))

            # Bouton de téléchargement des résultats
            csv_data = df_comparison.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv_data,
                file_name=f"covid_pipeline_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "ℹ️ Aucun résultat d'entraînement disponible. Lancez d'abord un entraînement dans l'onglet précédent."
        )
