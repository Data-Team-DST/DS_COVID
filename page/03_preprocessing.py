# 03_preprocessing.py

import streamlit as st
from streamlit_extras.colored_header import colored_header

# Structure comments:
# - Expose only run().
# - Cette page documente chaque transformation appliquée (quoi, pourquoi, comment, impact, tests).
# - Tous les blocs sont placeholders/lorem à remplacer par des descriptions projet-réelles.

# Imports systèmes
import subprocess
import os
import sys
from pathlib import Path


def run():
    # Configuration de la page
    st.set_page_config(page_title="Preprocessing — Description & Justification", layout="wide")
    # colored_header(
    #     label="Preprocessing — Description & Justification",
    #     description="Documenter chaque transformation : quoi, pourquoi, comment, impact et tests associés.",
    #     color_name="blue-70"
    # )

    st.divider()

    # Obtenir le chemin global du projet
    chemin_global = Path(__file__).parent.parent
    #rajouter /page/images/ au chemin global
    chemin_global = os.path.join(chemin_global, "page", "images/")


    

    p4 = rf"{chemin_global}distri_desquilibre.PNG"
    image_4 = Path(p4).relative_to(Path.cwd())

    p5 = rf"{chemin_global}config_dataset.PNG"
    image_5 = Path(p5).relative_to(Path.cwd())

    p6 = rf"{chemin_global}config_wsl.PNG"
    image_6 = Path(p6).relative_to(Path.cwd())

    p7 = rf"{chemin_global}strategie_reequilibrage.PNG"
    image_7 = Path(p7).relative_to(Path.cwd())

    p8 = rf"{chemin_global}split_stratifie.PNG"
    image_8 = Path(p8).relative_to(Path.cwd())

    p9 = rf"{chemin_global}config_data_augmentation.PNG"
    image_9 = Path(p9).relative_to(Path.cwd())
    
    p10 = rf"{chemin_global}visu_augmentation.PNG"
    image_10 = Path(p10).relative_to(Path.cwd())

    Config_container = st.container(border=True)
    with Config_container:
        st.title("Environnements de travail")

        p11 = rf"{chemin_global}config_auto_env.png"
        windows_col, wsl_col, collab_col = st.columns(3)
        
        with windows_col:
            windows_cont = st.container(border=True)
        with wsl_col:
            wsl_cont = st.container(border=True)         
        with collab_col:
            collab_cont = st.container(border=True)

        with windows_cont:
            st.header("Windows", text_alignment="center")
            st.success(" Environnement Simple")
            st.info(" Données stockées localement")
            st.error(" Pas de compatibilité avec GPU")

        with wsl_cont:
            st.header("WSL", text_alignment="center")
            st.warning(" Configuration plus complexe, Cuda à installer")
            st.info(" Données stockées localement")
            st.warning(" Compatibilité avec GPU, Carte Nvidia")

        with collab_cont:
            st.header("Google Colab", text_alignment="center")
            st.error(" Environnement le plus complexe")
            st.warning(" Données stockées sur le cloud (Drive)")
            st.success(" Compatibilité avec GPU/TPU")

        config_auto_container = st.container(border=True)
        with config_auto_container:
            config_col1, config_col2 = st.columns([0.5,0.5],gap="small")
            with config_col1:
                config_1_container = st.container(border=True)
                with config_1_container:
                    st.header("Configuration automatique", text_alignment="center")
                    st.write(" Script pour détecter et configurer automatiquement l'environnement de travail.")
                    st.image(p11, caption="Configuration Automatique")
            with config_col2:
                config_2_container = st.container(border=True)
                with config_2_container:
                    st.header("Configuration WSL", text_alignment="center")
                    st.write(" Étapes pour configurer l'environnement WSL pour le projet.")
                    st.image(str(p6), caption="WSL Configuration")

    st.divider()

    # Masking images
    p1 = rf"{chemin_global}covid_before_mask.png"
    p2 = rf"{chemin_global}covid_after_mask.png"
    p3 = rf"{chemin_global}arbo_augmented.png"
    

    Masking_container = st.container(border=True)

    with Masking_container:
        st.title("Masking des images")
        col_1, col_2 = st.columns([0.2,0.2],gap="small")

        with col_1:
            sub_col_1, sub_col_2 = st.columns(2)

            with sub_col_1:
                st.image(str(p1), caption="Before Masking")

            with sub_col_2:
                st.image(str(p2), caption="After Masking")

            st.info("Quoi ? Masque pour isoler la région d'intérêt")
            st.info("Pourquoi ? Réduire le bruit et améliorer la qualité des données en se concentrant sur les zones pertinentes")
            st.info("Comment ? Utilisation d'algorithmes de segmentation pour créer et appliquer le masque")
            st.info("Impact ? Amélioration potentielle des performances du modèle en réduisant les distractions")
            st.info("Tests associés ? Comparaison des performances du modèle avant et après l'application du masque")

        with col_2:
            st.image(str(p3), caption="Arborescence automatique")

        # with col_3:
        #     st.image(str(p2), caption="After Masking")
            
        # with col_4:
            

    st.divider()

    # Gestion du déséquilibre de classes


    desequilibre_container = st.container(border=True)
    with desequilibre_container:
        st.title("Gestion du déséquilibre de classes")
        
        des_col_1, des_col_2 = st.columns([0.3,0.3],gap="small")
        with des_col_1:
            reeq_container = st.container(border=True,horizontal_alignment="center",height="stretch")
            with reeq_container:
                st.title("Stratégie de rééquilibrage",text_alignment="center")
                st.image(p7, caption="Rebalancing Strategy")

        with des_col_2:
            split_strat_container = st.container(border=True,horizontal_alignment="center",height="stretch")
            with split_strat_container:
                st.title("Stratified Split",text_alignment="center")
                st.image(p8, caption="Stratified Split")

        dataset_config_cont = st.container(border=True,horizontal_alignment="center")
        with dataset_config_cont:
            st.title("Configuration du Dataset",text_alignment="center")
            st.image(p5, caption="Dataset Configuration",)
            

    st.divider()

    # Data Augmentation
    augmentation_container = st.container(border=True)
    with augmentation_container:
        st.title("Data Augmentation")
        aug_col_1, aug_col_2 = st.columns([0.5,0.5],gap="small")
        with aug_col_1:
            st.image(str(p9), caption="Data Augmentation Configuration")
        with aug_col_2:
            st.image(str(p10), caption="Augmentation Visualization")
    
