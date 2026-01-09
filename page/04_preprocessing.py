# 04_preprocessing.py
# Theming metadata:
# - Preferred: streamlit-extras mandatory; page inherits app-level dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: expert-grade preprocessing template focusing on "description et justification des traitements" — placeholders, checklists, pseudo-code et exigences CI.

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
    st.set_page_config(page_title="Preprocessing — Description & Justification", layout="wide")
    colored_header(
        label="Preprocessing — Description & Justification",
        description="Documenter chaque transformation : quoi, pourquoi, comment, impact et tests associés.",
        color_name="blue-70"
    )
    st.divider()

    chemin_global = Path(__file__).parent.parent
    #rajouter /page/images/ au chemin global
    chemin_global = os.path.join(chemin_global, "page", "images/")

    p1 = rf"{chemin_global}covid_before_mask.png"
    image_1 = Path(p1).relative_to(Path.cwd())

    p2 = rf"{chemin_global}covid_after_mask.png"
    image_2 = Path(p2).relative_to(Path.cwd())

    p3 = rf"{chemin_global}arbo_augmented.png"
    image_3 = Path(p3).relative_to(Path.cwd())

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



    

    st.write(f"Chemin global des images : {chemin_global}")

    st.info("**Cette page documente chaque transformation appliquée aux données.**")

    st.image(str(p1), caption="Before Masking", use_column_width=False)
    st.image(str(p2), caption="After Masking", use_column_width=False)
    st.image(str(p3), caption="Data Augmentation Example", use_column_width=False)
    st.image(str(p4), caption="Class Distribution Imbalance", use_column_width=False)
    st.image(str(p5), caption="Dataset Configuration", use_column_width=False)
    st.image(str(p6), caption="WSL Configuration", use_column_width=False)
    st.image(str(p7), caption="Rebalancing Strategy", use_column_width=False)
    st.image(str(p8), caption="Stratified Split", use_column_width=False)
    st.image(str(p9), caption="Data Augmentation Configuration", use_column_width=False)
    st.image(str(p10), caption="Augmentation Visualization", use_column_width=False)

    