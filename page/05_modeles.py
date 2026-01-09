# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark CSS.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: modelling lab template — compare models, visualise diagnostics, test sur images radiographiques.

import streamlit as st
from streamlit_extras.colored_header import colored_header
from pathlib import Path
import os

# Optional: placeholder for torch/keras if models are deep learning
# import torch
# from tensorflow import keras
# from PIL import Image
# import numpy as np

# Structure comments:
# - run() only. Provide model comparison, diagnostics, and direct inference on sample images.
# - Placeholders ready to link real DL models.

def run():
    # Header / hero
    colored_header(
        label="Modèles de Machine learning et de deep learning",
        color_name="blue-70"
    )
    st.divider()

    chemin_global = Path(__file__).parent.parent
    #rajouter /page/images/ au chemin global
    chemin_global = os.path.join(chemin_global, "page", "images/")

    p12 = rf"{chemin_global}prep_ML_features.PNG"
    image_12 = Path(p12).relative_to(Path.cwd())

    p13 = rf"{chemin_global}features_PCA.PNG"
    image_13 = Path(p13).relative_to(Path.cwd())

    p14 = rf"{chemin_global}feature_histogram.PNG"
    image_14 = Path(p14).relative_to(Path.cwd())

    p15 = rf"{chemin_global}feature_combined.PNG"
    image_15 = Path(p15).relative_to(Path.cwd())

    p16 = rf"{chemin_global}result_ml.PNG"
    image_16 = Path(p16).relative_to(Path.cwd())

    st.image(str(p12), caption="ML Feature Preparation", use_column_width=False)
    st.image(str(p13), caption="PCA Feature Visualization", use_column_width=False)
    st.image(str(p14), caption="Feature Histogram", use_column_width=False)
    st.image(str(p15), caption="Combined Features", use_column_width=False)
    st.image(str(p16), caption="ML Results Overview", use_column_width=False)
    

    st.divider()

    chemin_global=Path(__file__).parent.parent
    st.markdown("""**Modèles de machine learning** :""")
    st.markdown("""**Support Vector Machine (SVM)** :""")
    chemin_absolu = rf"{chemin_global}/page/images/SVM.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Support Vector Machine (SVM)", use_column_width=True)
    st.markdown("""**k-Nearest Neighbors (k-NN)** :""")
    chemin_absolu = rf"{chemin_global}/page/images/KNN.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="k-Nearest Neighbors (k-NN)", use_column_width=True)
    st.markdown("""**Random Forest** :""")
    chemin_absolu = rf"{chemin_global}/page/images/random forest.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Random Forest", use_column_width=True)
    st.info("""**Pour évaluer les trois modèles de machine learning, un échantillon équilibré de 200 données par classe a été utilisé.**""")
    st.info("""**L'évaluation s'est appuyée sur la matrice de confusion, qui est un outil fondamental permettant de visualiser les performances d'un classificateur en croisant :**""")
    st.info("""**Les prédictions du modèle (axes colonnes).**""")
    st.info("""**Les réalités (axes lignes).**""")
    chemin_absolu = rf"{chemin_global}/page/images/résultat_obtenu.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Matrice de confusion", use_column_width=True)
    chemin_absolu = rf"{chemin_global}/page/images/interprétation des résultats.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Interprétation des résultats", use_column_width=True)
    st.markdown("""**Modèle de deep learning Inception V3** :""")
    chemin_absolu = rf"{chemin_global}/page/images/Inceptionv3.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Schéma explicatif", use_column_width=True)
    chemin_absolu = rf"{chemin_global}/page/images/courbe de loss et d'accuracy.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Courbe de loss et d'accuracy", use_column_width=True)
    st.info("""**Courbe de loss d’entraînement : constante et très faible → optimisation maîtrisée sur données apprises**""")
    st.info("""**Courbe de loss de validation : fluctuations + pic important à l’époque 8 (correspond à la baisse de précision)**""")
    st.info("""**Après le pic : courbe retrouve sa tendance initiale**""")
    st.info("""**Explications possibles : bruit dans les données**""")
    st.info("""**Précision d’entraînement : très élevée (0.98-1.00) tout au long → apprentissage efficace sur données d’entraînement**""")
    st.info("""**Précision de validation : stable ∼0.90 + fluctuations légères + chute nette époque 8 → retour rapide aux valeurs élevées**""")
    st.info("""**Différence train/validation : signe de difficulté de généralisation sur certains batchs de validation**""")
    st.info("""**Pic de baisse : instabilité temporaire liée à la répartition des données**""")
    chemin_absolu = rf"{chemin_global}/page/images/matrice de confusion.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Matrice de confusion", use_column_width=True)
    st.info("""**Performance globale : modèle très performant, bonne identification des classes (COVID,pneumonie virale, normal)**""")
    st.info("""**Efficacité prouvée : transfert d’apprentissage avec InceptionV3 excellent pour classification d’images médicales**""")


if __name__ == "__main__":
    run()