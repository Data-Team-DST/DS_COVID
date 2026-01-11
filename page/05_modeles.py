# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark CSS.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: modelling lab template — compare models, visualise diagnostics, test sur images radiographiques.

import streamlit as st
from streamlit_extras.colored_header import colored_header
from PIL import Image
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
        label="Modèles de Machine learning et Optimisation",
        color_name="blue-70"
    )
    st.divider()


    st.markdown("""### **Extraction des caractéristiques**""")
    chemin_global = Path(__file__).parent.parent
    #rajouter /page/images/ au chemin global
    chemin_global = os.path.join(chemin_global, "page", "images/")

    p12 = rf"{chemin_global}prep_ML_features.PNG"
    image_12 = Path(p12).relative_to(Path.cwd())

    # p13 = rf"{chemin_global}features_PCA.PNG"
    # image_13 = Path(p13).relative_to(Path.cwd())

    # p14 = rf"{chemin_global}feature_histogram.PNG"
    # image_14 = Path(p14).relative_to(Path.cwd())

    # p15 = rf"{chemin_global}feature_combined.PNG"
    # image_15 = Path(p15).relative_to(Path.cwd())

    p16 = rf"{chemin_global}result_ml.PNG"
    image_16 = Path(p16).relative_to(Path.cwd())

    st.image(str(p12), caption="ML Feature Preparation", width=500)
    # st.image(str(p13), caption="PCA Feature Visualization", use_column_width=False)
    # st.image(str(p14), caption="Feature Histogram", use_column_width=False)
    # st.image(str(p15), caption="Combined Features", use_column_width=False)
    st.image(str(p16), caption="ML Results Overview", width=500)
    

    st.divider()

    st.markdown("""
    **Les Modèles de machine learning testés dans notre projet sont :**

    **• Support Vector Machine (SVM)**
                
    **• k-Nearest Neighbors (k-NN)**  
                
    **• Random Forest**
    """)

    st.info("**Pour évaluer les trois modèles de machine learning, un échantillon équilibré de 200 données par classe a été utilisé.**")
    st.info("""
    **L'évaluation repose  sur le calcule de la matrice de confusion, qui est un outil fondamental permettant de visualiser les performances d'un classificateur en croisant :**
    
    **Les prédictions du modèle (axes colonnes).**

    **Les réalités (axes lignes).**
    
    """)

    st.markdown("**Les matrices de confusion suivantes illustrent les performances des modèles évalués :**")

    MODELS = ["SVM", "KNN", "RF"]
    MATRICES_FOLDER = chemin_global / Path("matrices_confusion")

    # UN SEUL BLOC DE CONTRÔLES (ligne ~160)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: 
        choice1 = st.selectbox("Modèle :", ["all"] + MODELS, index=0, key="MATRICES_1")
    with col2: 
        n_images1 = st.number_input("N images / modèle :", 1, 2, 2, key="N1")
    with col3: 
        show_filenames1 = st.checkbox("Noms fichiers", value=True, key="NAMES1")

    if st.button("🔍 Matrices de confusion", key="LOAD1"): 
        
        sample_map = {}
        for model_name in MODELS:
            model_path = MATRICES_FOLDER / model_name
            if model_path.exists():
                img_files = list(model_path.glob("*.png")) + list(model_path.glob("*.jpg"))
                sample_map[model_name] = [
                    {"image": str(f)} for f in img_files[:n_images1]  
                ]
        st.session_state["sample_map"] = sample_map

    st.divider()

    # Affichage
    st.markdown("## 📈 Matrices de Confusion")
    sample_map = st.session_state.get("sample_map", {})
    if not sample_map: 
        st.info("👆 Cliquer sur  'Matrices de confusion'")
    else:
        total = 0
        targets = list(sample_map.keys()) if choice1 == "all" else [choice1]
        
        for model_name, entries in sample_map.items():
            if model_name not in targets: continue
                
            st.markdown(f"### {model_name} — {len(entries)} matrices")
            cols = st.columns(2)  # 2 colonnes car 2 images max
            
            for idx, entry in enumerate(entries):
                with cols[idx % 2]:  # 2 colonnes seulement
                    img_path = Path(entry["image"])
                    if img_path.exists():
                        im = Image.open(img_path).convert("RGB")
                        THUMBNAIL_MAX = (200, 200)
                        im.thumbnail(THUMBNAIL_MAX)
                        caption = img_path.name if show_filenames1 else f"Matrix {idx+1}"
                        st.image(im, caption=caption, width="content")
                    total += 1
        

    # chemin_absolu = rf"{chemin_global}/page/images/interprétation des résultats.png"
    # image_path = Path(chemin_absolu).relative_to(Path.cwd())
    # st.image(str(image_path), caption="Interprétation des résultats", use_column_width=True)
    
    st.markdown("### **Optimisation des modèles de machine learning**")
    
    st.markdown("""
    **Random Forest** : modèle le plus performant.
    
    **SVM** : modèle le moins performant.
    
    **Objectif de notre choix :**
    - Renforcer l'efficacité et améliorer les performancesdu de chaque modèle
    - Obtenir une analyse comparative complète
    """)
    
    st.divider()

    st.markdown("""
    ### **Grid Search**
    
    **Fonctionnement** : Grid Search teste toutes les combinaisons possibles d'hyperparamètres sur une "grille" prédéfinie.
                        
    
    **Évaluation** : Validation croisée (k-fold) évalue chaque combinaison de manière robuste en divisant les données en k splits.
    """)
    st.divider()

    st.markdown("### **Hyperparamètres testés**")

    st.markdown("""
    **SVM** : C (régularisation), max itérations
    
    **Random Forest** : nb arbres, profondeur max, critères division
    """)
   


    st.info("**Les différents hyperparamètres ont été sauvegardés dans un fichier.json**")

    chemin_absolu = rf"{chemin_global}parametres.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Les hyperparamètres", width="content")


    st.markdown("### **Résultats de la matrice de confusion pour les modèles SVM et Random Forest avec Grid Search**")

   
    choice2 = st.selectbox("Modèle :", ["all"] + MODELS)  
    
    MODELS = ["SVM", "KNN", "RF"]
    MATRICES_FOLDER = chemin_global / Path("matrices_confusion")

    # UN SEUL BLOC DE CONTRÔLES (ligne ~160)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: 
        choice2 = st.selectbox("Modèle :", ["all"] + MODELS, index=0, key="MATRICES_2")
    with col2: 
        n_images2 = st.number_input("N images / modèle :", 1, 2, 2, key="N2")
    with col3: 
        show_filenames2 = st.checkbox("Noms fichiers", value=True, key="NAMES2")

    if st.button("🔍 Matrices de confusion", key="LOAD2"): 
    
        sample_map = {}
        for model_name in MODELS:
            model_path = MATRICES_FOLDER / model_name
            if model_path.exists():
                img_files = list(model_path.glob("*.png")) + list(model_path.glob("*.jpg"))
                sample_map[model_name] = [
                    {"image": str(f)} for f in img_files[:n_images2]  # Max 2 images
                ]
        st.session_state["sample_map"] = sample_map
    
    st.divider()

    # Affichage
    st.markdown("## 📈 Matrices de Confusion")
    sample_map = st.session_state.get("sample_map", {})
    if not sample_map: 
        st.info("👆 Cliquer sur  'Matrices de confusion'")
    else:
        total = 0
        targets = list(sample_map.keys()) if choice2 == "all" else [choice2]
        
        for model_name, entries in sample_map.items():
            if model_name not in targets: continue
                
            st.markdown(f"### {model_name} — {len(entries)} matrices")
            cols = st.columns(2)  # 2 colonnes car 2 images max
            
            for idx, entry in enumerate(entries):
                with cols[idx % 2]:  # 2 colonnes seulement
                    img_path = Path(entry["image"])
                    if img_path.exists():
                        im = Image.open(img_path).convert("RGB")
                        THUMBNAIL_MAX = (200, 200)
                        im.thumbnail(THUMBNAIL_MAX)
                        caption = img_path.name if show_filenames2  else f"Matrix {idx+1}"
                        st.image(im, caption=caption, width="content")
                    total += 1

    st.info("**Les hyperparamètres optimaux identifiés par Grid Search ont permis daméliorer significativement les performances de base des modèles.**")


    st.divider()



if __name__ == "__main__":
    run()