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

    chemin_global = Path(__file__).parent.parent
    #rajouter /page/images/ au chemin global
    chemin_global = os.path.join(chemin_global, "page", "images/")

    # Extraction des caractéristiques
    extract_container = st.container(border=True)
    with extract_container:
        st.title("**Extraction des caractéristiques**")
        
        ext_col1, ext_col2 = st.columns([0.5,0.5],gap="small")
        with ext_col1:
            ext_cont1 = st.container(border=True, height="stretch")
            with ext_cont1:
                st.subheader("ML Feature Preparation",text_alignment="center")
                p12 = rf"{chemin_global}prep_ML_features.PNG"
                st.image(str(p12), caption="ML Feature Preparation", width="stretch")

        with ext_col2:
            ext_cont2 = st.container(border=True, height="stretch")
            with ext_cont2:
                st.subheader("ML Results Overview",text_alignment="center")
                p16 = rf"{chemin_global}result_ml.PNG"
                st.image(str(p16), caption="ML Results Overview", width="stretch")
        

    st.divider()

    #Evaluation des modèles
    eval_container = st.container(border=True)
    with eval_container:
        st.title("Évaluation des modèles de machine learning")

        # Intro
        intro_cont = st.container(border=True)
        with intro_cont:
            st.subheader("**Les Modèles de machine learning testés dans notre projet :**")
            st.info("**• Support Vector Machine (SVM)**")
            st.info("**• k-Nearest Neighbors (k-NN)**")
            st.info("**• Random Forest**")
        
        # Methode d'évaluation
        cont_method_eval = st.container(border=True, height="stretch")
        with cont_method_eval:
            st.subheader("**Méthodologie d'évaluation des modèles**")
            st.info("**Pour évaluer les trois modèles de machine learning, un échantillon équilibré de 200 données par classe a été utilisé.**")
            st.info("""
        **L'évaluation repose  sur le calcul de la matrice de confusion, qui est un outil fondamental permettant de visualiser les performances d'un classificateur en croisant :**
        
        **Les prédictions du modèle (axes colonnes).**

        **Les réalités (axes lignes).**
        
        """)
      
        select_matix_col1, select_matix_col2 = st.columns([0.5,0.5],gap="small")
        with select_matix_col1:
            cont_image_display = st.container(border=True, height="stretch")  
        with select_matix_col2:  
            cont_matric_selection = st.container(border=True, height="stretch")
    
    #Conteneur de sélection des matrices de confusion
    with cont_matric_selection:
        st.subheader("Sélection :")
        st.markdown("**Les matrices de confusion suivantes illustrent les performances des modèles évalués :**")

        MODELS = ["SVM", "KNN", "RF"]
        MATRICES_FOLDER = chemin_global / Path("matrices_confusion")

        choice1 = st.selectbox("Modèle :", ["all"] + MODELS, index=0, key="MATRICES_1")

        col1, col2 ,_ = st.columns([1,1,1])
        with col1: 
            cont_2 = st.container()
        with col2: 
            cont_1 = st.container()
            
            # Checkbox Affichage des noms de fichiers
            with cont_1:
                show_filenames1 = st.checkbox("Noms fichiers", value=True, key="NAMES1")
            
            # Button Affichage des matrices de confusion
            with cont_2:
                if st.button("Matrices de confusion", key="LOAD1"): 
                    sample_map = {}
                    for model_name in MODELS:
                        model_path = MATRICES_FOLDER / model_name
                        if model_path.exists():
                            img_files = sorted(model_path.glob("*.png")) + sorted(model_path.glob("*.jpg"))
                            if img_files:
                                sample_map[model_name] = [{"image": str(img_files[0])}]
                    st.session_state["sample_map_1"] = sample_map  # Key unique

    # Conteneur d'affichage des matrices de confusion
    with cont_image_display:
        st.markdown("## 📈 Matrices de Confusion")
        sample_map = st.session_state.get("sample_map_1", {})
        if not sample_map: 
            st.info("👆 Cliquez 'Matrices de confusion'")
        else:
            targets = list(sample_map.keys()) if choice1 == "all" else [choice1]
            for model_name, entries in sample_map.items():
                if model_name in targets:
                    st.markdown(f"### {model_name}")
                    if entries:
                        img_path = Path(entries[0]["image"])
                        if img_path.exists():
                            im = Image.open(img_path).convert("RGB")
                            im.thumbnail((500, 500))
                            caption = img_path.name if show_filenames1 else "Matrix"
                            st.image(im, caption=caption, output_format='PNG',width="stretch") 

    


    st.divider()

    # Modèles testés 
    model_test_container = st.container(border=True)
    with model_test_container:
        st.title("**Optimisation des modèles de machine learning**")
        st.success("**Random Forest** : modèle le plus performant.")
        st.error("**SVM** : modèle le moins performant.")

        st.info("""
    **Objectif de notre choix :**
    - Renforcer l'efficacité et améliorer les performances de chaque modèle
    - Obtenir une analyse comparative complète
    """)
    
    st.divider()

    # Grid Search
    grid_container = st.container(border=True)
    with grid_container:
        st.title("Optimisation par Grid Search")
        st.info("""
    ### **Grid Search**
    
    **Fonctionnement** : Grid Search teste toutes les combinaisons possibles d'hyperparamètres sur une "grille" prédéfinie.
                        
    
    **Évaluation** : Validation croisée (k-fold) évalue chaque combinaison de manière robuste en divisant les données en k splits.
    """)
        st.divider()
        hyp_col1, hyp_col2 = st.columns([0.5,0.5],gap="small")
        with hyp_col1:
            hyp_cont1 = st.container(border=True, height="stretch")
            with hyp_cont1:
                st.subheader("**Hyperparamètres testés**")
                st.info("**SVM** : C (régularisation), max itérations")
                st.info("**Random Forest** : nb arbres, profondeur max,.. etc")

        with hyp_col2:
            hyp_cont2 = st.container(border=True, height="stretch")
            with hyp_cont2:
                st.subheader("**Sauvegarde JSON**")
                st.info("**Les différents hyperparamètres ont été sauvegardés dans un fichier.json**")

                chemin_absolu = rf"{chemin_global}parametres.png"
                image_path = Path(chemin_absolu).relative_to(Path.cwd())
                st.image(str(image_path), caption="Les hyperparamètres", width="content")

    st.divider()

    # Résultats Grid Search
    results_container = st.container(border=True)
    with results_container:

        st.title("**Résultats de la matrice de confusion pour les modèles avec Grid Search**")

        # disposition en 2 colonnes : affichage images / sélection
        GS_col1, GS_col2 = st.columns([2,1],gap="small")
        with GS_col1:
            cont_image_display_GS = st.container(border=True, height="stretch")
        with GS_col2: 
            cont_select_GS = st.container(border=True, height="stretch")

        #Conteneur de sélection des matrices de confusion avec Grid Search
        with cont_select_GS:
            choice2 = st.selectbox("Modèle :", ["all"] + MODELS, index=0, key="MATRICES_2")
            n_images2 = 2 #st.number_input("N images / modèle :", 1, 2, 2, key="N2")
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

        # Affichage des matrices de confusion avec Grid Search
        with cont_image_display_GS:
            st.title("Matrices de Confusion")
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
                                THUMBNAIL_MAX = ((500, 500))
                                im.thumbnail(THUMBNAIL_MAX)
                                caption = img_path.name if show_filenames2  else f"Matrix {idx+1}"
                                st.image(im, caption=caption,output_format='PNG', width="stretch")
                            total += 1

        st.success("**Les hyperparamètres optimaux identifiés par Grid Search ont permis d'améliorer significativement les performances de base des modèles.**")


st.divider()



if __name__ == "__main__":
    run()