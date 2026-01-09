# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits app-wide dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: deep-dive template for best model — interpretability, error analysis, subgroup performance.

import streamlit as st
from streamlit_extras.colored_header import colored_header
from pathlib import Path
import os

# Optional imports placeholders
# import torch
# from tensorflow import keras
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, roc_curve, auc

def run():
    # Header / hero
    colored_header(
        label="Optimisation des modèles de machine learning et Interprétabilité du Inception V3",
        color_name="blue-70"
    )
    st.divider()
    st.markdown("""
    **Random Forest** : modèle le plus performant → **optimisation approfondie**
    
    **SVM** : modèle le moins performant → **amélioration ciblée**
    
    **Objectif :**
    - Renforcer l'efficacité du meilleur modèle
    - Améliorer les performances du modèle faible  
    - Obtenir une analyse comparative complète
    """)
    
    st.divider()

    st.markdown("""
    **Principe** : Grid Search + validation croisée
    
    **Fonctionnement** : Grid Search teste toutes les combinaisons possibles d'hyperparamètres sur une "grille" prédéfinie.
                        
    
    **Évaluation** : Validation croisée (k-fold) évalue chaque combinaison de manière robuste en divisant les données en k plis.
    """)
    st.divider()

    st.markdown("### **Hyperparamètres testés**")
    st.markdown("""
    **SVM** : C (régularisation), max itérations
    
    **Random Forest** : nb arbres, profondeur max, critères division
    """)
    chemin_global = Path(__file__).parent.parent
    #rajouter /page/images/ au chemin global
    chemin_global = os.path.join(chemin_global, "page", "images/")
    st.write(f"Chemin global du projet : {chemin_global}")
    st.info("**Les différents hyperparamètres ont été sauvegardés dans un fichier.json**")
    chemin_absolu = rf"{chemin_global}parametres.png"
    image_path = Path(chemin_absolu).relative_to(Path.cwd())
    st.image(str(image_path), caption="Les hyperparamètres", use_column_width=False)
    st.markdown("### **Résultats de la matrice de confusion pour les modèles SVM et Random Forest avec Grid Search**")
    chemin_absolu_1 = rf"{chemin_global}grid_search_randomforest.png"
    image_path_1 = Path(chemin_absolu_1).relative_to(Path.cwd())
    st.image(str(image_path_1), caption="Matrice de confusion random forest", use_column_width=False)
    chemin_absolu_2 = rf"{chemin_global}grid_search_svm.png"
    image_path_2 = Path(chemin_absolu_2).relative_to(Path.cwd())
    st.image(str(image_path_2), caption="Matrice de confusion SVM", use_column_width=False)
    st.info("**Les hyperparamètres optimaux identifiés par Grid Search ont permis daméliorer significativement les performances de base des modèles.**")

    st.markdown("### **Interprétabilité**")
    st.markdown("**LIME (Local Interpretable Model-agnostic Explanations) :**")
    st.markdown("""
                **Principe**: 

                1. IMAGE originale + prédiction du modèle complexe
                
                2. PERTURBATIONS : masque pixels → crée N versions modifiées
                
                3. PRÉDICTIONS : modèle complexe sur chaque version perturbée
                
                4. RÉGRESSION LINÉAIRE : trouve coefficients expliquant les prédictions
                
                5. CARTE DE CHALEUR : pixels importants = coefficients élevés (Jaune)


                **Avantages clés** : compréhensible, fiable, universel, généralisable (SP-LIME).

                """)
    st.info("**Notre modèle a été entraîné sur 2000 images 20 epochs de feature extraction + 30 epochs de fine-tuning (20 dernières couches dégelées)**")
    chemin_absolu_3 = rf"{chemin_global}lime.png"
    image_path_3 = Path(chemin_absolu_3).relative_to(Path.cwd())
    st.image(str(image_path_3), caption="Résultats obtenus avec la méthode LIME", use_column_width=True)
    chemin_absolu_4 = rf"{chemin_global}lime2.png"
    image_path_4 = Path(chemin_absolu_4).relative_to(Path.cwd())
    st.image(str(image_path_4), caption="Résultats obtenus avec la méthode LIME", use_column_width=True)
    st.info("**Faux positifs** : risque élevé classe Normal (malades non détectés)")
    st.info("**Faux négatifs** : risque critique COVID-19 (cas passés inaperçus)")
    st.info("**Robustesse** : bonne Lung-Opacity, faible Normal/COVID-19")
    st.info("**InceptionV3** : performances encourageantes, bonne localisation anomalies")
    st.info("**Limites** : classe Normal + déséquilibre features COVID à améliorer")

if __name__ == "__main__":
    run()





   
# STATUS: page/06_analyse_du_meilleur_modele.py — intégrale, Streamlit Extras obligatoire, sections 1–10 complètes avec placeholders interactifs, pipelines et checklists.
