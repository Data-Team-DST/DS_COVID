# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: honest critique & lessons-learned interactive template — document limits, biases, prioritized fixes, future directions.

import streamlit as st
from streamlit_extras.colored_header import colored_header
import pandas as pd 

def run():
    
    colored_header(
        label="8. Les défis rencontrés et les solutions",
        color_name="blue-70"
    )
    st.divider()

    # Fonction tableau déplacée au bon endroit
    def afficher_tableau_difficultes():
        
        # Données du tableau
        data = {
            "Les défis": [
                "Organisation du pipeline",
                "Synchronisation Git",
                "Les ressources Matérielles",
                "Interprétabilité des modèles"
            ],
            "Description": [
                "Gestion complexe des étapes du projet",
                "Conflits de versions et états",
                "Ressources limitées restreignant la taille des batches et la complexité du modèle",
                "Modèles complexes sur classes déséquilibrées"
            ],
            "Solution mise en œuvre": [
                "Utilisation de scripts d'orchestration et ajustement de l'architecture",
                "Mise en place de branches dédiées et expérimentations itératives",
                "Utilisation du transfert d apprentissage, réduction de la taille des batches et ajustement de l architecture",
                "**SHAP + Grad-CAM**, nuançage des résultats et transparence des décisions"
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Affichage stylé
        st.table(df)
        st.divider()

    # Appel de la fonction
    afficher_tableau_difficultes()

# Lancement
if __name__ == "__main__":
    run()
