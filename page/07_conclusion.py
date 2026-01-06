# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: executive summary interactive template — synthèse, recommandations et plan d'action clair, orienté métiers.

import streamlit as st
from streamlit_extras.colored_header import colored_header

def run():
    # Header / hero
    colored_header(
        label="7. Conclusions & Perspectives",
        color_name="blue-70"
    )
    st.divider()


    conclusion_text = """
    Cette étude s'inscrit dans une démarche visant à comparer la pertinence des approches d'apprentissage profond et des méthodes d'apprentissage automatique classiques pour la classification d'images thoraciques.

    Le prétraitement (augmentation, rééchantillonnage, redimensionnement, niveaux de gris, normalisation) a été essentiel pour corriger le déséquilibre des classes et stabiliser l'entraînement.

    Random Forest surpasse SVM et KNN par ses performances et son équilibre pour cette tâche de classification. Il démontre la meilleure capacité de généralisation, particulièrement pour les classes "normal" et "viral", malgré une confusion résiduelle entre COVID et "lung".

    L'application d'une optimisation par Grid Search a permis d'améliorer significativement les performances de base grâce à un ajustement précis des hyperparamètres et à une meilleure capacité de généralisation du modèle.

    L'intégration de la méthode LIME a permis d'identifier explicitement les zones discriminantes activées par le réseau de neurones, bien que cette approche puisse encore être améliorée pour obtenir de meilleurs résultats.

    Les résultats mettent en avant les performances remarquables du modèle InceptionV3, qui se distingue par une grande capacité à reconnaître et différencier les différents types de cas. Ils confirment ainsi l'efficacité et la fiabilité du transfert d'apprentissage, particulièrement bien adapté aux contraintes et exigences du domaine médical.
    """
    
    st.markdown(conclusion_text)
    
    st.divider()

    st.markdown("### **Perspectives**")
    
    perspectives = """
    Plusieurs axes d'amélioration peuvent être envisagés pour la suite de ce travail :
    
    L'exploration de modèles plus récents, comme EfficientNet ou Vision Transformer (ViT), pourrait encore renforcer la capacité de généralisation du système.
    
    Le recours à des approches hybrides, combinant réseaux de neurones convolutifs et méthodes d'ensemble telles que XGBoost, constitue également une piste intéressante.
    
    L'adoption d'outils d'explicabilité plus avancés, comme SHAP ou Grad-CAM++, permettrait d'affiner la compréhension des décisions du modèle et de renforcer la confiance dans son déploiement en contexte clinique.
    """
   
    st.markdown(perspectives)

    st.divider()

# STATUS: page/07_conclusion.py — intégrale, Streamlit Extras obligatoire, executive-ready interactive conclusion template, liaison directe résultats → problématique métier, priorisation actions et plan de production.
# Lancement
if __name__ == "__main__":
    run()