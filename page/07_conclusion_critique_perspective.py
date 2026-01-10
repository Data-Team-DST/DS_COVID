# page/07_conclusion_critique_perspective.py
# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: unified conclusion —
#   synthèse décisionnelle + critique honnête + perspectives réalistes.

import streamlit as st
from streamlit_extras.colored_header import colored_header


def run():
    # Header / hero
    colored_header(
        label="Conclusion critique & perspectives",
        description=(
            "Synthèse finale du projet : ce que les données permettent de conclure, "
            "leurs limites structurelles et les perspectives réalistes d’évolution."
        ),
        color_name="blue-70"
    )
    st.divider()

    # 1. Project positioning
    st.markdown(
        "## 1. Positionnement du projet\n\n"
        "Ce projet avait pour objectif principal d’évaluer la **faisabilité exploratoire** "
        "d’une classification automatique de radiographies thoraciques à partir d’un dataset public "
        "(COVID-19 Radiography Database).\n\n"
        "Il ne s’agit **ni d’un outil médical**, ni d’un système de diagnostic, mais d’un **POC analytique** "
        "visant à identifier des signaux visuels exploitables et à comprendre, dès les premières étapes, "
        "les limites méthodologiques associées."
    )
    st.divider()

    # 2. What the data actually allows
    st.markdown(
        "## 2. Ce que les données permettent réellement de dire\n\n"
        "Les analyses exploratoires (EDA, visualisations, statistiques descriptives) conduisent "
        "aux constats suivants :"
    )
    st.text_area(
        "Constats clés",
        value=(
            "- Volume de données conséquent (>21 000 images), suffisant pour une analyse exploratoire robuste\n"
            "- Déséquilibre structurel marqué entre les classes (classe *Normal* majoritaire)\n"
            "- Images majoritairement en faux-RGB (grayscale dupliqué sur 3 canaux)\n"
            "- Qualité visuelle globalement homogène (luminosité, contraste, entropie)\n"
            "- Absence de signaux triviaux exploitables directement pour une classification naïve"
        ),
        height=140,
        key="conclusion_findings"
    )
    st.info(
        "Ces éléments indiquent que la classification n’est pas triviale, "
        "ce qui constitue un point positif pour un POC sérieux."
    )
    st.divider()

    # 3. Data & methodological limitations
    st.markdown(
        "## 3. Limites des données et biais méthodologiques\n\n"
        "Une lecture critique du dataset et des analyses est indispensable pour éviter toute sur-interprétation."
    )
    st.text_area(
        "Limites identifiées",
        value=(
            "- Déséquilibre des classes susceptible d’introduire un biais modèle important\n"
            "- Faible volumétrie pour certaines classes rares\n"
            "- Absence de métadonnées cliniques (âge, sexe, contexte d’acquisition)\n"
            "- Agrégation de sources hétérogènes pouvant induire des biais d’acquisition\n"
            "- Généralisation très limitée hors du périmètre du dataset étudié"
        ),
        height=150,
        key="conclusion_limits"
    )
    st.divider()

    # 4. Preprocessing & analytical choices
    st.markdown(
        "## 4. Choix analytiques & preprocessing\n\n"
        "Les décisions de preprocessing ont été guidées par un compromis entre "
        "rigueur méthodologique, contraintes de calcul et interprétabilité."
    )
    st.text_area(
        "Analyse critique des choix",
        value=(
            "- Normalisation des intensités nécessaire pour stabiliser l’apprentissage\n"
            "- Redimensionnement des images pouvant entraîner une perte de détails fins\n"
            "- Les transformations ont volontairement été limitées tant que les modèles n’avaient pas démontré de gain clair en performance.\n"
            "- Choix dictés par des contraintes exploratoires, non par une validation clinique"
        ),
        height=130,
        key="conclusion_preproc"
    )
    st.divider()

    # 5. Risks & usage boundaries
    st.markdown(
        "## 5. Risques et périmètre d’usage\n\n"
        "Les risques suivants doivent être explicitement reconnus."
    )
    st.text_area(
        "Risques majeurs",
        value=(
            "- Sur-interprétation de résultats exploratoires\n"
            "- Généralisation abusive à d’autres contextes cliniques\n"
            "- Sensibilité élevée aux variations d’acquisition\n"
            "- Absence de cadre réglementaire et clinique"
        ),
        height=120,
        key="conclusion_risks"
    )
    st.divider()

    # 6. Prioritized perspectives
    st.markdown(
        "## 6. Perspectives et axes d’amélioration\n\n"
        "Pistes d’évolution identifiées, priorisées selon effort et impact."
    )

    st.text_area(
        "Perspectives",
        value=(
            "- Court terme : rééquilibrage du dataset et mise en place de tests statistiques simples\n"
            "- Moyen terme : calcul d’embeddings profonds hors-ligne, analyses de similarité robustes "
            "et structuration du versionnage des données pour garantir la reproductibilité\n"
            "- Long terme : validation multi-sources, intégration de méthodes d’explicabilité "
            "et mise en place d’un pipeline MLOps incluant gestion des versions et automatisation\n"
            "- Enrichissement futur par métadonnées cliniques si disponibles"
        ),
        height=160,
        key="conclusion_future"
    )

    st.divider()


    # 7. Final takeaway
    st.markdown(
        "## 7. Conclusion finale\n\n"
        "**La valeur principale de ce projet réside moins dans un modèle que dans le raisonnement appliqué aux données.**\n\n"
        "Il démontre l’importance d’une approche prudente, critique et méthodologiquement rigoureuse, "
        "en particulier dans des contextes sensibles comme l’imagerie médicale."
    )
