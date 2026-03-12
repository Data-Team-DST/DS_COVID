# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: executive summary — synthèse fondée sur analyses 1–3 (EDA & visualisations).

import streamlit as st
from streamlit_extras.colored_header import colored_header


def run():
    # Header / hero
    colored_header(
        label="Executive Summary & Recommendations",
        description="Synthèse décisionnelle fondée sur l'analyse des données et des visualisations (sections 1–3).",
        color_name="blue-70"
    )
    st.divider()

    # 1. Topic overview & context
    st.markdown(
        "## 1. Topic overview & context\n\n"
        "Ce projet vise à évaluer la faisabilité d’un système d’analyse automatique de radiographies thoraciques "
        "à partir d’un dataset public (COVID-19 Radiography Database).\n\n"
        "L’objectif n’est pas de produire un outil médical final, mais de déterminer si les données disponibles "
        "présentent des signaux visuels suffisamment exploitables pour un **POC de classification**, "
        "et d’identifier les limites méthodologiques dès les premières étapes."
    )
    st.divider()

    # 2. Data adequacy & limitations
    st.markdown(
        "## 2. Data adequacy & limitations\n\n"
        "Cette section synthétise l’adéquation des données au regard des analyses exploratoires menées."
    )
    st.text_area(
        "Synthèse données",
        value=(
            "- Volume conséquent (>21 000 images), suffisant pour une analyse exploratoire robuste\n"
            "- Déséquilibre marqué entre classes (classe *Normal* fortement majoritaire)\n"
            "- Images majoritairement en faux-RGB (grayscale dupliqué sur 3 canaux)\n"
            "- Qualité visuelle globalement homogène (exposition, contraste, entropie)\n"
            "- Dataset adapté à un POC exploratoire, mais insuffisant pour un usage clinique sans enrichissement"
        ),
        height=130,
        key="conclusion_data"
    )
    st.divider()

    # 3. Key insights (visual summary)
    st.markdown(
        "## 3. Key insights (from visual analysis)\n\n"
        "Les analyses visuelles et statistiques mettent en évidence plusieurs enseignements clés :"
    )
    st.markdown(
        "- **Déséquilibre structurel des classes**, susceptible d’introduire un biais modèle sans stratégie corrective.\n"
        "- **Similarités visuelles inter-classes**, confirmant que la classification n’est pas triviale.\n"
        "- **Homogénéité globale des niveaux de luminosité et de contraste**, limitant les risques de fuite triviale.\n"
        "- **Présence généralisée de faux-RGB**, conforme aux pratiques en imagerie médicale mais à prendre en compte."
    )
    st.info(
        "Ces constats justifient une approche prudente : modèles robustes, métriques adaptées et interprétation encadrée."
    )
    st.divider()

    # 4. Preprocessing impact (fondé sur ce qui est déjà observé)
    st.markdown(
        "## 4. Preprocessing impact (observed so far)\n\n"
        "Les analyses exploratoires permettent déjà d’anticiper l’impact de certains prétraitements."
    )
    st.text_area(
        "Prétraitements clés identifiés",
        value=(
            "- Normalisation des intensités nécessaire pour stabiliser l’apprentissage\n"
            "- Vérification fake-RGB indispensable mais non bloquante\n"
            "- Aucune transformation lourde justifiée à ce stade sans modèle entraîné\n"
            "- Prétraitements doivent rester simples pour préserver l’interprétabilité"
        ),
        height=110,
        key="conclusion_preproc"
    )
    st.divider()

    # 5. Model summary & results (positionnement réaliste)
    st.markdown(
        "## 5. Model perspective & expected outcomes\n\n"
        "À ce stade du projet, aucun modèle final n’est présenté volontairement."
    )
    st.text_area(
        "Positionnement modèle",
        value=(
            "- Modèles envisagés : CNN standards pour classification multi-classes\n"
            "- Objectif : évaluer la séparabilité globale, pas un diagnostic automatisé\n"
            "- Valeur principale : aide à l’exploration et à la priorisation des cas\n"
            "- Toute performance devra être interprétée à la lumière des biais identifiés"
        ),
        height=120,
        key="conclusion_model"
    )
    st.divider()

    # 6. Key considerations & risks
    st.markdown(
        "## 6. Key considerations & risks\n\n"
        "Plusieurs points de vigilance doivent être explicitement pris en compte."
    )
    st.text_area(
        "Risques et limites",
        value=(
            "- Biais liés au déséquilibre des classes\n"
            "- Généralisation limitée à d’autres contextes cliniques\n"
            "- Absence de GPU en environnement Streamlit Cloud (calculs lourds à externaliser)\n"
            "- Usage strictement exploratoire hors cadre réglementaire médical"
        ),
        height=120,
        key="conclusion_risks"
    )
    st.divider()

    # 7. Prioritized recommendations (Top 3)
    st.markdown(
        "## 7. Prioritized recommendations (Top 3)\n\n"
        "Actions recommandées à l’issue des analyses 1–3 :"
    )
    st.markdown(
        "1. **Court terme** — Consolider un POC reproductible basé sur les analyses actuelles.\n"
        "2. **Moyen terme** — Rééquilibrer et enrichir le dataset avant tout entraînement sérieux.\n"
        "3. **Long terme** — Externaliser les calculs lourds (embeddings, modèles) hors Streamlit."
    )
    st.divider()

    # 8. Future perspectives
    st.markdown(
        "## 8. Future perspectives\n\n"
        "Pistes d’évolution identifiées pour la suite du projet :"
    )
    st.text_area(
        "Backlog priorisé",
        value=(
            "- Analyse des erreurs critiques (faux négatifs)\n"
            "- Études de robustesse inter-sources\n"
            "- Intégration de signaux non visuels (métadonnées cliniques)\n"
            "- Comparaison CNN classiques vs approches plus légères"
        ),
        height=140,
        key="conclusion_backlog"
    )
    st.divider()

    # 9. CI/CD & production plan (réaliste)
    st.markdown(
        "## 9. CI/CD & production considerations\n\n"
        "Toute mise en production devra respecter une séparation claire entre calcul et visualisation."
    )
    st.markdown(
        "- **Offline** : calcul des features, embeddings, modèles (GPU/local)\n"
        "- **Online (Streamlit)** : visualisation, audit, exploration\n"
        "- **Artefacts** : métriques versionnées, rapports, jeux d’échantillons figés\n"
        "- **Monitoring** : dérive des données, cohérence des distributions"
    )
