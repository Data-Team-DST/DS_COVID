# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: honest critique & lessons-learned interactive template —
#   document limits, biases, prioritized fixes, future directions.

import streamlit as st
from streamlit_extras.colored_header import colored_header


def run():
    # Header / hero
    colored_header(
        label="Critical Review & Lessons Learned",
        description=(
            "Analyse critique du projet : limites méthodologiques, biais identifiés, "
            "risques pour l’interprétation et axes d’amélioration priorisés."
        ),
        color_name="blue-70"
    )
    st.divider()

    # 1. Topic overview & context
    st.markdown(
        "## 1. Topic overview & context\n\n"
        "Cette section confronte les objectifs initiaux du projet aux résultats réellement obtenus. "
        "L’ambition était de conduire une analyse exploratoire approfondie de radiographies thoraciques "
        "COVID afin d’identifier des signaux visuels discriminants et d’évaluer la faisabilité d’outils "
        "d’aide à la classification.\n\n"
        "Certaines analyses avancées (embeddings profonds, explicabilité temps réel) ont été limitées "
        "par des contraintes de calcul, d’infrastructure cloud et de temps."
    )
    st.divider()

    # 2. Data critique
    st.markdown(
        "## 2. Data critique\n\n"
        "Analyse critique du dataset utilisé : biais potentiels, limites structurelles "
        "et conséquences sur l’interprétation des résultats."
    )
    st.text_area(
        "Limites et biais des données",
        value=(
            "- Déséquilibre marqué entre les classes (classe \"Normal\" sur-représentée).\n"
            "- Faible volumétrie pour certaines classes rares (ex: Viral Pneumonia).\n"
            "- Absence de métadonnées cliniques (âge, sexe, hôpital, type de machine).\n"
            "- Agrégation de sources hétérogènes pouvant introduire des biais d’acquisition."
        ),
        height=140,
        key="critique_data"
    )
    st.divider()

    # 3. Analyses & visualizations
    st.markdown(
        "## 3. Analyses & visualizations\n\n"
        "Limites liées aux analyses exploratoires et aux visualisations produites."
    )
    st.text_area(
        "Limites analytiques",
        value=(
            "- Analyses majoritairement descriptives, sans tests statistiques formels.\n"
            "- Visualisations utiles pour l’exploration mais non suffisantes pour inférences causales.\n"
            "- Certaines métriques (ex: similarité inter-classes) reposent sur des heuristiques.\n"
            "- Absence de validation croisée inter-datasets ou inter-sites."
        ),
        height=140,
        key="critique_analysis"
    )
    st.divider()

    # 4. Preprocessing & rationale
    st.markdown(
        "## 4. Preprocessing & rationale\n\n"
        "Discussion critique des choix de preprocessing et des risques associés."
    )
    st.text_area(
        "Risques liés au preprocessing",
        value=(
            "- Redimensionnement des images pouvant entraîner une perte de détails fins.\n"
            "- Normalisation globale susceptible de lisser certains signaux pathologiques.\n"
            "- Choix guidés par contraintes de calcul plutôt que par validation clinique.\n"
            "- Documentation initiale des étapes perfectible."
        ),
        height=120,
        key="critique_preproc"
    )
    st.divider()

    # 5. Model critique
    st.markdown(
        "## 5. Model critique\n\n"
        "Limites observées ou anticipées sur les modèles de classification."
    )
    st.text_area(
        "Limites modèles",
        value=(
            "- Sensibilité élevée aux classes rares.\n"
            "- Risque d’overfitting sur un dataset déséquilibré.\n"
            "- Absence de validation externe sur des données indépendantes.\n"
            "- Robustesse non évaluée face à des variations d’acquisition."
        ),
        height=130,
        key="critique_model_perf"
    )
    st.divider()

    # 6. Best model analysis
    st.markdown(
        "## 6. Best model analysis\n\n"
        "Vulnérabilités potentielles du meilleur modèle identifié dans le cadre exploratoire."
    )
    st.text_area(
        "Vulnérabilités & tests manquants",
        value=(
            "- Risque de drift visuel lié à de nouveaux appareils ou protocoles d’imagerie.\n"
            "- Absence d’explicabilité robuste exploitable en production.\n"
            "- Fairness non évaluée faute de métadonnées patient.\n"
            "- Tests de robustesse et scénarios adverses non réalisés."
        ),
        height=140,
        key="critique_model"
    )
    st.divider()

    # 7. Conclusions & business relevance
    st.markdown(
        "## 7. Conclusions & business relevance\n\n"
        "Conséquences des limites identifiées pour un usage métier ou clinique."
    )
    st.text_area(
        "Risques pour l’usage réel",
        value=(
            "- Usage limité à des fins exploratoires ou pédagogiques.\n"
            "- Non adapté à une aide au diagnostic sans supervision médicale.\n"
            "- Nécessité de garde-fous forts avant toute mise en production.\n"
            "- Interprétation des résultats à contextualiser systématiquement."
        ),
        height=130,
        key="critique_conclusion"
    )
    st.divider()

    # 8. Critique & future perspectives
    st.markdown(
        "## 8. Critique & future perspectives\n\n"
        "Axes d’amélioration identifiés, priorisés selon effort et impact."
    )
    st.text_area(
        "Plan d’action priorisé",
        value=(
            "- Quick wins : rééquilibrage du dataset, ajout de tests statistiques simples.\n"
            "- Moyen terme : calcul d’embeddings profonds hors-ligne, analyses de similarité robustes.\n"
            "- Long terme : pipeline MLOps, validation multi-centres, intégration d’explicabilité."
        ),
        height=150,
        key="critique_plan"
    )
    st.divider()

    # 9. CI/CD & production controls
    st.markdown(
        "## 9. CI/CD & production controls\n\n"
        "Recommandations pour un éventuel passage en production."
    )
    st.markdown(
        "- **Monitoring** : performance, dérive des données, dérive des prédictions.\n"
        "- **Qualité** : tests automatiques, versioning des datasets et modèles.\n"
        "- **Sécurité** : plan de rollback, audits réguliers, traçabilité des décisions."
    )


# STATUS: page/08_critique.py — intégrale, revue critique honnête et professionnelle,
# alignée avec les analyses exploratoires (pages 1–3),
# prête à évoluer avec les sections modèles et MLOps (pages 4–6).
