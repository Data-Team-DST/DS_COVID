# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: executive summary interactive template — synthèse, recommandations et plan d'action clair, orienté métiers.

import streamlit as st
from streamlit_extras.colored_header import colored_header

def run():
    # Header / hero
    colored_header(
        label="Executive Summary & Recommendations",
        description="Synthèse projet, KPIs clés, priorisation des actions et plan de production.",
        color_name="blue-70"
    )
    st.divider()

    # 1. Topic overview & context
    st.markdown(
        "## 1. Topic overview & context\n\n"
        "Bref rappel du périmètre du projet et des objectifs métiers : pourquoi ce projet a été lancé et quelles questions il visait à résoudre."
    )
    st.divider()

    # 2. Data adequacy & limitations
    st.markdown(
        "## 2. Data adequacy & limitations\n\n"
        "Synthèse sur l'adéquation des données : volumes, qualité, granularité. Mentionner les limitations critiques qui ont été résolues ou restant à surveiller."
    )
    st.text_area(
        "Résumé data",
        value="- Data quality check completed\n- Gaps identified for feature X\n- Hold-out set validated",
        height=80,
        key="conclusion_data"
    )
    st.divider()

    # 3. Key insights (visual summary)
    st.markdown(
        "## 3. Key insights (visual summary)\n\n"
        "Mettre en avant 2–4 figures ou tableaux clés qui supportent les conclusions métier. Ajouter placeholders pour figures ou liens vers les sections détaillées."
    )
    st.columns([1,1])
    st.info("Placeholder figures : KPI trends, feature impact, error distributions, segmentation results.")
    st.divider()

    # 4. Preprocessing impact
    st.markdown(
        "## 4. Preprocessing impact\n\n"
        "Rappel succinct des transformations majeures qui ont eu un impact sur les résultats métier (ex: imputations, filtrages spécifiques)."
    )
    st.text_area(
        "Transformations clés",
        value="- Filtrage données post-2019\n- Normalisation images\n- Feature selection",
        height=80,
        key="conclusion_preproc"
    )
    st.divider()

    # 5. Model summary & results
    st.markdown(
        "## 5. Model summary & results\n\n"
        "Annonce du modèle retenu avec métriques clefs de manière concise (ex: AUC, RMSE, accuracy). Lier directement aux KPIs métiers."
    )
    st.text_area(
        "Modèle retenu et KPIs",
        value="- Best model: CNN lung X-ray classifier\n- Accuracy: 93%, Sensitivity: 91%\n- KPI impact: détection précoce de pneumonie",
        height=80,
        key="conclusion_model"
    )
    st.divider()

    # 6. Key considerations & risks
    st.markdown(
        "## 6. Key considerations & risks\n\n"
        "Points d'attention majeurs : conditions d'utilisation, limites pratiques, risques identifiés (drift, biais, sous-populations)."
    )
    st.text_area(
        "Risques / limites",
        value="- Drift possible sur nouvelles cliniques\n- Limité aux images frontal-view\n- Besoin d'anonymisation stricte",
        height=80,
        key="conclusion_risks"
    )
    st.divider()

    # 7. Prioritized recommendations (Top 3)
    st.markdown(
        "## 7. Prioritized recommendations (Top 3)\n\n"
        "Formuler actions concrètes avec responsables et KPI impacté."
    )
    st.markdown(
        "1. **Action immédiate** : Déploiement pilote sur hôpital X – Responsable: PO – KPI: réduction délai diagnostic\n"
        "2. **Action moyen terme** : Collecte images supplémentaires et amélioration du dataset – Plan: Data Owner & DS team\n"
        "3. **Action long terme** : Mise en place pipeline MLOps / Feature store – Plan: Infra & DS Ops"
    )
    st.divider()

    # 8. Future perspectives
    st.markdown(
        "## 8. Future perspectives\n\n"
        "Liste priorisée des actions ou études à mener avec plus de temps ou de données."
    )
    st.text_area(
        "Backlog priorisé (3–6 items)",
        value="- Test cross-hospital generalization\n- Augmentation data pour pathologies rares\n- Exploration modèles hybrides CNN+Transformer",
        height=120,
        key="conclusion_backlog"
    )
    st.divider()

    # 9. CI/CD & production plan
    st.markdown(
        "## 9. CI/CD & production plan\n\n"
        "Plan de mise en production avec étapes clés, rôles, SLA, et monitoring. Relier directement aux artefacts produits et tests automatisés."
    )
    st.markdown(
        "- **Deliverables** : modèles pickled/h5, métriques CSV/JSON, rapport d'évaluation\n"
        "- **Monitoring** : performance, dérive, latence\n"
        "- **Runbook déploiement** : checklist pré-déploiement, tests end-to-end, plan rollback"
    )

# STATUS: page/07_conclusion.py — intégrale, Streamlit Extras obligatoire, executive-ready interactive conclusion template, liaison directe résultats → problématique métier, priorisation actions et plan de production.
