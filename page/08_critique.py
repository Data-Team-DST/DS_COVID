# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: honest critique & lessons-learned interactive template — document limits, biases, prioritized fixes, future directions.

import streamlit as st
from streamlit_extras.colored_header import colored_header

def run():
    # Header / hero
    colored_header(
        label="Critical Review & Lessons Learned",
        description="Documenter limites, biais, risques, backlog priorisé et recommandations futures.",
        color_name="blue-70"
    )
    st.divider()

    # 1. Topic overview & context
    st.markdown(
        "## 1. Topic overview & context\n\n"
        "Cadre critique : rappeler les ambitions initiales vs réalisations, ce qui a fonctionné et ce qui a été limité par contraintes techniques, temps ou data."
    )
    st.divider()

    # 2. Data critique
    st.markdown(
        "## 2. Data critique\n\n"
        "Documenter biais détectés, lacunes, limitations d'accès ou qualité, et impact sur les analyses."
    )
    st.text_area(
        "Problèmes data identifiés",
        value="- Biais sur sous-populations\n- Manque de données pour certaines pathologies\n- Échantillon non équilibré",
        height=100,
        key="critique_data"
    )
    st.divider()

    # 3. Analyses & visualizations
    st.markdown(
        "## 3. Analyses & visualizations\n\n"
        "Limites dans l’analyse : granularité, échantillonnage, tests statistiques non effectués, visualisations manquantes ou non reproductibles."
    )
    st.text_area(
        "Limites analyses",
        value="- Manque tests de robustesse cross-site\n- Graphiques simplifiés pour rapidité\n- Analyse de séries temporelles limitée",
        height=100,
        key="critique_analysis"
    )
    st.divider()

    # 4. Preprocessing & rationale
    st.markdown(
        "## 4. Preprocessing & rationale\n\n"
        "Risques induits par le preprocessing (leakage, perte d’information). Actions prises pour limiter ces risques et leçons apprises."
    )
    st.text_area(
        "Risques preprocessing",
        value="- Possibilité de leakage sur certaines features\n- Imputation median vs mean discutée\n- Documentation des étapes manquante",
        height=80,
        key="critique_preproc"
    )
    st.divider()

    # 5. Model critique
    st.markdown(
        "## 5. Model critique\n\n"
        "Sensibilité et robustesse : variance entre folds, performance par segment, hyperparamètres sensibles."
    )
    st.text_area(
        "Limites modèles",
        value="- Variance élevée sur petits sous-groupes\n- Certaines classes rares mal prédites\n- Besoin de validation externe",
        height=100,
        key="critique_model_perf"
    )
    st.divider()

    # 6. Best model analysis
    st.markdown(
        "## 6. Best model analysis\n\n"
        "Vulnérabilités identifiées : drift potentiel, fairness issues, tests complémentaires nécessaires avant production."
    )
    st.text_area(
        "Vulnérabilités & tests requis",
        value="- Drift sur nouvelles cliniques\n- Fairness check manquant pour certaines classes\n- Tests adverses à planifier",
        height=100,
        key="critique_model"
    )
    st.divider()

    # 7. Conclusions & business relevance
    st.markdown(
        "## 7. Conclusions & business relevance\n\n"
        "Risques résiduels pour le déploiement et recommandations de mitigation immédiate."
    )
    st.text_area(
        "Risques résiduels & mitigations",
        value="- Monitorer métriques critiques post-déploiement\n- Plan de rollback prêt\n- Alertes sur performances critiques",
        height=100,
        key="critique_conclusion"
    )
    st.divider()

    # 8. Critique & future perspectives
    st.markdown(
        "## 8. Critique & future perspectives\n\n"
        "Plan d’action pour corriger les points faibles, prioriser quick wins vs R&D, estimer effort/impact, et identifier opportunités pour améliorations futures."
    )
    st.text_area(
        "Plan d'action (priorisé)",
        value="- Quick wins: nettoyer features manquantes, enrichir dataset\n- Moyen terme: tests fairness + augmentation données\n- Long terme: pipeline MLOps complet, modèles hybrides",
        height=140,
        key="critique_plan"
    )
    st.divider()

    # 9. CI/CD & production controls
    st.markdown(
        "## 9. CI/CD & production controls\n\n"
        "Ajouter contrôles pour détecter régressions, drift, dérive, garantir rollbacks sûrs. Inclure tests adverses et fairness dans le pipeline de gating."
    )
    st.markdown(
        "- **Recommandations** : intégrer tests éthiques, fairness, robustness et alerting automatisé dans pipeline de CI/CD.\n"
        "- **Artifacts** : logs QA, snapshots métriques, reports d’audit."
    )

# STATUS: page/08_critique.py — intégrale, Streamlit Extras obligatoire, interactive critique template ready for committee review; links directly limitations → improvements → future perspectives.
