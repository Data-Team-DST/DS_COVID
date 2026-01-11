# page/08_cicd_pipeline.py
# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: CI pipeline presentation —
#   quality-focused, reproducible, deliberately non-industrial.

import streamlit as st
from streamlit_extras.colored_header import colored_header


def run():
    # Header / hero
    colored_header(
        label="CI/CD & qualité logicielle",
        description=(
            "Présentation du pipeline CI réellement implémenté : "
            "objectifs, outils, limites assumées et positionnement pédagogique."
        ),
        color_name="blue-70"
    )
    st.divider()

    # 1. Why CI/CD here?
    st.markdown(
        "## 1. Pourquoi un pipeline CI/CD dans ce projet ?\n\n"
        "Même en l’absence de mise en production, un effort volontaire a été réalisé "
        "sur la **qualité du code**, la **reproductibilité** et la **maintenabilité**.\n\n"
        "L’objectif n’est pas l’industrialisation, mais d’éviter : "
        "code fragile, régressions silencieuses et dette technique."
    )
    st.divider()

    # 2. Implemented CI pipeline
    st.markdown(
        "## 2. Pipeline CI implémenté\n\n"
        "Le pipeline est exécuté automatiquement via GitHub Actions à chaque push et pull request."
    )
    st.text_area(
        "CI en place",
        value=(
            "- Linting Python avec **pylint** (seuil bloquant : score ≥ 8)\n"
            "- Tests unitaires avec **pytest**\n"
            "- Génération de rapports de couverture\n"
            "- Analyse statique centralisée via **SonarCloud**"
        ),
        height=130,
        key="cicd_core"
    )
    st.divider()

    # 3. Code quality & testing philosophy
    st.markdown(
        "## 3. Philosophie qualité et tests\n\n"
        "Les choix CI/CD reflètent une approche pragmatique et réaliste."
    )
    st.text_area(
        "Approche qualité",
        value=(
            "- Priorité à la lisibilité et à la robustesse du code\n"
            "- Tests unitaires ciblés sur les composants critiques\n"
            "- Couverture volontairement modeste mais contrôlée\n"
            "- Détection précoce des régressions"
        ),
        height=130,
        key="cicd_quality"
    )
    st.divider()

    # 4. Artefacts & traçabilité
    st.markdown(
        "## 4. Artefacts et traçabilité\n\n"
        "Le pipeline produit des éléments exploitables pour audit et suivi."
    )
    st.text_area(
        "Artefacts générés",
        value=(
            "- Rapports de couverture pytest (coverage.xml)\n"
            "- Analyses SonarCloud\n"
            "- Logs GitHub Actions\n"
            "- Historique des exécutions CI"
        ),
        height=120,
        key="cicd_artifacts"
    )
    st.divider()

    # 5. What is intentionally missing
    st.markdown(
        "## 5. Ce qui est volontairement absent\n\n"
        "Certains éléments classiques de CI/CD ne sont pas implémentés, par choix."
    )
    st.text_area(
        "Limites assumées",
        value=(
            "- Pas de build Docker\n"
            "- Pas de déploiement continu (CD)\n"
            "- Pas d’orchestration Kubernetes\n"
            "- Pas de monitoring temps réel\n\n"
            "Ces choix sont cohérents avec le périmètre académique, "
            "les contraintes de ressources et les objectifs pédagogiques."
        ),
        height=160,
        key="cicd_limits"
    )
    st.divider()

    # 6. Perspective & alignment
    st.markdown(
        "## 6. Perspectives et alignement formation\n\n"
        "Le pipeline constitue une base saine pour des évolutions futures."
    )
    st.text_area(
        "Évolutions possibles",
        value=(
            "- Introduction progressive de Docker\n"
            "- Séparation CI / CD\n"
            "- Déploiement contrôlé en environnement de test\n"
            "- Monitoring basique des performances et dérives"
        ),
        height=140,
        key="cicd_future"
    )
    st.divider()

    # 7. Final positioning
    st.markdown(
        "## 7. Positionnement final\n\n"
        "- Pipeline **CI réel et fonctionnel**, orienté qualité\n"
        "- Implémenté au-delà des exigences minimales\n"
        "- Adapté à un projet académique avancé\n"
        "- Base crédible pour une industrialisation future, sans sur-promesse"
    )
