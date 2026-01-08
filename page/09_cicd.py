# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: interactive CI/CD documentation —
#   décrit pipeline réel implémenté + limites assumées + perspectives réalistes.

import streamlit as st
from streamlit_extras.colored_header import colored_header


def run():
    # Header / hero
    colored_header(
        label="CI/CD Pipeline & Code Quality",
        description=(
            "Documentation du pipeline CI/CD réellement implémenté : "
            "qualité de code, tests automatisés, analyse statique et limites actuelles."
        ),
        color_name="blue-70"
    )
    st.divider()

    # 1. Topic overview & context
    st.markdown(
        "## 1. Topic overview & context\n\n"
        "Bien que la mise en place d’un pipeline CI/CD complet ne soit pas explicitement requise "
        "dans le cadre du projet, un effort particulier a été porté sur la **qualité du code**, "
        "la **maintenabilité** et la **reproductibilité**.\n\n"
        "Le pipeline mis en place vise donc à garantir un socle technique propre et fiable, "
        "plutôt qu’un déploiement industriel complet."
    )
    st.divider()

    # 2. Implemented CI pipeline (GitHub Actions)
    st.markdown(
        "## 2. Implemented CI pipeline (GitHub Actions)\n\n"
        "Pipeline CI actuellement en place, exécuté automatiquement sur chaque push et pull request."
    )
    st.text_area(
        "Pipeline CI implémenté",
        value=(
            "- Linting Python avec **pylint** (fail si score < 8)\n"
            "- Tests unitaires avec **pytest**\n"
            "- Mesure de couverture de code (coverage.xml)\n"
            "- Analyse statique et qualité globale via **SonarCloud**"
        ),
        height=120,
        key="cicd_existing"
    )
    st.divider()

    # 3. Linting & code quality
    st.markdown(
        "## 3. Linting & code quality\n\n"
        "Contrôle de la qualité du code afin de garantir lisibilité, cohérence et respect des bonnes pratiques."
    )
    st.text_area(
        "Linting",
        value=(
            "- Outil : pylint\n"
            "- Analyse par sous-dossier (src, features, utils, notebooks)\n"
            "- Seuil bloquant : score minimal de 8/10\n"
            "- Objectif : éviter dette technique et code fragile"
        ),
        height=120,
        key="cicd_lint"
    )
    st.divider()

    # 4. Unit tests & functional validation
    st.markdown(
        "## 4. Unit tests & functional validation\n\n"
        "Validation automatique du bon fonctionnement du code."
    )
    st.text_area(
        "Tests unitaires",
        value=(
            "- Framework : pytest\n"
            "- Tests exécutés après succès du lint\n"
            "- Couverture minimale requise : 30%\n"
            "- Objectif : détecter régressions et erreurs fonctionnelles"
        ),
        height=120,
        key="cicd_tests"
    )
    st.divider()

    # 5. Static analysis & SonarCloud
    st.markdown(
        "## 5. Static analysis & SonarCloud\n\n"
        "Surcouche d’analyse statique pour centraliser qualité, dette technique et couverture."
    )
    st.text_area(
        "SonarCloud",
        value=(
            "- Analyse automatique du code Python\n"
            "- Intégration du rapport de couverture pytest\n"
            "- Détection de code smells, duplications et vulnérabilités\n"
            "- Version gratuite : analyses limitées mais suffisantes pour un projet académique"
        ),
        height=130,
        key="cicd_sonar"
    )
    st.divider()

    # 6. Artefacts generated
    st.markdown(
        "## 6. Generated artefacts\n\n"
        "Fichiers produits automatiquement par le pipeline CI."
    )
    st.text_area(
        "Artefacts",
        value=(
            "- coverage.xml (rapport de couverture pytest)\n"
            "- Rapports SonarCloud\n"
            "- Logs CI GitHub Actions\n"
            "- Artefacts temporaires stockés pour inspection"
        ),
        height=110,
        key="cicd_artefacts"
    )
    st.divider()

    # 7. What is intentionally missing
    st.markdown(
        "## 7. What is intentionally missing\n\n"
        "Éléments volontairement non implémentés à ce stade."
    )
    st.text_area(
        "Limites assumées du pipeline",
        value=(
            "- Pas de build Docker\n"
            "- Pas de déploiement Kubernetes\n"
            "- Pas de CD (Continuous Deployment)\n"
            "- Pas de monitoring temps réel en production\n\n"
            "Ces éléments ne sont ni demandés dans le cadre du projet, "
            "ni compatibles avec les contraintes de coût, d’infrastructure "
            "et de périmètre pédagogique."
        ),
        height=160,
        key="cicd_limits"
    )
    st.divider()

    # 8. Perspective & training alignment
    st.markdown(
        "## 8. Perspective & training alignment\n\n"
        "Lien entre ce pipeline et la suite logique de la formation."
    )
    st.text_area(
        "Évolutions possibles",
        value=(
            "- Introduction progressive de Docker\n"
            "- Séparation CI / CD\n"
            "- Déploiement contrôlé (staging uniquement)\n"
            "- Monitoring basique (logs, métriques)\n\n"
            "Ces évolutions correspondent aux modules avancés de la formation "
            "et nécessitent des compétences et ressources supplémentaires."
        ),
        height=160,
        key="cicd_future"
    )
    st.divider()

    # 9. Summary & positioning
    st.markdown(
        "## 9. Summary & positioning\n\n"
        "Positionnement clair du pipeline dans le cadre du projet."
    )
    st.markdown(
        "- Pipeline **CI solide et fonctionnel**, orienté qualité et fiabilité\n"
        "- Implémenté volontairement au-delà des exigences minimales\n"
        "- Adapté à un projet académique avancé\n"
        "- Base saine pour une industrialisation future, sans sur-promesse"
    )


# STATUS: page/09_cicd.py — intégrale,
# pipeline CI réel documenté (lint, tests, couverture, SonarCloud),
# positionnement honnête, sans déploiement fictif,
# valorisation claire des compétences CI/CD sans bullshit.
