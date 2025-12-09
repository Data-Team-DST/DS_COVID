# Theming metadata:
# - Preferred: streamlit-extras mandatory; inherits global dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: interactive CI/CD template — documente pipeline actuel + futur, artefacts et monitoring.

import streamlit as st
from streamlit_extras.colored_header import colored_header

def run():
    # Header / hero
    colored_header(
        label="CI/CD Pipeline & Artefacts",
        description="Documenter l'état actuel et futur du pipeline CI/CD, artefacts générés, monitoring et plan de production.",
        color_name="blue-70"
    )
    st.divider()

    # 1. Topic overview & context
    st.markdown(
        "## 1. Topic overview & context\n\n"
        "Pipeline CI/CD actuel : linting, unit tests, couverture, SonarCloud.\n"
        "Pipeline futur : containerisation backend/frontend, déploiement Kubernetes, monitoring Grafana/Alertmanager."
    )
    st.divider()

    # 2. Existing CI Jobs
    st.markdown(
        "## 2. Existing CI Jobs\n\n"
        "Résumé du pipeline actuel (GitHub Actions) : lint, tests unitaires, coverage, SonarCloud."
    )
    st.text_area(
        "Jobs existants",
        value="- lint : pylint (fail if score < 8)\n- unit_tests : pytest + coverage\n- sonarqube : analyse SonarCloud",
        height=100,
        key="cicd_existing"
    )
    st.expander("Exemple YAML actuel").markdown(
        """```yaml
jobs:
  lint: ...
  unit_tests: ...
  sonarqube: ...
```"""
    )
    st.divider()

    # 3. Artefacts & metrics
    st.markdown(
        "## 3. Artefacts & metrics\n\n"
        "Documents et fichiers générés par le pipeline actuel et ceux prévus pour futur deployment."
    )
    st.text_area(
        "Artefacts & metrics",
        value="- coverage.xml\n- reports/coverage.xml\n- SonarCloud report\n- futur : docker images, manifests k8s, dashboards Grafana",
        height=100,
        key="cicd_artefacts"
    )
    st.divider()

    # 4. Planned CI/CD Extensions
    st.markdown(
        "## 4. Planned CI/CD Extensions\n\n"
        "Indique les étapes prévues non encore implémentées : build backend/frontend, containerisation, déploiement Kubernetes, monitoring et alerting."
    )
    st.text_area(
        "Futures étapes",
        value="- Backend FastAPI containerisé\n- Frontend Streamlit containerisé\n- Déploiement sur AWS via ArgoCD\n- Monitoring via Grafana + Alertmanager\n- Canary deploys & rollback",
        height=140,
        key="cicd_future"
    )
    st.divider()

    # 5. Secrets & access management
    st.markdown(
        "## 5. Secrets & access management\n\n"
        "Liste des secrets nécessaires (API keys, tokens, credentials) pour CI/CD et monitoring."
    )
    st.text_area(
        "Secrets",
        value="- SONAR_TOKEN\n- AWS_ACCESS_KEY / AWS_SECRET_KEY\n- DockerHub credentials\n- ArgoCD token",
        height=100,
        key="cicd_secrets"
    )
    st.divider()

    # 6. Gating & thresholds
    st.markdown(
        "## 6. Gating & thresholds\n\n"
        "Définir les règles pour promotion vers staging/prod : metrics minimales, fairness checks, smoke tests."
    )
    st.text_input(
        "Gating rules",
        value="AUC >= 0.80 AND lint pass AND unit tests pass",
        key="cicd_gate"
    )
    st.divider()

    # 7. Deployment & monitoring
    st.markdown(
        "## 7. Deployment & monitoring\n\n"
        "Checklists pour staging, production, rollbacks et alerting."
    )
    st.text_area(
        "Deployment plan",
        value="- Staging deployment via ArgoCD\n- Smoke tests automated\n- Production deployment with manual approval\n- Grafana dashboards setup\n- Alertmanager notifications",
        height=140,
        key="cicd_deploy"
    )
    st.divider()

    # 8. Critique & future perspectives
    st.markdown(
        "## 8. Critique & future perspectives\n\n"
        "Évaluer ce qui manque pour industrialiser complètement : pipelines front/back, tests E2E, scaling, observabilité complète."
    )
    st.text_area(
        "Backlog CI/CD",
        value="- Compléter pipeline frontend\n- End-to-end tests automatisés\n- Auto-scaling Kubernetes\n- Observabilité complète (logs, metrics, traces)",
        height=120,
        key="cicd_backlog"
    )
    st.divider()

    # 9. Summary & recommendations
    st.markdown(
        "## 9. Summary & recommendations\n\n"
        "Documenter le plan pour passer de l’état actuel à un pipeline complet prêt pour production."
    )
    st.markdown(
        "- Prioriser containerisation et ArgoCD\n"
        "- Ajouter monitoring Grafana/Alertmanager\n"
        "- Finaliser secrets management et gating rules\n"
        "- Mettre en place rollback et canary deployments"
    )

# STATUS: page/09_cicd.py — intégrale, Streamlit Extras obligatoire, interactive CI/CD template ready to document pipeline actuel + futur, artefacts et monitoring.
