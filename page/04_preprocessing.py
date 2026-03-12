# 04_preprocessing.py
# Theming metadata:
# - Preferred: streamlit-extras mandatory; page inherits app-level dark theme.
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: expert-grade preprocessing template focusing on "description et justification des traitements" — placeholders, checklists, pseudo-code et exigences CI.

import streamlit as st
from streamlit_extras.colored_header import colored_header

# Structure comments:
# - Expose only run().
# - Cette page documente chaque transformation appliquée (quoi, pourquoi, comment, impact, tests).
# - Tous les blocs sont placeholders/lorem à remplacer par des descriptions projet-réelles.

# Imports systèmes
import subprocess
import os
import sys

# --------------- Helpers ---------------
def run_shell_cmd(cmd: list, timeout: int = 6):
    """Execute a shell command and return (ok, output)."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return True, out.strip()
    except subprocess.CalledProcessError as e:
        return False, (e.output.strip() if hasattr(e, "output") and e.output else str(e))
    except Exception as e:
        return False, str(e)

def check_nvidia_smi():
    ok, out = run_shell_cmd(["nvidia-smi"])
    return ok, out

def check_torch():
    try:
        import importlib
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            return False, "torch package not found"
        import torch
        cuda_avail = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_avail else 0
        device_name = torch.cuda.get_device_name(0) if cuda_avail and device_count > 0 else "N/A"
        summary = {
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(cuda_avail),
            "cuda_device_count": int(device_count),
            "device_name_0": device_name
        }
        return True, summary
    except Exception as e:
        return False, str(e)

# --------------- Run UI ---------------
def run():
    st.set_page_config(page_title="Preprocessing — Description & Justification", layout="wide")
    colored_header(
        label="Preprocessing — Description & Justification",
        description="Documenter chaque transformation : quoi, pourquoi, comment, impact et tests associés.",
        color_name="blue-70"
    )
    st.divider()

    # Left / Right layout for system checks and doc template
    col_sys, col_doc = st.columns([1, 2])

    # ========= System checks (col_sys) =========
    with col_sys:
        st.subheader("Vérifications système")
        st.markdown("Bouton pour (re)lancer les checks système utiles avant calculs lourds (GPU / torch).")

        # interactive run button
        if st.button("Relancer vérifs système (nvidia-smi, torch, env)"):
            st.session_state["_preproc_last_checks"] = None  # force refresh below

        # cached container for results within the page session
        if "_preproc_last_checks" not in st.session_state or st.session_state["_preproc_last_checks"] is None:
            checks = {}
            # 1) nvidia-smi
            ok_smi, out_smi = check_nvidia_smi()
            checks["nvidia-smi"] = {"ok": ok_smi, "output": out_smi}

            # 2) torch
            ok_torch, out_torch = check_torch()
            checks["torch"] = {"ok": ok_torch, "output": out_torch}

            # 3) env var
            checks["env"] = {"NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"), "PYTHON_VERSION": sys.version}

            st.session_state["_preproc_last_checks"] = checks
        else:
            checks = st.session_state["_preproc_last_checks"]

        # Render results
        st.markdown("**nvidia-smi**")
        if checks["nvidia-smi"]["ok"]:
            st.success("nvidia-smi exécuté avec succès.")
            st.code(checks["nvidia-smi"]["output"])
        else:
            st.warning("nvidia-smi non disponible ou erreur.")
            st.code(checks["nvidia-smi"]["output"])

        st.markdown("**PyTorch**")
        if checks["torch"]["ok"]:
            st.success("torch détecté.")
            # pretty print dict
            info = checks["torch"]["output"]
            st.json(info)
        else:
            st.error("PyTorch absent ou erreur d'import.")
            st.text(checks["torch"]["output"])
            st.markdown(
                "Si tu veux exécuter des modèles localement :\n"
                "- Sur Linux local : `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` (CPU-only) ou installer la version CUDA adaptée.\n"
                "- Sur serveur distant : vérifier présence de GPU exposé et drivers NVIDIA + installer wheel compatible CUDA."
            )

        st.markdown("**Variables d'environnement**")
        st.write(f"NVIDIA_VISIBLE_DEVICES: `{checks['env']['NVIDIA_VISIBLE_DEVICES']}`")
        st.write(f"Python: `{checks['env']['PYTHON_VERSION']}`")

        st.markdown("---")
        st.markdown(
            "**Conseil rapide** :** si tu as besoin d'embeddings/Grad-CAM en production, pré-calculer les embeddings en local/GPU et stocker le fichier `.npz`/`.json` pour que Streamlit ne calcule rien en live."
        )

    # ========= Documentation template (col_doc) =========
    with col_doc:
        # 1. Topic overview & context
        st.markdown(
            "## 1. Topic overview & context\n\n"
            "**Objectif du preprocessing (résumé 1–2 lignes)** : fiabiliser les features, éviter leakage, "
            "et produire pipelines réutilisables pour entraînement et production.\n"
        )
        st.divider()

        # 2. Data intro (sources, volume, structure)
        st.markdown(
            "## 2. Data intro (sources, volume, structure)\n\n"
            "Rappeler les jeux impactés par ces transformations (raw → staging → features). "
            "Indiquer la volumétrie cible et la fenêtre temporelle utilisée pour validation.\n"
            "- **Sources impliquées** : `raw/events`, `raw/users`, `external/geo`\n"
            "- **Volumétrie indicative** : ex. ~120M rows / mois\n"
            "- **Période** : ex. 2018-01 → 2024-12"
        )
        st.divider()

        # 3. Data analysis & visualizations
        st.markdown(
            "## 3. Data analysis & visualizations\n\n"
            "Diagnostics utilisés pour décider des transformations : distributions pré/post, taux de NA, "
            "proportion d'outliers, corrélations avec la cible. Lister figures de référence (liens/artifacts).\n"
            "- **Figures utiles** : histogrammes (avant/après), boxplots, heatmaps corrélation, série NA rate par source."
        )
        st.divider()

        # 4. Preprocessing description & rationale
        st.markdown(
            "## 4. Preprocessing description & rationale\n\n"
            "**Pour chaque étape ci-dessous, remplir les champs :** "
            "`Nom étape | Objectif | Action exacte | Paramètres | Rationale métier/technique | Impact attendu | Tests associés | Artefacts produits`."
        )

        st.markdown("### Modèle d'entrée (copier pour chaque étape)")
        st.markdown(
            "- **Nom étape** : (ex. Imputation_date)\n"
            "- **Objectif** : (ex. réduire bias dû à valeurs manquantes sur feature X)\n"
            "- **Action exacte** : (ex. `df['col'] = df.groupby('segment')['col'].transform(lambda s: s.fillna(s.median()))`)\n"
            "- **Paramètres** : (ex. strategy=median, group_by='segment')\n"
            "- **Rationale (pourquoi)** : (ex. median robuste vs mean; découpage par segment préserve structure métier)\n"
            "- **Impact attendu** : (ex. réduction du NA rate < 5%; meilleure calibration du modèle)\n"
            "- **Tests associés** : (ex. null_rate target group, distribution shift pre/post, unit test for edge cases)\n"
            "- **Artefacts produits** : (ex. `staging/table_v2`, log transform parameters, schema.json)"
        )
        st.divider()

        st.markdown("### Étapes documentées (exemples — compléter)")
        st.markdown(
            "1. **Deduplication** — supprimer doublons stricts sur `event_id` ; justification : éviter surcomptage dans features d'usage.\n"
            "2. **Imputation ciblée** — colonnes `income`, `age` : imputation median per `region` ; justification : éviter biais socio-démographique.\n"
            "3. **Cap & floor (winsorization)** — outliers sur `amount` à 1% / 99% ; justification : robustesse des moyennes.\n"
            "4. **Feature engineering** — rolling windows 7/30 jours pour `events_count` ; justification : capturer comportement récent.\n"
            "5. **Encodage** — target encoding pour catégories cardinales élevées (avec K-fold leakage control) ; justification : performance vs sparsity."
        )
        st.divider()

        # Snippets & pseudo-code
        st.markdown("### Snippets & pseudo-code (copiable)")
        st.expander("Deduplication (pseudo)").markdown(
            """```python
# Pseudo
# input: raw/events.csv
df = read_raw("events")
df = df.drop_duplicates(subset=["event_id"])
write_staging(df, "events_clean_v1")
```"""
        )
        st.expander("Imputation groupée (pandas pseudo)").markdown(
            """```python
# Imputation median per group
df['col'] = df.groupby('region')['col'].transform(lambda s: s.fillna(s.median()))
```"""
        )
        st.expander("Target encoding safe (K-fold)").markdown(
            """```python
# Target encoding with out-of-fold to avoid leakage
for fold in cv_folds:
    train_oof = train[train.fold != fold]
    mapping = train_oof.groupby('cat')['target'].mean()
    val.loc[val.fold==fold, 'cat_te'] = val.loc[val.fold==fold, 'cat'].map(mapping)
```"""
        )
        st.divider()

        # 5. Model summary/results
        st.markdown(
            "## 5. Model summary/results\n\n"
            "Documenter les effets chiffrés du pipeline : delta métriques (avant vs après), variance reduction, "
            "temps d'entraînement ajouté.\n"
            "Exemple de table à remplir : `Étape | Metric_before | Metric_after | Δ | Notes`."
        )
        st.text_area("Tableau impact (coller CSV/Markdown)", value="", height=80, key="preproc_impact_table")
        st.divider()

        # 6. Best model analysis (transformations finales)
        st.markdown(
            "## 6. Best model analysis\n\n"
            "Préciser le pipeline final utilisé pour le meilleur modèle (ordre exact des transformations, versions des scripts, seeds). "
            "Indiquer si certaines étapes sont appliquées différemment en production (ex : approximations pour latence)."
        )
        st.text_input("Référence pipeline final (module/script & version)", value="", key="preproc_final_ref")
        st.divider()

        # 7. Conclusions & business relevance
        st.markdown(
            "## 7. Conclusions & business relevance\n\n"
            "Récapituler comment chaque transformation soutient un enjeu métier (ex : réduction des faux positifs, amélioration calibration). "
            "Indiquer les trade-offs (coût compute vs gain métrique)."
        )
        st.text_area("Conclusions synthétiques (3 bullets max)", value="- ...", height=80, key="preproc_conclusions")
        st.divider()

        # 8. Critique & future perspectives
        st.markdown(
            "## 8. Critique & future perspectives\n\n"
            "Limitations des traitements actuels (ex : imputations biaisées, risque de leakage via features temporelles). "
            "Pistes d'amélioration (feature store, tests, monitoring, enrichissements externes)."
        )
        st.text_area("Backlog améliorations & risques", value="- ...", height=100, key="preproc_critique")
        st.divider()

        # 9. CI/CD pipeline overview
        st.markdown(
            "## 9. CI/CD pipeline overview\n\n"
            "**Tests à automatiser** :\n"
            "- Schema tests (schema.json vs expected)\n"
            "- Null-rate tests (per column thresholds)\n"
            "- Distribution shift tests (KL / Earth Mover) pre/post\n"
            "- Unit tests for transform functions (edge cases)\n"
            "- Non-regression tests for metrics (compare to baseline tolerance)\n\n"
            "**Gating rules** : ex. `null_rate(col) < 5%`, `KL(col) < 0.2` for selected features.\n\n"
            "**Artifacts** : `staging/*.parquet`, `features/*.parquet`, `transform_params.json`, `preproc_log.json`.\n\n"
            "**Rollback & reproducibility** : stocker snapshot DVC/S3, versionner code + params, et fournir runbook pour rollback à une version antérieure en cas de régression."
        )
        st.divider()

        # Final checklist
        st.markdown("### Checklist opérationnelle (prête à cocher)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.checkbox("Toutes les étapes documentées", key="preproc_chk_doc")
        with c2:
            st.checkbox("Tests unitaires créés", key="preproc_chk_tests")
        with c3:
            st.checkbox("Artefacts versionnés (DVC/S3)", key="preproc_chk_artifacts")

        st.markdown(
            "<small style='color:#98a1b3'>Status: template expertisé — remplir chaque bloc d'étape avec la description précise et versionner les scripts / artefacts pour assurer traçabilité et CI.</small>",
            unsafe_allow_html=True
        )

# STATUS: page/04_preprocessing.py — version intégrale, Streamlit Extras obligatoire, sections 1–9 avec checklists, pseudo-code et placeholders.
