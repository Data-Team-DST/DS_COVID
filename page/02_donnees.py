# 02_donnees.py — version enrichie focalisée sur volumétrie, architecture, dictionnaire, QA et CI
# Theming metadata:
# - Preferred: streamlit-extras optional; inherits global dark theme from app
# - Palette: navy/dark background, high-contrast highlights; sans-serif font.
# - File status: upgraded data presentation template — volumétrie, lineage, data dictionary, QA.
# - Note: compatible with streamlit-extras v0.7.8 (uses colored_header and badge where available).

import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.badges import badge

_CSS = """
<style>
.data-hero {
  background: linear-gradient(135deg, rgba(6,12,24,0.85), rgba(10,18,34,0.85));
  padding: 12px;
  border-radius: 10px;
  color: #e6eef6;
  margin-bottom: 8px;
}
.small-note { font-size:12px; color:#98a1b3; }
.section-card {
  background: linear-gradient(90deg, rgba(12,18,30,0.95), rgba(8,12,20,0.95));
  padding: 12px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.03);
  color: #cfe8ff;
  margin-bottom: 10px;
}
.label { font-weight:700; color:#cfe8ff; margin-bottom:6px; }
.code-box { background:#071024; padding:8px; border-radius:6px; font-family:monospace; color:#b8d6ff; }
</style>
"""

def _render_section(title: str, body: str, container=st):
    container.markdown(f"<div class='section-card'><div class='label'>{title}</div><div>{body}</div></div>", unsafe_allow_html=True)

def run():
    st.markdown(_CSS, unsafe_allow_html=True)

    # header
    try:
        colored_header(
            label="📦 Présentation des données",
            description="Volumétrie, architecture, dictionnaire, accès, qualité et exemples reproductibles.",
            color_name="blue-70"
        )
    except Exception:
        st.markdown("### 📦 Présentation des données\n*Volumétrie, architecture, dictionnaire, accès, qualité et exemples reproductibles.*")

    st.divider()

    # 1. Topic overview & context
    st.markdown("## 1. Rôle des données & périmètre")
    st.markdown(
        "Décrire brièvement le rôle métier des données (ex : support au diagnostic), la période couverte, "
        "la granularité et la fréquence d'usage. Indiquer les décisions que les données doivent aider à prendre."
    )
    st.divider()

    # 2.A Inventory synthétique des sources
    st.markdown("## 2. Data intro (sources, volumétrie, architecture)")
    st.markdown("### A — Inventaire synthétique des sources")
    _render_section(
        "Inventaire (synthétique)",
        "<strong>Format</strong> : <em>Source | Type | Propriétaire | Fréquence | Volume estimé | Accès</em><br>"
        "Ex : <code>cxr_images_raw</code> | images + metadata | équipe projet | snapshot | ~N images | S3 / local"
    )
    st.divider()

    # 2.B Volumétrie détaillée
    st.markdown("### B — Volumétrie (à renseigner)")
    cols_v = st.columns(3)
    with cols_v[0]:
        st.text_input("Nom source (ex: cxr_images_raw)", key="data_vol_src", value=st.session_state.get("data_vol_src","cxr_images_raw"))
    with cols_v[1]:
        st.text_input("Nombre d'images (est.)", key="data_vol_images", value=st.session_state.get("data_vol_images","--"))
    with cols_v[2]:
        st.text_input("Taille estimée (GB)", key="data_vol_size", value=st.session_state.get("data_vol_size","--"))
    st.markdown("**Conseil** : fournir un snapshot anonymisé si possible (échantillon) pour la reproductibilité.")
    st.divider()

    # 2.C Architecture & lineage
    st.markdown("### C — Architecture & lineage (schéma logique)")
    st.text_area("Schéma / lineage (raw -> clean -> features -> models)", value=st.session_state.get("data_lineage","raw/images -> preprocess -> train/val/test splits -> models"), height=80, key="data_lineage")
    st.divider()

    # 2.D Data dictionary & sample schema
    st.markdown("### D — Data dictionary & sample schema")
    st.markdown("Fournir pour chaque fichier/CSV : colonne | type | description | exemples | contraintes.")
    st.text_area("Dictionnaire / Schéma (col, type, description, example)", value=st.session_state.get("data_dictionary","image_id | str | identifiant image | img_0001.jpg\nlabel | str | étiquette (COVID/Non-COVID) | COVID"), height=140, key="data_dictionary")
    st.markdown("**Snippet utile (pandas)** — génère un mini-dictionnaire (copy/paste dans ton repo).")
    st.expander("Afficher snippet pandas (copy/paste)").markdown(
        """```python
# snippet (not executed here)
import pandas as pd
df = pd.read_csv('sample_labels.csv')
schema = pd.DataFrame({
  'col': df.columns,
  'dtype': df.dtypes.astype(str),
  'null_rate': df.isna().mean(),
  'n_unique': df.nunique()
})
print(schema)
```"""
    )
    st.divider()

    # 3. Accès, sécurité & gouvernance
    st.markdown("## 3. Accès, sécurité & gouvernance")
    st.text_area("Accès & contraintes (endpoints, roles, masking)", value=st.session_state.get("data_access","Ex : accès S3 read-only pour l'équipe projet; PII must be masked"), height=100, key="data_access")
    st.markdown("- Checklist : accès testés ✓, masking documenté ✓, DPO contacté si nécessaire.")
    try:
        badge(type="info", text="Vérifier conformité RGPD & accès")
    except Exception:
        st.markdown("<div class='small-note'>Vérifier conformité RGPD & accès</div>", unsafe_allow_html=True)
    st.selectbox("DPO contacté ?", ["Non", "Oui (Nom)"], key="data_dpo", index=0)
    st.divider()

    # 4. Data quality & metrics
    st.markdown("## 4. Data quality & monitoring (nulls, uniqueness, freshness, drift)")
    st.text_area("Checks implémentés / thresholds", value=st.session_state.get("data_q_checks","- null_rate < 5%\n- duplicates < 1%"), height=100, key="data_q_checks")
    st.expander("QA snippet (pandas) — copy/paste").markdown(
        """```python
# Example QA snippet
import pandas as pd
df = pd.read_csv('sample.csv')
report = {
  'rows': len(df),
  'cols': len(df.columns),
  'null_rate': df.isna().mean().to_dict(),
  'duplicates': df.duplicated().mean()
}
print(report)
```"""
    )
    st.divider()

    # 5. Sample preview & reproducibility
    st.markdown("## 5. Aperçu d'échantillon & reproductibilité")
    st.text_input("Chemin échantillon / snapshot (DVC / S3 / URL)", value=st.session_state.get("data_sample_path","s3://bucket/project/snapshots/sample_2025-01-01.csv"), key="data_sample_path")
    st.markdown("**Astuce** : fournir un `scripts/inspect_data.py` minimal qui produit le mini-dictionnaire et un HTML report dans CI.")
    st.divider()

    # 6. Relation with modelling & temporal considerations
    st.markdown("## 6. Jeux pour modélisation (train / val / test) & logique de split")
    st.text_area("Logique de split & justification (chronological / patient-level)", value=st.session_state.get("data_split_logic","Chronological split / patient-level split si metadata disponible"), height=100, key="data_split_logic")
    st.divider()

    # 7. Contraintes & risques détaillés (déplacés ici depuis Accueil)
    st.markdown("## 7. Contraintes & risques (détaillés)")
    st.markdown(
        "Documentez ici les risques et contraintes spécifiques aux données :\n"
        "- **Biais d'échantillonnage** (source, sélection)\n"
        "- **Hétérogénéité des annotations / labels** (inter-observateur)\n"
        "- **Problèmes de qualité d'image** (artefacts, résolutions variées)\n"
        "- **Conformité & anonymisation** (PII, métadonnées DICOM)\n\n"
        "Indiquez les mesures d'atténuation (ex : reweighting, augmentation stratifiée, anonymisation pipeline)."
    )
    st.text_area("Notes contraintes / risques (à compléter)", value=st.session_state.get("data_constraints",""), height=120, key="data_constraints")
    st.divider()

    # 8. Artefacts recommandés & CI integration
    st.markdown("## 8. Artefacts recommandés & intégration CI")
    st.text_area("Artefacts & jobs CI", value=st.session_state.get("data_artifacts","schema.json\nsample_anonymized.csv\ndata_report.html"), height=80, key="data_artifacts")
    st.markdown("**Recommandation** : stocker un snapshot DVC et un script `scripts/inspect_data.py` exécuté dans CI.")
    st.divider()

    # 9. Summary & next actions (priorités)
    st.markdown("## 9. Résumé & prochaines actions (prioriser)")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.text_input("Action 1 (haute)", value=st.session_state.get("data_next_1","Fournir snapshot DVC"), key="data_next_1")
    with a2:
        st.text_input("Action 2 (moyenne)", value=st.session_state.get("data_next_2","Documenter dictionnaire"), key="data_next_2")
    with a3:
        st.text_input("Action 3 (basse)", value=st.session_state.get("data_next_3","Automatiser QA en CI"), key="data_next_3")
    st.divider()

    # Footer guidance
    st.markdown("<small class='small-note'>Status: template amélioré — remplissez les champs pour obtenir une section 'Présentation des données' complète et professionnelle.</small>", unsafe_allow_html=True)

# STATUS: page/02_donnees.py — focalisé sur volumétrie, dictionnaire, QA et CI (compatible streamlit-extras v0.7.8).
