# streamlit_app.py — version UX/visuelle ultime, stable et immersive
import streamlit as st
import importlib.util
from pathlib import Path
from typing import List

# --- Liste des pages ---
_PAGE_DIR = Path(__file__).parent / "page"
_PAGE_FILENAMES = [
    "01_accueil.py",
    "02_donnees.py",
    "03_analyse_visualisations.py",
    "04_preprocessing.py",
    "05_modeles.py",
    "06_analyse_du_meilleur_modele.py",
    "09_cicd.py",
    "07_conclusion_critique_perspective.py",
    "10_Selection_Pipeline.py",
]

# --- Import dynamique des modules ---
_loaded_pages = []
_import_errors = []
for fname in _PAGE_FILENAMES:
    fpath = _PAGE_DIR / fname
    module_name = f"page_{fname.replace('.', '_')}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(fpath))
        if spec is None or spec.loader is None:
            raise ImportError("spec ou loader est None")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "run"):
            raise AttributeError(f"Fonction `run()` absente dans {fname}")
        _loaded_pages.append((fname, mod))
    except Exception as e:
        _import_errors.append((fname, str(e)))

if _import_errors:
    st.set_page_config(page_title="Erreur d'import", layout="centered")
    for fname, err in _import_errors:
        st.error(f"Impossible d'importer `{fname}` — {err}")
    st.stop()

# --- Page config ---
st.set_page_config(page_title="Projet DS COVID - Streamlit", layout="wide", page_icon="🧪")

# --- CSS sombre + UX/transitions ---
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Roboto", "Montserrat", sans-serif;
    background-color: #071022 !important;
    color: #e6eef6 !important;
}

/* Hero header animé avec gradient selon onglet actif */
.hero-header {
    padding: 20px;
    margin-bottom: 18px;
    border-radius: 12px;
    font-size: 26px;
    font-weight: 600;
    text-align: center;
    background-size: 200% 200%;
    color: #fff;
    animation: gradientShift 6s ease infinite, pulse 2s infinite;
}

/* Animation gradient et pulse */
@keyframes gradientShift {
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}
@keyframes pulse {
    0%{transform:scale(1);}
    50%{transform:scale(1.02);}
    100%{transform:scale(1);}
}

/* Onglets stylés avec active glow */
.stTabs [role="tablist"] button {
    padding: 8px 14px;
    border-radius: 6px 6px 0 0;
    margin-right: 3px;
    font-weight:600;
    transition: all 0.3s ease;
}
.stTabs [role="tablist"] button[aria-selected="true"] {
    background: linear-gradient(90deg, #0f204f, #1a296b);
    color:#fff !important;
    box-shadow: 0 3px 8px #00000080;
}
.stTabs [role="tablist"] button:hover {
    transform: scale(1.03);
}

/* Card style + hover */
.card { 
    background-color: #131a2b; 
    padding: 14px; 
    margin: 10px 0; 
    border-radius: 10px; 
    box-shadow: 1px 1px 8px #00000080;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 2px 4px 12px #000000a0;
}

/* Badge animé */
.badge {
    display:inline-block;
    background:#0f204f;
    color:#fff;
    padding: 2px 6px;
    border-radius:8px;
    margin-left:6px;
    font-size:11px;
    animation: pulseBadge 1.5s infinite alternate;
}
@keyframes pulseBadge {
    0%{transform: scale(1);}
    100%{transform: scale(1.2);}
}
</style>
""", unsafe_allow_html=True)

# --- Navigation onglets ---
_nav_labels = [
    "01 - Accueil",
    "02 - Données",
    "03 - Analyse & Visu",
    "04 - Preprocessing",
    "05 - Modèles",
    "06 - Analyse meilleur modèle",
    "07 - CI/CD",
    "08 - Conclusion, critique & perspectives",
    "09 - Sélection Pipeline",
]

# --- Badges simples via emoji ---
_badges_emojis = ["🔹", "🟢", "🔴", "🟡", "🟣", "🟠", "🔵", "⚡", "⭐", "🔬"]

# Onglets avec badges emoji
_nav_labels_with_badges = [f"{label} {_badges_emojis[i]}" for i, label in enumerate(_nav_labels)]

# Création onglets
tabs = st.tabs(_nav_labels_with_badges)

for idx, tab in enumerate(tabs):
    fname, mod = _loaded_pages[idx]
    with tab:
        # Hero header dynamique
        gradients = [
            "linear-gradient(90deg, #0f204f, #071022)",
            "linear-gradient(90deg, #1b2a50, #0a1020)",
            "linear-gradient(90deg, #2a3b6c, #071022)",
            "linear-gradient(90deg, #10203f, #0f204f)",
            "linear-gradient(90deg, #1a204f, #0f1830)",
            "linear-gradient(90deg, #071022, #1a296b)",
            "linear-gradient(90deg, #0a1a30, #071022)",
            "linear-gradient(90deg, #142850, #0f204f)",
            "linear-gradient(90deg, #0f204f, #071022)",
            "linear-gradient(90deg, #1a3050, #0f1a35)"
        ]
        st.markdown(
            f"<div class='hero-header' style='background:{gradients[idx]}'>{_nav_labels[idx]}</div>",
            unsafe_allow_html=True
        )
        try:
            mod.run()
            st.markdown(
                f"<div style='font-size:11px;color:#98a1b3;margin-top:6px;'>Page chargée : `{fname}` — run() OK.</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Erreur `run()` dans `{fname}` : {e}")
            st.markdown(
                "<div style='font-size:12px;color:#d88;'>Vérifiez la fonction run() dans le module.</div>",
                unsafe_allow_html=True
            )
