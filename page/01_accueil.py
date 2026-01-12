# 01_accueil.py —
# - Header développé (4–8 lignes)
# - Objectifs SMART (S M A R T)
# - Pas de notes orateur / risques

import streamlit as st
from streamlit_extras.colored_header import colored_header
from pathlib import Path

st.set_page_config(
    page_title="Analyse de radiographies — Accueil",
    layout="wide",
    page_icon="analyse"
)

_CSS = """
<style>
:root{ --bg:#0f1115; --card:#131416; --muted:#9aa1a6; --accent:#4fc3f7; }
body, .stApp { background: var(--bg); color: #e6eef6; font-family: "Inter", sans-serif; }
.project-hero { background: linear-gradient(135deg, rgba(10,20,40,0.85), rgba(6,10,20,0.75)); padding: 18px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.5); color: #e6eef6; }
.card { background: linear-gradient(90deg, rgba(15,23,36,0.95), rgba(10,14,22,0.95)); padding: 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.03); box-shadow: 0 4px 10px rgba(0,0,0,0.35); color: #dbe9ff; margin-bottom:10px; }
.card-title { font-weight:700; font-size:14px; margin-bottom:6px; color:#cfe8ff; }
.card-body { font-size:13px; color:#98a7bf; }
.small-note { font-size:12px; color:#98a1b3; }
.kv { font-size:12px; color:var(--muted); }
abbr { text-decoration: none; border-bottom: 1px dotted rgba(255,255,255,0.12); cursor: help; }
@media (max-width:720px){ .project-hero { padding:12px; } .card { padding:10px; } }
</style>
"""


# --- Page
def run():
    st.markdown(_CSS, unsafe_allow_html=True)

    # Header (développé)
    try:
        colored_header(
            label="Analyse de radiographies pulmonaires — Classification COVID-19 et aide au diagnostic",
            description="Projet réalisé par Cirine B., Lena B., Steven M., Rafael C., encadré par Nicolas M.",
            color_name="blue-70"
        )
    except Exception:
        st.markdown(
            "<h2>Analyse de radiographies pulmonaires — Classification COVID-19</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='small-note'>Prototype d'assistance diagnostique visuelle — rapide, interprétable et prêt pour la démonstration.</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # Contexte étendu avec problématique et solutions
    st.markdown("### Contexte de l’épidémie de COVID-19")

    # Bilan mondial
    st.markdown(
        "<div class='project-hero'>"
        "<strong>Bilan mondial</strong><br>"
        "• Trois ans d’épidémie mondiale<br>"
        "• Plus de <strong>700 millions</strong> de cas confirmés<br>"
        "• Plus de <strong>7 millions</strong> de décès<br>"
        "• Type spécifique de pneumonie virale (<em>viral pneumonia</em>) causée par le virus <strong>SARS-CoV-2</strong>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("")

    # Problème 1 → Solution 1
    st.markdown(
        "<div class='project-hero'>"
        "<strong>Problème 1 : limites des tests <abbr title='Reverse Transcription PCR'>RT-PCR</abbr></strong><br>"
        "• Tests moléculaires <strong>lents</strong> (délais de traitement importants)<br>"
        "• Sensibilité <strong>variable</strong> (environ 70 % en conditions réelles contre 95 % théoriques)<br>"
        "• <strong>Pénuries de stocks</strong> et dépendance aux laboratoires<br>"
        "<br>"
        "<strong>Solution 1 : imagerie médicale complémentaire</strong><br>"
        "Utiliser des méthodes d’imagerie telles que la <abbr title='Chest X-Ray'>CXR</abbr> (radiographie thoracique) "
        "ou la <abbr title='Computed Tomography'>CT</abbr> (tomodensitométrie) pour un diagnostic plus rapide et plus accessible."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("")

    # Caractéristiques radiologiques du COVID-19 avec image
    img_col, text_col = st.columns([1, 2])

    with img_col:
        images_data = {
            "Fig1": (
                "Fig1.jpg",
                "Figure 1 — Opacité en verre dépoli. Radiographie thoracique postéro-antérieure d’un patient atteint de pneumonie COVID-19. "
                "Présence d’opacités en verre dépoli dans les zones moyennes et inférieures des deux poumons, principalement en périphérie "
                "(flèches blanches), avec préservation des marquages pulmonaires. Une opacité linéaire est visible à la périphérie de la zone moyenne gauche (flèche noire)."
            ),
            "Fig2": (
                "Fig2.jpg",
                "Figure 2 — Consolidation. Radiographie thoracique antéro-postérieure d’un patient atteint de pneumonie COVID-19 sévère, "
                "montrant une consolidation périphérique dense bilatérale et une perte des marquages pulmonaires dans les zones moyennes et inférieures."
            ),
            "Fig3": (
                "Fig3.jpg",
                "Figure 3 — Progression radiologique de la pneumonie COVID-19 chez un même patient. "
                "(a) Radiographie thoracique postéro-antérieure normale prise 12 mois avant l’admission. "
                "(b) Radiographie thoracique antéro-postérieure au jour 0 montrant des opacités en verre dépoli périphériques bilatérales. "
                "(c) Radiographie thoracique antéro-postérieure au jour 10 montrant une progression vers une pneumonie sévère avec consolidation dense."
            ),
            "Fig4": (
                "Fig4.jpg",
                "Figure 4 — Progression radiologique chez un patient atteint de COVID-19. "
                "(a) Radiographie thoracique antéro-postérieure normale au jour de l’admission. "
                "(b) Radiographie thoracique antéro-postérieure au jour 8 montrant une opacification en verre dépoli aux bases pulmonaires et une consolidation périphérique."
            ),
        }

        selected_fig = st.selectbox("Sélectionner une image :", list(images_data.keys()), index=0)

        current_img, current_caption = images_data[selected_fig]
        img_path = Path(__file__).parent / "images" / "covid_cxr_symptoms" / current_img

        if img_path.exists():
            st.image(str(img_path), caption=current_caption)
        else:
            st.info(f"Image {current_img} non disponible.")

    with text_col:
        st.markdown(
            "<div class='project-hero'>"
            "<strong>Signes radiologiques typiques du COVID-19 sur CXR</strong><br>"
            "Une pneumonie COVID-19 se caractérise généralement par :<br>"
            "• Des <strong>opacités bilatérales floues</strong>, souvent en <strong>périphérie</strong> ou dans les régions postérieures des poumons<br>"
            "• Une atteinte prédominante des <strong>lobes inférieurs</strong><br>"
            "• Un <strong>stade précoce</strong> marqué par des opacités diffuses légères<br>"
            "• Un <strong>stade avancé</strong> avec consolidation dense liée à un remplissage alvéolaire<br>"
            "<br>"
            "<strong>Problème clé :</strong> ces signes sont très proches de ceux observés dans d’autres pneumonies virales, "
            "rendant le diagnostic visuel complexe, même pour un radiologue expérimenté."
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("")

        st.markdown(
            "<div class='project-hero'>"
            "<strong>Problème 2 : difficulté du diagnostic visuel</strong><br>"
            "La similarité des patterns radiologiques complique la distinction entre COVID-19 et autres pneumonies.<br>"
            "<br>"
            "<strong>Problème 2.1 : surcharge des radiologues</strong><br>"
            "Le volume d’examens durant l’épidémie a généré une surcharge importante, réduisant le temps disponible par examen.<br>"
            "<br>"
            "<strong>Solution 2 : intelligence artificielle</strong><br>"
            "L’utilisation de modèles de machine learning et de deep learning permet d’accélérer la détection des cas positifs, "
            "tout en proposant une solution interprétable et reproductible pour un POC et une démonstration."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "La radiographie thoracique reste un examen rapide et largement accessible pour le triage initial des pathologies pulmonaires. "
        "Notre pipeline combine des modèles classiques de machine learning et un réseau convolutionnel deep learning "
        "(InceptionV3 affiné par fine-tuning) afin d’extraire des signaux diagnostiques robustes. "
        "Des méthodes d’interprétabilité visuelle (Grad-CAM) et basées sur les caractéristiques (SHAP) sont intégrées "
        "pour rendre les décisions du modèle plus transparentes et faciliter leur validation clinique. "
        "L’objectif est de proposer une solution équilibrée entre performance, explicabilité et faisabilité opérationnelle."
    )

    st.divider()

    # Objectifs SMART
    st.markdown("## Objectifs SMART (propositions retenues)")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Objectif 1 — Performance diagnostique")
        st.markdown(
            "- **S** : AUC ≥ **0,88** pour la classification COVID vs non-COVID (patient-level hold-out).\n"
            "- **M** : ROC AUC avec intervalle de confiance à 95 % par bootstrap.\n"
            "- **A** : fine-tuning d’InceptionV3 avec augmentation et repondération des classes.\n"
            "- **R** : priorité clinique permettant un triage plus rapide et fiable.\n"
            "- **T** : démontré en phase de validation."
        )

    with col2:
        st.subheader("Objectif 2 — Sensibilité et spécificité")
        st.markdown(
            "- **S** : Sensibilité ≥ **0,85** et spécificité ≥ **0,90**.\n"
            "- **M** : matrices de confusion, courbes ROC et PR.\n"
            "- **A** : calibration du modèle et ajustement du seuil.\n"
            "- **R** : réduction du risque de faux négatifs tout en contrôlant les faux positifs.\n"
            "- **T** : preuve opérationnelle pour la démonstration."
        )
