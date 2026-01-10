# 01_accueil.py — 
# - Header developpé (4-8 lignes)
# - Objectifs SMART (S M A R T)
# - Pas de notes orateur / risques

import streamlit as st
from streamlit_extras.colored_header import colored_header
from pathlib import Path

st.set_page_config(page_title="Analyse de Radiographies — Accueil", layout="wide", page_icon="🧪")

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
            label="Analyse de Radiographies pulmonaires — Classification COVID-19 & Aide au diagnostic",
            description=("Projet réalisé par Cirine B., Lena B., Steven M., Rafael C., Encadré par : Nicolas M."),
            color_name="blue-70"
        )
    except Exception:
        st.markdown("<h2>Analyse de Radiographies pulmonaires — Classification COVID-19</h2>", unsafe_allow_html=True)
        st.markdown("<div class='small-note'>Prototype d'assistance diagnostique visuelle — rapide, interprétable et prêt pour la démo.</div>", unsafe_allow_html=True)

    st.divider()

    # Contexte étendu avec problématique et solutions
    st.markdown("### 📊 Contexte de l'épidémie COVID-19")
    
    # Bilan mondial
    st.markdown(
        "<div class='project-hero'>"
        "<strong>🌍 Bilan mondial</strong><br>"
        "• 3 ans d'épidémie mondiale<br>"
        "• Plus de <strong>700 millions</strong> de cas confirmés<br>"
        "• Plus de <strong>7 millions</strong> de décès<br>"
        "• Type spécifique de pneumonie virale (<em>Viral Pneumonia</em>) causée par le virus <strong>SARS-CoV-2</strong>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown("")
    
    # Problème 1 → Solution 1
    st.markdown(
        "<div class='project-hero'>"
        "<strong>⚠️ Problème 1 : Limites des tests <abbr title='Reverse Transcription PCR'>RT-PCR</abbr></strong><br>"
        "• Tests moléculaires <strong>lents</strong> (délais de traitement importants)<br>"
        "• Sensibilité <strong>variable</strong> (~70% en conditions réelles vs 95% théoriques)<br>"
        "• <strong>Pénuries de stocks</strong> et dépendance aux laboratoires<br>"
        "<br>"
        "<strong>✅ Solution 1 : Imagerie médicale complémentaire</strong><br>"
        "Utiliser des méthodes d'imagerie telles que la <abbr title='Chest X-Ray'>CXR</abbr> (radiographie thoracique) "
        "ou la <abbr title='Computed Tomography'>CT</abbr> (tomodensitométrie) pour un diagnostic plus rapide et accessible."
        "</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("")

     # Caractéristiques radiologiques du COVID-19 avec image
    img_col, text_col = st.columns([1, 2])
    
    with img_col:
        # Initialiser l'index de l'image dans session_state
        if 'img_index' not in st.session_state:
            st.session_state.img_index = 0
        
        # Liste des images et leurs légendes
        images_data = [
            ("Fig1.jpg","Figure 1 — Opacité en verre dépoli. Radiographie thoracique postéro-antérieure d'un patient atteint de pneumonie COVID-19. "
                        "Les caractéristiques incluent une opacité en verre dépoli dans les zones moyennes et inférieures des deux poumons, "
                        "principalement en périphérie (flèches blanches) avec préservation des marquages pulmonaires. "
                        "Une opacité linéaire (zone blanche allongée et fine) est visible à la périphérie de la zone moyenne gauche (flèche noire)."),

            ("Fig2.jpg","Figure 2 — Consolidation. Radiographie thoracique antéro-postérieure (AP) d'un patient atteint de pneumonie COVID-19 sévère, "
                        "montrant une consolidation périphérique dense bilatérale et une perte des marquages pulmonaires dans les zones moyennes et inférieures (flèches délimitées)."),

            ("Fig3.jpg","Figure 3 — Progression radiologique des symptômes de la pneumonie covid-19 chez un même patient C. (a) Radiographie thoracique postéro-antérieure normale du patient C, "
                        "(prise 12 mois avant son admission à l'hôpital). (b) Radiographie thoracique AP du patient C lorsqu'il " 
                        "a développé une pneumonie covid-19 (jour 0 de l'admission), montrant des opacités en verre dépoli en périphérie " 
                        "(tiers externe du poumon) des deux poumons dans les zones moyennes et inférieures (flèches blanches), préservation des marquages pulmonaires, et opacité linéaire dans " 
                        "la périphérie de la zone moyenne gauche (flèche noire). (c) Radiographie thoracique AP du patient C au jour 10 de l'admission, montrant une progression vers une pneumonie" 
                        "covid-19 sévère : patient intubé avec tube endotrachéal, lignes centrales et sonde nasogastrique en place. Une consolidation dense avec perte des marquages pulmonaires est" 
                        "maintenant visible derrière le cœur dans la zone inférieure gauche (flèche délimitée). Une extension des modifications en verre dépoli périphériques vues en (b) " 
                        "peut être observée dans la périphérie des zones moyennes et inférieures droites et de la zone moyenne gauche (flèches blanches)."),

            ("Fig4.jpg","Figure 4 - Progression radiologique des symptômes de la pneumonie covid-19 chez un même patient D. (a) Radiographie thoracique antéro-postérieure normale du patient D, infecté par le COVID-19 (jour de l'admission). " 
                        ". (b) Radiographie thoracique antéro-postérieure du patient D au jour 8, montrant une opacification en verre dépoli maintenant présente aux deux bases pulmonaires (flèches blanches). " 
                        "Une consolidation est également visible dans la périphérie des zones supérieures et moyennes gauches (flèches délimitées). Une densité accrue (blancheur) est également présente dans la périphérie de la zone supérieure droite ; " 
                        "ceci n'est pas aussi dense ou blanc que ce qui est observé dans le poumon gauche, montrant la progression des modifications pulmonaires de l'opacification en verre dépoli à la consolidation (flèches délimitées).")

        ]
        
        # Navigation
        col_prev, col_counter, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("◀️", key="prev_img", use_container_width=True):
                st.session_state.img_index = (st.session_state.img_index - 1) % len(images_data)
        
        with col_counter:
            st.markdown(f"<div style='text-align: center; padding-top: 5px;'>{st.session_state.img_index + 1} / {len(images_data)}</div>", unsafe_allow_html=True)
        
        with col_next:
            if st.button("▶️", key="next_img", use_container_width=True):
                st.session_state.img_index = (st.session_state.img_index + 1) % len(images_data)
        
        # Afficher l'image actuelle
        current_img, current_caption = images_data[st.session_state.img_index]
        img_path = Path(__file__).parent / "images" / "covid_cxr_symptoms" / current_img
        
        if img_path.exists():
            st.image(str(img_path), caption=current_caption)
        else:
            st.info(f"💡 Image {current_img} non disponible.")
    
    with text_col:
        st.markdown(
            "<div class='project-hero'>"
            "<strong>🔬 Signes radiologiques typiques du COVID-19 sur CXR</strong><br>"
            "Sur une radiographie thoracique, une pneumonie COVID-19 présente généralement :<br>"
            "• <strong>Zones blanches floues (opacités en verre dépoli ou linéaires)</strong> visibles des <strong>deux côtés des poumons</strong>, "
            "souvent en <strong>périphérie</strong> (vers l'extérieur) ou à l'<strong>arrière</strong> des poumons<br>, "
            "qui masquent les marquages pulmonaires normaux (vaisseaux sanguins, etc) <br>"
            "• Localisation surtout dans la <strong>partie basse des poumons</strong> (lobes inférieurs)<br>"
            "• <strong>Début de la maladie</strong> : zones floues légères et diffuses<br>"
            "• <strong>Stade avancé</strong> : zones deviennent plus denses et blanches (consolidation = poumon rempli de liquide/cellules inflammatoires)<br>"
            "<br>"
            "⚠️ <strong>Problème clé</strong> : ces signes ressemblent beaucoup à d'autres pneumonies virales, ou a des opacités causées par d'autres maladies pulmonaires, "
            "rendant le diagnostic visuel très difficile même pour un expert."
            "</div>",
            unsafe_allow_html=True
        )
    
        st.markdown("")
        # Problème 2 → Solution 2
        st.markdown(
            "<div class='project-hero'>"
            "<strong>⚠️ Problème 2 : Difficulté du diagnostic visuel</strong><br>"
            "Même un radiologue expérimenté a du mal à <strong>distinguer</strong> un cas de COVID-19 "
            "d'un autre type de pneumonie sur une image radiographique, en raison de la similarité des patterns visuels.<br>"
            "<br>"
            "<strong>⚠️ Problème 2.1 : Surcharge des radiologues</strong><br>"
            "Pendant l'épidémie, les radiologues sont <strong>surchargés</strong> et n'ont pas le temps "
            "de lire toutes les radiographies avec l'attention nécessaire, créant un goulot d'étranglement diagnostique.<br>"
            "<br>"
            "<strong>✅ Solution 2 : Intelligence artificielle</strong><br>"
            "Utiliser <strong>ML/DL</strong> pour accélérer la détection des cas positifs depuis les images "
            "et fournir une solution <strong>interprétable</strong> et <strong>reproductible</strong> "
            "pour la démonstration et le POC (Proof of Concept)."
            "</div>",
            unsafe_allow_html=True
        )


    # Paragraphe central
    st.markdown(
        "La radiographie thoracique reste un examen rapide, simple d'accès et couramment utilisé dans le triage initial des pathologies pulmonaires. "
        "Notre pipeline combine des modèles classiques de machine learning et un réseau convolutionnel deep-learning (InceptionV3 fine-tuned) "
        "pour extraire des signaux diagnostiques robustes à partir des images. "
        "Nous ajoutons des méthodes d'interprétabilité visuelle (Grad-CAM) et feature-based (SHAP) afin de rendre la décision du modèle plus transparente "
        "pour le clinicien et de faciliter la validation visuelle des prédictions. "
        "L'approche vise une solution équilibrée entre performance, explicabilité et faisabilité opérationnelle, adaptée pour une démonstration en soutenance.",
        unsafe_allow_html=True
    )

    st.divider()

    # Objectifs SMART
    st.markdown("## Objectifs SMART (propositions retenues)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Objectif 1 — Performance diagnostique")
        st.markdown(
            "- **S** : AUC ≥ **0.88** sur classification COVID vs non-COVID (patient-level hold-out).\n"
            "- **M** : ROC AUC + IC 95% via bootstrap; rapport automatisé.\n"
            "- **A** : fine-tuning InceptionV3 + augmentation et reweighting.\n"
            "- **R** : priorite clinique — permet un triage plus rapide et fiable.\n"
            "- **T** : démontré en Phase Validation (S11)."
        )
    with col2:
        st.subheader("Objectif 2 — Sensibilité & Spécificité")
        st.markdown(
            "- **S** : Sensibilité ≥ **0.85** & Spécificité ≥ **0.90** (seuil choisi via ROC).\n"
            "- **M** : confusion matrix, ROC/PR, rapports par fold.\n"
            "- **A** : calibration du modèle et ajustement de seuils.\n"
            "- **R** : réduit le risque de faux négatifs tout en contrôlant faux positifs.\n"
            "- **T** : preuve opérationnelle pour la démo (S13)."
        )
