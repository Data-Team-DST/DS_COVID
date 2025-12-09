# 01_accueil.py — version finale : glossaire FR+EN dans un seul .rtf (Word-friendly)
# - Un seul bouton : .rtf contenant sections FR + EN (longues)
# - Glossaire court visible (FR)
# - Header developpé (4-8 lignes)
# - Objectifs SMART (S M A R T)
# - Pas de notes orateur / risques

import streamlit as st
from streamlit_extras.colored_header import colored_header
import datetime
import html

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

def _render_card(title: str, text: str, container=st):
    container.markdown(
        f"<div class='card'><div class='card-title'>{html.escape(title)}</div><div class='card-body'>{text}</div></div>", 
        unsafe_allow_html=True
    )

# Glossaire court (FR) visible
GLOSSARY_MIN_FR = {
    "CXR": "Radiographie thoracique — image standard, rapide.",
    "CT": "Tomodensitométrie — examen plus sensible, plus coûteux.",
    "RT-PCR": "Test moléculaire de référence pour détection d'ARN viral.",
    "AUC": "Aire sous la courbe ROC — métrique de discrimination.",
    "LIME": "Méthodes d'explicabilité locale pour modèles.",
    "SHAP": "Méthodes d'explicabilité basées sur valeurs de Shapley.",
    "Grad-CAM": "Carte visuelle indiquant régions d'intérêt pour CNN."
}

# Définitions longues FR
GLOSSARY_LONG_FR = {
    "CXR": (
        "La radiographie thoracique (CXR) est un examen d'imagerie de projection, "
        "généralement réalisé en vues frontale et/ou latérale, permettant d'examiner rapidement "
        "les poumons, le cœur et la cage thoracique. "
        "Elle est largement disponible, peu coûteuse et rapide à effectuer, ce qui la rend adaptée "
        "au triage initial des patients suspectés d'atteinte pulmonaire. "
        "Même si sa sensibilité est inférieure à celle de la tomodensitométrie pour certaines lésions, "
        "la CXR demeure une étape clinique essentielle dans de nombreux flux de diagnostic."
    ),
    "CT": (
        "La tomodensitométrie (CT) produit des images en coupes successives du thorax avec une résolution spatiale élevée. "
        "Elle permet une exploration très détaillée des structures pulmonaires et cardiaques, détectant souvent "
        "des anomalies que la radiographie simple peut manquer. "
        "Son emploi est toutefois limité par la disponibilité de l'appareillage, le coût et l'exposition "
        "accumulée aux radiations ionisantes pour le patient."
    ),
    "RT-PCR": (
        "La RT-PCR est une méthode moléculaire qui convertit l'ARN viral en ADN complémentaire puis amplifie "
        "des séquences cibles afin de détecter la présence du virus. "
        "C'est la méthode de référence pour le diagnostic de la COVID-19 en raison de sa sensibilité analytique élevée. "
        "Ses limites pratiques incluent le temps de traitement, la dépendance aux conditions d'échantillonnage et "
        "la nécessité d'infrastructures de laboratoire."
    ),
    "AUC": (
        "L'aire sous la courbe ROC (AUC) résume la performance discriminante d'un classifieur indépendamment "
        "du choix d'un seuil de décision. "
        "Elle représente la probabilité que le modèle classe correctement un exemple positif par rapport à un négatif. "
        "Une AUC élevée indique un fort pouvoir discriminant; l'AUC est utile pour comparer différents modèles."
    ),
    "LIME": (
        "LIME (Local Interpretable Model-agnostic Explanations) explique des décisions individuelles en "
        "construisant un modèle simple local autour d'une instance donnée. "
        "En perturbant légèrement les entrées et en observant la sensibilité de la prédiction, LIME fournit "
        "un sur-ensemble interprétable expliquant pourquoi le modèle a prédit d'une certaine manière pour cet exemple."
    ),
    "SHAP": (
        "SHAP (SHapley Additive exPlanations) s'appuie sur la théorie des jeux pour répartir de manière cohérente "
        "la contribution de chaque variable à une prédiction. "
        "SHAP fournit des explications locales précises et peut être agrégé pour obtenir une vue globale de l'importance "
        "des variables sur l'ensemble des prédictions."
    ),
    "Grad-CAM": (
        "Grad-CAM (Gradient-weighted Class Activation Mapping) produit des cartes de chaleur "
        "qui superposées à l'image indiquent les zones les plus influentes pour la décision d'un réseau convolutionnel. "
        "Cette visualisation aide à vérifier que le modèle se concentre sur des régions anatomiquement pertinentes et "
        "facilite l'interprétation qualitative des prédictions en imagerie."
    )
}

# Définitions longues EN
GLOSSARY_LONG_EN = {
    "CXR": (
        "Chest X-Ray (CXR) is a projection radiograph acquired in frontal and/or lateral views to visualize the lungs, heart and thoracic cage. "
        "It is fast, widely available and cost-effective, making it ideal for initial screening and triage in many clinical settings. "
        "While less sensitive than CT for certain lesions, CXR remains an essential first-line imaging modality in routine care."
    ),
    "CT": (
        "Computed Tomography (CT) provides cross-sectional imaging with high spatial resolution, enabling detailed assessment of pulmonary parenchyma. "
        "CT is highly sensitive for detecting small consolidations and interstitial disease, but requires specialized equipment and involves higher radiation exposure."
    ),
    "RT-PCR": (
        "Reverse Transcription Polymerase Chain Reaction (RT-PCR) is a molecular assay converting viral RNA to DNA and amplifying target sequences for detection. "
        "RT-PCR is the reference standard for diagnosing SARS-CoV-2 due to its high analytical sensitivity, though it requires laboratory infrastructure and time to return results."
    ),
    "AUC": (
        "Area Under the Receiver Operating Characteristic Curve (AUC-ROC) quantifies a classifier's discrimination ability across all thresholds. "
        "AUC values range from 0.5 (no discrimination) to 1.0 (perfect discrimination) and are commonly used to compare binary classifiers."
    ),
    "LIME": (
        "Local Interpretable Model-agnostic Explanations (LIME) approximates a complex model locally with a simple surrogate model to explain individual predictions. "
        "By perturbing the input and observing prediction changes, LIME offers human-readable explanations that are model-agnostic."
    ),
    "SHAP": (
        "SHapley Additive exPlanations (SHAP) uses Shapley values from cooperative game theory to assign additive contributions of each feature to a prediction. "
        "SHAP provides consistent and locally accurate explanations and can be aggregated for global feature importance analysis."
    ),
    "Grad-CAM": (
        "Gradient-weighted Class Activation Mapping (Grad-CAM) generates heatmaps highlighting image regions that most influenced a convolutional neural network's decision. "
        "These visual explanations help verify whether the network focuses on anatomically relevant regions and support qualitative interpretation."
    )
}

# Fonction pour échapper les caractères RTF et accents
def _escape_rtf(text):
    result = ""
    for c in text:
        if c in "\\{}":
            result += "\\" + c
        elif ord(c) > 127:  # non-ascii
            result += r"\u" + str(ord(c)) + "?"
        else:
            result += c
    return result

# Génération du RTF combiné FR+EN
def _make_combined_rtf_bytes(fr_map: dict, en_map: dict) -> bytes:
    header = r"{\rtf1\ansi\ansicpg1252\deff0"
    body = []
    body.append(r"\b Glossaire des acronymes — FR \b0\par\par")
    for k, v in fr_map.items():
        safe_v = _escape_rtf(v)
        body.append(r"\b " + k + r" \b0 : " + safe_v + r"\par\par")
    body.append(r"\par\par")
    body.append(r"\b Glossary of acronyms — EN \b0\par\par")
    for k, v in en_map.items():
        safe_v = _escape_rtf(v)
        body.append(r"\b " + k + r" \b0 : " + safe_v + r"\par\par")
    footer = r"}"
    rtf = header + "\n" + "\n".join(body) + "\n" + footer
    return rtf.encode("utf-8")

# --- Page
def run():
    st.markdown(_CSS, unsafe_allow_html=True)

    # Header (développé)
    try:
        colored_header(
            label="Analyse de Radiographies pulmonaires — Classification COVID-19",
            description=(
                "Prototype d'assistance diagnostique visuelle conçu pour démontrer la faisabilité d'un outil "
                "rapide et interprétable. Le pipeline intègre acquisition, prétraitement, modèles ML/DL et "
                "méthodes d'interprétabilité afin de rapprocher la décision algorithmique de l'intuition clinique. "
                "L'objectif de cette application est double : (1) présenter des résultats reproductibles et chiffrés, "
                "et (2) offrir une démo interactive qui illustre clairement comment le modèle prend ses décisions."
            ),
            color_name="blue-70"
        )
    except Exception:
        st.markdown("<h2>Analyse de Radiographies pulmonaires — Classification COVID-19</h2>", unsafe_allow_html=True)
        st.markdown("<div class='small-note'>Prototype d'assistance diagnostique visuelle — rapide, interprétable et prêt pour la démo.</div>", unsafe_allow_html=True)

    st.divider()

    # Hero avec tooltips
    left, right = st.columns([3, 1])
    with left:
        hero = (
            "<div class='project-hero'>"
            "<strong>Contexte</strong>: Utiliser "
            "<abbr title='Chest X-Ray'>CXR</abbr> + ML/DL pour accélérer la détection lorsque "
            "<abbr title='Reverse Transcription PCR'>RT-PCR</abbr> / <abbr title='Computed Tomography'>CT</abbr> sont indisponibles ou lentes. "
            "<br><br><strong>But</strong>: fournir une solution interprétable et reproductible pour la démonstration et le POC."
            "</div>"
        )
        st.markdown(hero, unsafe_allow_html=True)

    with right:
        st.markdown(
            "<div style='text-align:center'>"
            "<div style='font-size:18px;'>🧑‍🔬 Équipe</div>"
            "<div class='kv'>Cirine Bouamrane<br>Lena Bacot<br>Steven Moire<br>Rafael Cepa</div>"
            "<div style='margin-top:8px;font-size:12px;color:#98a1b3;'>Encadré par : Nicolas Mormiche</div>"
            f"<div style='margin-top:6px;font-size:12px;color:#98a1b3;'>{datetime.date.today().strftime('%d %B %Y')}</div>"
            "</div>", unsafe_allow_html=True
        )

    st.divider()

    # Glossaire visible (short defs)
    st.markdown("## Glossaire des acronymes")
    for k, short_def in GLOSSARY_MIN_FR.items():
        st.markdown(f"- **{k}** — {short_def}")

    # Bouton de téléchargement RTF combiné
    combined_rtf = _make_combined_rtf_bytes(GLOSSARY_LONG_FR, GLOSSARY_LONG_EN)
    st.download_button(
        label="Télécharger le glossaire détaillé (FR+EN — .rtf)",
        data=combined_rtf,
        file_name="glossaire_acronymes_FR_EN.rtf",
        mime="application/rtf"
    )

    st.divider()

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

    st.divider()

    # Livrables succincts
    st.markdown("## Livrables & artefacts (résumé)")
    _render_card(
        "Livrables attendus",
        "<ul><li>Rapport final</li><li>Notebooks & scripts reproductibles</li><li>Modèle(s) exporté(s) (pickle/ONNX)</li><li>Application Streamlit (POC)</li></ul>",
        st
    )

    st.divider()

    # Timeline courte
    st.markdown("## Timeline & jalons")
    st.markdown(
        "- Phase 0 — Kickoff & snapshot données (S0)\n"
        "- Phase 1 — Exploration & dataViz (S1–S4)\n"
        "- Phase 2 — Modélisation & tests (S5–S10)\n"
        "- Phase 3 — Validation & CI (S11–S12)\n"
        "- Phase 4 — Démo / Soutenance (S13+)"
    )

    st.divider()
    st.markdown("<small class='small-note'>Détails sur données et plan d'atténuation : onglet **02 - Données**.</small>", unsafe_allow_html=True)
