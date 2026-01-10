# 02_donnees.py — version améliorée : UI harmonisée, preview + ZIP, robust

from pathlib import Path
from typing import Optional
import streamlit as st
from PIL import Image

# KaggleHub pour datasets publics
try:
    import kagglehub
except Exception:
    kagglehub = None

# Streamlit-extras
try:
    from streamlit_extras.colored_header import colored_header
except Exception:
    colored_header = None


# ---------------- CONFIG ----------------
DATASET_SLUG = "tawsifurrahman/covid19-radiography-database"
DATASET_DIR = Path("dataset")
N_PER_CLASS_DEFAULT = 6
THUMBNAIL_MAX = (512, 512)

DEFAULT_CLASS_COUNTS = {"COVID": 3616, "Normal": 10192, "Viral Pneumonia": 1345, "Lung Opacity": 6012}
DEFAULT_TOTAL = sum(DEFAULT_CLASS_COUNTS.values())
CLASS_NAMES = list(DEFAULT_CLASS_COUNTS.keys())

# ---------------- CSS ----------------
_CSS = """
<style>
.section-card { 
    background: linear-gradient(90deg, rgba(12,18,30,0.95), rgba(8,12,20,0.95)); 
    padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); color:#cfe8ff; margin-bottom:12px; 
}
.card { 
    background:#131416; padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.04); 
    width:100%; max-width:260px; box-shadow:0 6px 14px rgba(0,0,0,0.35); margin-bottom:8px; 
}
.label { font-weight:700; color:#cfe8ff; margin-bottom:6px; }
.kv { font-size:12px; color:#98a1b3; }
.small-note { font-size:12px; color:#98a1b3; }
</style>
"""

def _render_section(title: str, body: str):
    st.markdown(f"<div class='section-card'><div class='label'>{title}</div><div>{body}</div></div>", unsafe_allow_html=True)

# ---------------- Helpers ----------------
@st.cache_resource
def get_kaggle_dataset_path(dataset_slug: str) -> Optional[Path]:
    if kagglehub is None:
        return None
    try:
        p = kagglehub.dataset_download(dataset_slug)
        return Path(p)
    except Exception:
        return None


def looks_like_images(p: Path):
    if not p.exists() or not p.is_dir():
        return False
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in img_exts:
            return True
    if (p / "images").exists():
        for f in (p / "images").iterdir():
            if f.is_file() and f.suffix.lower() in img_exts:
                return True
    return False

def find_dataset_root(base: Path) -> Path:
    """Retourne la vraie racine du dataset COVID-19 Radiography.
    
    Structure Kaggle connue :
    <kaggle_download>/COVID-19_Radiography_Dataset/
        ├── COVID/images/
        ├── Lung_Opacity/images/
        ├── Normal/images/
        └── Viral Pneumonia/images/
    """
    # Structure standard Kaggle (double niveau)
    expected = base / "COVID-19_Radiography_Dataset" / "COVID-19_Radiography_Dataset"
    if expected.exists() and (expected / "COVID").exists():
        return expected
    
    # Un seul niveau (cas local ou structure différente)
    nested = base / "COVID-19_Radiography_Dataset"
    if nested.exists() and (nested / "COVID").exists():
        return nested
    
    # Déjà au bon niveau
    if (base / "COVID").exists() or (base / "Normal").exists():
        return base
    
    # Fallback : retourner base
    return base



# ---------------- UI ----------------
def run():
    st.markdown(_CSS, unsafe_allow_html=True)
    
    # Header narratif développé
    header_text = (
        "Le dataset COVID-19 Radiography Database rassemble plusieurs milliers d'images de radiographies thoraciques (CXR), "
        "classées par type de pathologie : COVID-19, Normal, Viral Pneumonia et Lung Opacity, et leurs masques de segmentation. Ces images ont été collectées "
        "à partir de sources publiques et de publications de recherche, et représentent trois pathologies pulmonaires différentes, " \
        "mais avec des caractéristiques visuelles parfois similaires, surtout sur une radiographie en noir et blanc. "
        "Ces images permettent d'illustrer les capacités d'analyse et de modélisation dans le cadre d'un POC, mais ne seraient pas " \
        "vraiment utiles sans masques de segmentation. On peut imaginer, dans ce contexte, deux types de masques : des masques de lésions "
        "appelés 'lesions masks', ils sont très difficiles à obtenir par un radiologiste, et cela prendrait autant de temps " \
        "presque que le diagnostic lui même à partir de l'image et des masques de poumons ('lung masks'), qui sont simples à obtenir, informatiquement. "
        "Dans notre dataset, chaque image est accompagnée d'un masque de segmentation des poumons, ce qui sera utile pour éviter l'overfitting par exemple, "
        "et tout simplement pour être sûr que notre modèle se concentre sur la zone d'intérêt. En effet, on pourrait imaginer que certaines images contiennent des artefacts, " \
        "des annotations textuelles, ou d'autres éléments non pertinents qui pourraient biaiser l'apprentissage.  "
        "L'ensemble offre une bonne variabilité et représente un volume suffisant pour visualiser la distribution des classes, "
        "tester le pipeline de preprocessing et générer des échantillons reproductibles. "
        "Cette section fournit un aperçu rapide des classes, de la volumétrie et des échantillons disponibles pour exploration."
    )
    if colored_header:
        try: 
            colored_header("📦 Présentation des données", header_text, color_name="blue-70")
        except: 
            st.markdown(f"### 📦 Présentation des données\n{header_text}")
    else:
        st.markdown(f"### 📦 Présentation des données\n{header_text}")
    st.divider()

    # 2. Inventaire & volumétrie
    _render_section(
        "2. Inventaire & volumétrie",
        f"Dataset : {DATASET_SLUG}\nTotal images/masques référencées : {DEFAULT_TOTAL}\n"
        "Les images sont réparties selon les classes suivantes, permettant une visualisation claire de la disponibilité des données par catégorie :"
    )

    table_md = "| Classe | Images | Masks |\n|---:|---:|---:|\n"
    for k,v in DEFAULT_CLASS_COUNTS.items(): table_md += f"| {k} | {v} | {v} |\n"
    st.markdown(table_md)

    st.markdown(
    "**Note sur le déséquilibre de classes** : La distribution montre un déséquilibre notable "
    "(Normal : 10,192 vs Viral Pneumonia : 1,345). Pour atténuer l'impact sur l'entraînement, "
    "plusieurs stratégies ont été envisagées : "
    "**sous/sur-échantillonnage** (SMOTE, augmentation de données, undersampling, oversampling...), "
    "**pondération de la loss** (pénaliser davantage les erreurs sur classes minoritaires), "
    "**sampling stratifié** (échantillonnage équilibré lors du train/val split), et "
    "**class weighting** (ajustement des poids dans le modèle). "
    "Ces techniques seront comparées dans la section modélisation pour déterminer la stratégie optimale."
)

    # 3. Caractéristiques graphiques

    _render_section(
        "3. Caractéristiques graphiques des images et masques",
        "- Format : PNG (Portable Network Graphics) \n"
        "- Résolution : 299x299 pixels \n"
        "- Couleurs : L (1 canal, niveaux de gris) ou fake RGB (3 canaux identiques) \n"
        "- Masques : Binaires, alignés avec les images correspondantes \n"
        "- Variabilité : Diversité dans les angles, contrastes et éléments présents \n"
        "Ces caractéristiques influencent les étapes de pré-traitement et de modélisation."
    )
    # 4. Import & aperçu rapide
    st.markdown("## 4. Import & aperçu rapide (Kaggle)")
    if kagglehub is None:
        st.warning("KaggleHub non disponible — téléchargement automatique impossible.")
        return
    try:
        dataset_root = get_kaggle_dataset_path(DATASET_SLUG)
        if not dataset_root:
            st.error("Dataset Kaggle introuvable ou téléchargement échoué.")
            return
    except Exception as e:
        st.error(f"Erreur téléchargement Kaggle : {e}")
        return
    detected_root = find_dataset_root(dataset_root)
    st.write(f"Racine détectée : `{detected_root}`")
    

    # Classes
    classes = sorted([p.name for p in detected_root.iterdir() if looks_like_images(p)])
    if not classes: st.error("Aucune classe détectée."); return
    st.write(f"Classes détectées : {classes}")
    st.session_state["detected_root"] = str(detected_root)
    st.session_state["classes"] = classes


    