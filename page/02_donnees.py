# 02_donnees.py — version améliorée : UI harmonisée, preview + ZIP, robust

from pathlib import Path
from typing import Optional
import streamlit as st
from streamlit_extras.colored_header import colored_header
from PIL import Image
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import random
import plotly.express as px

# KaggleHub pour datasets publics
import kagglehub

# ---------------- CONFIG ----------------
DATASET_DIR = Path("dataset")
N_PER_CLASS_DEFAULT = 6
THUMBNAIL_MAX = (512, 512)

DEFAULT_CLASS_COUNTS = {"COVID": 3616, "Normal": 10192, "Viral Pneumonia": 1345, "Lung Opacity": 6012}
DEFAULT_TOTAL = sum(DEFAULT_CLASS_COUNTS.values())
CLASS_NAMES = list(DEFAULT_CLASS_COUNTS.keys())

IMG_EXTS = {".png"}
KAGGLE_SLUG = "tawsifurrahman/covid19-radiography-database"
THUMBNAIL_MAX = (512, 512)
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


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


# ============================================================================
# UTILITAIRES - CHARGEMENT DATASET
# ============================================================================

def _is_image_file(p: Path) -> bool:
    """Vérifie si un fichier est une image."""
    return p.is_file() and p.suffix.lower() in IMG_EXTS




# ============================================================================
# UTILITAIRES - MÉTRIQUES & ÉCHANTILLONNAGE
# ============================================================================

def compute_image_metrics(img: Image.Image) -> Dict:
    """Calcule luminosité, contraste, entropie."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    L_channel = 0.299 * r + 0.587 * g + 0.114 * b
    
    mean_lum = float(np.mean(L_channel))
    std_lum = float(np.std(L_channel))
    
    hist, _ = np.histogram(L_channel.flatten(), bins=256, range=(0, 255))
    probs = hist / (hist.sum() + 1e-12)
    probs = probs[probs > 0]
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size > 0 else 0.0
    
    return {
        "luminosity_mean": mean_lum,
        "contrast_std": std_lum,
        "entropy": entropy
    }


def mask_coverage(mask_path: Path) -> Optional[float]:
    """Retourne pourcentage de pixels masqués (0..100)."""
    if not mask_path.exists():
        return None
    
    try:
        m = Image.open(mask_path).convert("L")
        arr = np.array(m)
        covered = np.count_nonzero(arr)
        total = arr.size
        return 100.0 * covered / total if total > 0 else 0.0
    except Exception:
        return None


def sample_images_from_class(root: Path, cls: str, n: int) -> List[Path]:
    """Récupère n images aléatoires depuis root/<cls>/images/."""
    images_dir = root / cls / "images"
    
    if not images_dir.exists():
        return []
    
    imgs = sorted([p for p in images_dir.iterdir() if _is_image_file(p)])
    
    if len(imgs) <= n:
        return imgs
    
    rng = random.Random()
    return rng.sample(imgs, k=n)


def overlay_mask_on_image(img_path: Path, mask_path: Path, alpha: float = 0.4) -> Image.Image:
    """Superpose un masque rouge sur une image."""
    img = Image.open(img_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(img.size)
    
    mask_arr = np.array(mask)
    alpha_layer = (mask_arr > 0).astype(np.uint8) * int(255 * alpha)
    alpha_img = Image.fromarray(alpha_layer, mode="L")
    
    red = Image.new("RGBA", img.size, (255, 0, 0, 0))
    red.putalpha(alpha_img)
    
    return Image.alpha_composite(img, red)


# ============================================================================
# SCAN COMPLET DU DATASET
# ============================================================================

@st.cache_data(show_spinner=False)
def run_full_dataset_scan(root: Path, classes: List[str], include_masks: bool = True) -> Dict:
    """Scanne tout le dataset et retourne les métriques agrégées.
    
    Structure retournée :
    {
        "per_image": [{"path": str, "class": str, "metrics": dict, "mask": str, "mask_coverage": float}, ...],
        "by_class": {
            "COVID": {"count": int, "avg_lum": float, "avg_std": float, "avg_entropy": float, "mask_coverages": [...]},
            ...
        }
    }
    """
    results = {"per_image": [], "by_class": {}}
    
    for cls in classes:
        results["by_class"][cls] = {
            "count": 0,
            "metrics": [],
            "mask_coverages": [],
            "files": []
        }
        
        images_dir = root / cls / "images"
        
        if not images_dir.exists():
            continue
        
        files = sorted([p for p in images_dir.iterdir() if _is_image_file(p)])
        results["by_class"][cls]["count"] = len(files)
        
        for img_path in files:
            try:
                # Calculer métriques image
                img = Image.open(img_path).convert("RGB")
                metrics = compute_image_metrics(img)
                
                # Chercher le masque correspondant
                mask_path = None
                mask_cov = None
                
                if include_masks:
                    mask_candidate = root / cls / "masks" / img_path.name
                    if mask_candidate.exists():
                        mask_path = mask_candidate
                        mask_cov = mask_coverage(mask_path)
                
                # Enregistrer
                entry = {
                    "path": str(img_path),
                    "class": cls,
                    "metrics": metrics,
                    "mask": str(mask_path) if mask_path else None,
                    "mask_coverage": mask_cov
                }
                
                results["per_image"].append(entry)
                results["by_class"][cls]["metrics"].append(metrics)
                
                if mask_cov is not None:
                    results["by_class"][cls]["mask_coverages"].append(mask_cov)
                
                results["by_class"][cls]["files"].append(str(img_path))
            
            except Exception:
                continue
    
    # Calculer statistiques agrégées par classe
    for cls, info in results["by_class"].items():
        ms = info["metrics"]
        
        if ms:
            info["avg_lum"] = float(np.mean([m["luminosity_mean"] for m in ms]))
            info["avg_std"] = float(np.mean([m["contrast_std"] for m in ms]))
            info["avg_entropy"] = float(np.mean([m["entropy"] for m in ms]))
        else:
            info["avg_lum"] = 0.0
            info["avg_std"] = 0.0
            info["avg_entropy"] = 0.0
    
    return results


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_luminosity_distributions(df_metrics: pd.DataFrame):
    """Affiche les distributions de luminosité et contraste par classe."""
    
    st.markdown("### 📊 Distribution de Luminosité par Classe")
    
    fig_lum = px.violin(
        df_metrics, x="class", y="lum",
        box=True, points="all",
        labels={"lum": "Luminosité moyenne", "class": "Classe"},
        title="Distribution de luminosité par classe",
        color="class"
    )
    st.plotly_chart(fig_lum, use_container_width=True)
    
    st.markdown("### 📊 Distribution de Contraste par Classe")
    
    fig_std = px.violin(
        df_metrics, x="class", y="std",
        box=True, points="all",
        labels={"std": "Contraste (std)", "class": "Classe"},
        title="Distribution du contraste par classe",
        color="class"
    )
    st.plotly_chart(fig_std, use_container_width=True)


def plot_mask_coverage(by_class: Dict, classes: List[str]):
    """Affiche la distribution de couverture des masques."""
    
    st.markdown("### 🎭 Distribution de Couverture des Masques")
    
    # Construire données long format
    long_data = []
    for cls in classes:
        for cov in by_class[cls].get("mask_coverages", []):
            long_data.append({"class": cls, "mask_cov": cov})
    
    if not long_data:
        st.info("Aucune donnée de masque disponible.")
        return
    
    df_masks = pd.DataFrame(long_data)
    
    fig = px.box(
        df_masks, x="class", y="mask_cov",
        labels={"mask_cov": "Couverture (%)", "class": "Classe"},
        title="Distribution de la couverture des masques par classe",
        color="class"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_mask_overlays(per_image: List[Dict], max_examples: int = 3):
    """Affiche des exemples de masques superposés."""
    
    st.markdown("### 🔍 Exemples de Masques Superposés")
    
    mask_examples = [e for e in per_image if e.get("mask")]
    
    if not mask_examples:
        st.info("Aucun masque détecté dans le dataset.")
        return
    
    examples = mask_examples[:max_examples]
    cols = st.columns(len(examples))
    
    for i, entry in enumerate(examples):
        with cols[i]:
            try:
                img_path = Path(entry["path"])
                mask_path = Path(entry["mask"])
                
                overlay = overlay_mask_on_image(img_path, mask_path, alpha=0.4)
                overlay.thumbnail((320, 320))
                
                st.image(overlay, caption=f"{img_path.name}")
                
                cov = entry.get("mask_coverage")
                if cov is not None:
                    st.metric("Coverage", f"{cov:.1f}%")
            
            except Exception as e:
                st.error(f"Erreur: {e}")



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
        f"Dataset : {KAGGLE_SLUG}\nTotal images/masques référencées : {DEFAULT_TOTAL}\n"
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
        dataset_root = get_kaggle_dataset_path(KAGGLE_SLUG)
        if not dataset_root:
            st.error("Dataset Kaggle introuvable ou téléchargement échoué.")
            return
        else:
            dataset_root = dataset_root / "COVID-19_Radiography_Dataset" # dataset_root = ****.cache/kagglehub/datasets/tawsifurrahman/covid19-radiography-database/versions/5/COVID-19_Radiography_Dataset
    except Exception as e:
        st.error(f"Erreur téléchargement Kaggle : {e}")
        return
    st.write(f"Racine détectée : `{dataset_root}`")
    

    # Classes
    classes = sorted([p.name for p in dataset_root.iterdir() if looks_like_images(p)])
    if not classes: st.error("Aucune classe détectée."); return
    st.write(f"Classes détectées : {classes}")
    st.session_state["dataset_root"] = str(dataset_root)
    st.session_state["classes"] = classes

    st.set_page_config(layout="wide")
    
    # Styles
    st.markdown(
        "<style>.small-note{font-size:12px;color:#98a1b3}</style>",
        unsafe_allow_html=True
    )
    
    # En-tête
    colored_header(
        "Analyse & Visualisations du Dataset",
        "Exploration visuelle et métriques des images radiographiques COVID-19",
        color_name="violet-70"
    )
    st.divider()
    
    # ========================================================================
    # 1. CHARGEMENT DU DATASET
    # ========================================================================
    
    
    # ========================================================================
    # 2. ÉCHANTILLONNAGE RAPIDE
    # ========================================================================
    
    colored_header(
        "1. Échantillonnage Rapide",
        "Visualisation d'un petit échantillon d'images",
        color_name="violet-70"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        choice = st.selectbox("Choisir une classe :", options=classes)
    
    with col2:
        n = st.number_input("Nombre d'images :", 1, 5, 5)
    
    with col3:
        if st.button("🔄 Nouvel échantillon"):
            st.session_state.pop("viz_sample", None)
    
    # Gestion de l'échantillonnage
    sample_key = f"{choice}__{n}"
    
    if "viz_sample" not in st.session_state or st.session_state["viz_sample"].get("key") != sample_key:
        imgs = sample_images_from_class(dataset_root, choice, n)
        st.session_state["viz_sample"] = {
            "key": sample_key,
            "images": [str(p) for p in imgs]
        }
    
    img_paths = [Path(p) for p in st.session_state["viz_sample"]["images"]]
    
    # Affichage
    if not img_paths:
        st.info("Aucune image disponible pour cette classe.")
    else:
        # Sélection de l'image à afficher
        img_names = [p.name for p in img_paths]
        selected_name = st.selectbox(
            "Choisir une image :",
            options=img_names,
            key="selected_sample_image"
        )
        
        # Trouver l'image correspondante
        selected_idx = img_names.index(selected_name)
        img_path = img_paths[selected_idx]
        
        # Chercher le masque correspondant
        mask_path = dataset_root / choice / "masks" / img_path.name
        
        # Affichage de l'image sélectionnée
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(THUMBNAIL_MAX)
            
            # Afficher image et masque en colonnes
            col_img, col_mask = st.columns(2)
            
            with col_img:
                st.markdown("**📷 Image**")
                st.image(img, caption=img_path.name)

                # Métriques de l'image
                st.markdown("---")
                metrics = compute_image_metrics(img)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Luminosité", f"{metrics['luminosity_mean']:.1f}")
                with col_b:
                    st.metric("Contraste", f"{metrics['contrast_std']:.1f}")
                with col_c:
                    st.metric("Entropie", f"{metrics['entropy']:.2f}")
            
            with col_mask:
                st.markdown("**🎭 Masque**")
                if mask_path.exists():
                    try:
                        mask = Image.open(mask_path).convert("L")
                        mask.thumbnail(THUMBNAIL_MAX)
                        st.image(mask, caption=mask_path.name)
                        
                        # Calculer couverture du masque
                        mask_cov = mask_coverage(mask_path)
                        if mask_cov is not None:
                            st.metric("Couverture", f"{mask_cov:.1f}%")
                    except Exception as e:
                        st.warning(f"Erreur chargement masque : {e}")
                else:
                    st.info("Aucun masque disponible pour cette image.")
            

        
        except Exception as e:
            st.error(f"Erreur : {e}")
    
    st.divider()
    
    # ========================================================================
    # 3. ANALYSE COMPLÈTE DU DATASET
    # ========================================================================
    
    colored_header(
        "2. Analyse Complète du Dataset",
        "Scan et visualisations sur l'ensemble des données",
        color_name="violet-70"
    )
    
    if st.button("🚀 Lancer le scan complet", type="primary"):
        st.session_state.pop("last_scan", None)
    
    # Scan avec cache
    if "last_scan" not in st.session_state:
        with st.spinner("⏳ Scan en cours... Cela peut prendre quelques instants."):
            scan_data = run_full_dataset_scan(dataset_root, classes, include_masks=True)
            st.session_state["last_scan"] = scan_data
    else:
        scan_data = st.session_state["last_scan"]
    
    if not scan_data or not scan_data.get("per_image"):
        st.warning("⚠️ Scan vide ou incomplet.")
        return
    
    # Stats globales
    total_images = len(scan_data["per_image"])
    st.metric("Images analysées", f"{total_images:,}")
    
    st.divider()
    
    # ========================================================================
    # 4. VISUALISATIONS
    # ========================================================================
    
    st.markdown("## 📊 Visualisations & Analyses")
    
    # Préparer DataFrame pour Plotly
    rows = []
    for entry in scan_data["per_image"]:
        m = entry["metrics"]
        rows.append({
            "class": entry["class"],
            "lum": m["luminosity_mean"],
            "std": m["contrast_std"],
            "entropy": m["entropy"]
        })
    
    df_metrics = pd.DataFrame(rows)
    
    # Luminosité & Contraste
    plot_luminosity_distributions(df_metrics)
    
    st.divider()
    
    # Masques - Exemples
    show_mask_overlays(scan_data["per_image"], max_examples=3)
    
    st.divider()
    
    # Masques - Distribution
    plot_mask_coverage(scan_data["by_class"], classes)
    
    st.divider()
    
    # ========================================================================
    # 5. RECOMMANDATIONS
    # ========================================================================
    
    colored_header(
        "3. Recommandations",
        "Actions suggérées pour l'amélioration du dataset",
        color_name="violet-70"
    )


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    run()


    