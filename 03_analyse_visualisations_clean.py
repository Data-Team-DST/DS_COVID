"""
Module d'analyse et visualisation du dataset COVID-19 Radiography.
Version simplifiée et restructurée.
"""

import streamlit as st
from streamlit_extras.colored_header import colored_header
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import numpy as np
import pandas as pd
import random
import plotly.express as px


# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KAGGLE_SLUG = "tawsifurrahman/covid19-radiography-database"
THUMBNAIL_MAX = (512, 512)
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


# ============================================================================
# UTILITAIRES - CHARGEMENT DATASET
# ============================================================================

def _is_image_file(p: Path) -> bool:
    """Vérifie si un fichier est une image."""
    return p.is_file() and p.suffix.lower() in IMG_EXTS


@st.cache_resource
def get_kaggle_dataset_path(dataset_slug: str) -> Optional[Path]:
    """Télécharge et retourne le chemin du dataset Kaggle."""
    try:
        import kagglehub
        p = kagglehub.dataset_download(dataset_slug)
        return Path(p) if p else None
    except Exception:
        return None


def find_dataset_root(base: Path) -> Path:
    """Retourne la vraie racine du dataset COVID-19 Radiography.
    
    Structure Kaggle connue :
    <kaggle_download>/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/
        ├── COVID/images/
        ├── Lung_Opacity/images/
        ├── Normal/images/
        └── Viral Pneumonia/images/
    """
    # Structure standard Kaggle (double niveau)
    expected = base / "COVID-19_Radiography_Dataset" / "COVID-19_Radiography_Dataset"
    if expected.exists() and (expected / "COVID").exists():
        return expected
    
    # Un seul niveau
    nested = base / "COVID-19_Radiography_Dataset"
    if nested.exists() and (nested / "COVID").exists():
        return nested
    
    # Déjà au bon niveau
    if (base / "COVID").exists() or (base / "Normal").exists():
        return base
    
    return base


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
    
    # Analyse automatique
    with st.expander("💡 Interprétation"):
        for cls in df_metrics["class"].unique():
            subset = df_metrics[df_metrics["class"] == cls]
            mean_lum = subset["lum"].mean()
            std_lum = subset["std"].mean()
            
            st.markdown(f"**{cls}** : luminosité moyenne {mean_lum:.1f}, contraste {std_lum:.1f}")
        
        overall_lum = df_metrics["lum"].mean()
        
        if overall_lum < 60:
            st.warning(
                "⚠️ Images globalement sombres. "
                "Envisager normalisation d'histogramme ou CLAHE."
            )
        elif overall_lum > 220:
            st.warning("⚠️ Images très claires → risque de clipping.")
        else:
            st.success("✅ Luminosité globale dans une plage raisonnable.")
    
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
    
    # Analyse automatique
    with st.expander("💡 Interprétation"):
        for cls in classes:
            covs = by_class[cls].get("mask_coverages", [])
            
            if covs:
                mean_cov = np.mean(covs)
                st.markdown(f"**{cls}** : couverture moyenne {mean_cov:.1f}% (n={len(covs)})")
                
                if mean_cov < 1.0:
                    st.warning(f"⚠️ Couverture très faible — masques possiblement vides.")
                elif mean_cov > 40.0:
                    st.success(f"✅ Bonne couverture — utile pour segmentation.")
            else:
                st.markdown(f"**{cls}** : aucun masque détecté")


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


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def run():
    """Point d'entrée principal."""
    
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
    
    detected_root = None
    
    # Vérifier session state (page 02_donnees)
    if "detected_root" in st.session_state:
        try:
            detected_root = Path(st.session_state["detected_root"])
            if detected_root.exists():
                st.caption(f"✅ Dataset chargé depuis session : `{detected_root}`")
        except Exception:
            detected_root = None
    
    # Sinon télécharger via Kaggle
    if detected_root is None:
        with st.spinner("📥 Téléchargement du dataset via Kaggle..."):
            kaggle_path = get_kaggle_dataset_path(KAGGLE_SLUG)
            
            if kaggle_path:
                detected_root = find_dataset_root(kaggle_path)
                st.success(f"✅ Dataset téléchargé : `{detected_root}`")
            else:
                st.error(
                    "❌ Impossible de charger le dataset.\n\n"
                    "**Actions requises :**\n"
                    "1. Exécuter la page **02_donnees** pour initialiser le dataset\n"
                    "2. Configurer vos credentials Kaggle dans `~/.kaggle/kaggle.json`"
                )
                return
    
    classes = CLASS_NAMES
    
    # Info compacte
    with st.expander("ℹ️ Configuration du Dataset"):
        st.markdown(f"**Racine** : `{detected_root}`")
        st.markdown(f"**Classes** : {', '.join(classes)}")
    
    st.divider()
    
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
        imgs = sample_images_from_class(detected_root, choice, n)
        st.session_state["viz_sample"] = {
            "key": sample_key,
            "images": [str(p) for p in imgs]
        }
    
    img_paths = [Path(p) for p in st.session_state["viz_sample"]["images"]]
    
    # Affichage
    if not img_paths:
        st.info("Aucune image disponible pour cette classe.")
    else:
        cols = st.columns(min(3, len(img_paths)))
        
        for idx, img_path in enumerate(img_paths):
            with cols[idx % len(cols)]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.thumbnail(THUMBNAIL_MAX)
                    
                    st.image(img, use_column_width=True, caption=img_path.name)
                    
                    metrics = compute_image_metrics(img)
                    
                    st.markdown(
                        f"**Luminosité** : {metrics['luminosity_mean']:.1f}  \n"
                        f"**Contraste** : {metrics['contrast_std']:.1f}  \n"
                        f"**Entropie** : {metrics['entropy']:.2f}"
                    )
                
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
            scan_data = run_full_dataset_scan(detected_root, classes, include_masks=True)
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
    
    st.markdown("""
    #### 📋 Points clés :
    
    1. **Qualité des images**
       - Vérifier les images avec luminosité/contraste extrêmes
       - Identifier les artefacts visuels (texte, logos, bandes noires)
    
    2. **Masques de segmentation**
       - Valider la cohérence des masques (couverture > 0%)
       - Vérifier l'alignement image/masque
    
    3. **Équilibrage des classes**
       - Surveiller le déséquilibre entre classes
       - Envisager augmentation de données si nécessaire
    
    4. **Documentation**
       - Tracer toutes les transformations appliquées
       - Documenter les choix de preprocessing
    """)
    
    st.markdown(
        "<div class='small-note'>"
        "💡 Conseil : Exécutez régulièrement cette analyse après modification du dataset."
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    run()
