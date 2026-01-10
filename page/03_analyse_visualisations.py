# 03_analyse_visualisations.py — version robuste & fonctionnelle : échantillonnage, métriques image, export RTF+ZIP
# Ajoutées : 10 visualisations (MVP -> avancées) utilisant plotly quand possible.
import streamlit as st
from streamlit_extras.colored_header import colored_header
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import numpy as np
import random
import plotly.express as px



# ---------------- CONFIG ----------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KAGGLE_SLUG = "tawsifurrahman/covid19-radiography-database"
THUMBNAIL_MAX = (512, 512)
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# ---------------- Helpers ----------------
def _is_image_file(p: Path) -> bool:
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

def sample_images_from_class(root: Path, cls: str, n: int) -> List[Path]:
    """Récupère n images depuis root/<cls>/images/ (structure connue)."""
    # Structure connue : root/<CLASS>/images/
    images_dir = root / cls / "images"
    
    if not images_dir.exists():
        return []
    
    imgs = sorted([p for p in images_dir.iterdir() if _is_image_file(p)])
    
    if len(imgs) <= n:
        return imgs
    
    rng = random.Random()
    return rng.sample(imgs, k=n)

def get_lum

def compute_image_metrics(img: Image.Image) -> Dict:
    """Calcule luminosité (L), contraste (std), entropie approxi., fake-RGB."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r,g,b = arr[...,0], arr[...,1], arr[...,2]
    L_channel = 0.299*r + 0.587*g + 0.114*b
    fake_rgb = bool(np.allclose(r, g) and np.allclose(g, b))
    mean_lum = float(np.mean(L_channel))
    std_lum = float(np.std(L_channel))
    hist, _ = np.histogram(L_channel.flatten(), bins=256, range=(0,255))
    probs = hist / (hist.sum() + 1e-12)
    probs = probs[probs>0]
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size>0 else 0.0
    return {"luminosity_mean": mean_lum, "contrast_std": std_lum, "entropy": entropy, "fake_rgb": fake_rgb}

def mask_coverage(mask_path: Path) -> Optional[float]:
    """Retourne pourcentage de pixels masqués (0..100) ou None si indisponible."""
    if not mask_path.exists():
        return None
    try:
        m = Image.open(mask_path).convert("L")
        arr = np.array(m)
        covered = np.count_nonzero(arr)
        total = arr.size
        return 100.0 * covered / total
    except Exception:
        return None




# Helper: render analyses summary (name + interpretation) with real metrics
def render_analyses_summary(analyses, scan=None, by_class=None, classes=None, viz_id=None):
    """
    Affiche un résumé des analyses automatiques.
    Si viz_id est fourni: affiche seulement la visualisation spécifiée.
    Sinon: affiche toutes les visualisations + recommandations.
    """
    # Extract data if scan provided, else use provided by_class/classes
    if scan and not by_class:
        by_class = scan.get("by_class", {})
        classes = list(by_class.keys())
    
    per_image = scan.get("per_image", []) if scan else []
    per_image_count = len(per_image)
    
    # Si viz_id spécifié, afficher seulement celui-là
    if viz_id:
        if viz_id in analyses:
            metrics = _build_metrics_for_viz(viz_id, scan, per_image, by_class or {}, classes or [], per_image_count)

    else:
        # Sinon afficher tous les viz + recommandations
        for k in sorted([kk for kk in analyses.keys() if kk.startswith("viz")]):
            # Build metrics dict from scan data for this viz
            metrics = _build_metrics_for_viz(k, scan, per_image, by_class or {}, classes or [], per_image_count)
        rec = analyses.get("recommendation")
        if rec:
            st.markdown("**Recommandations globales**")
            st.markdown(rec)

def _build_metrics_for_viz(viz_id, scan, per_image, by_class, classes, per_image_count):
    """Construit le dict metrics pour une visualisation donnée à partir du scan."""
    metrics = {}

    if viz_id == "viz1":
        # Placeholder for viz1 metrics if needed
        pass
    elif viz_id == "viz2":
        # Luminosité par classe
        if by_class and classes:
            for cls in classes:
                cls_data = by_class.get(cls, {})
                metrics[f'{cls}_lum'] = f"{cls_data.get('avg_lum', 0.0):.1f}"
                metrics[f'{cls}_std'] = f"{cls_data.get('avg_std', 0.0):.1f}"
                metrics[f'{cls}_entropy'] = f"{cls_data.get('avg_entropy', 0.0):.2f}"
            # Map short names
            metrics['covid_lum'] = metrics.get('COVID_lum', metrics.get('covid_lum', 'N/A'))
            metrics['covid_std'] = metrics.get('COVID_std', metrics.get('covid_std', 'N/A'))
            metrics['covid_entropy'] = metrics.get('COVID_entropy', metrics.get('covid_entropy', 'N/A'))
            metrics['lo_lum'] = metrics.get('Lung_Opacity_lum', metrics.get('lo_lum', 'N/A'))
            metrics['lo_std'] = metrics.get('Lung_Opacity_std', metrics.get('lo_std', 'N/A'))
            metrics['lo_entropy'] = metrics.get('Lung_Opacity_entropy', metrics.get('lo_entropy', 'N/A'))
            metrics['norm_lum'] = metrics.get('Normal_lum', metrics.get('norm_lum', 'N/A'))
            metrics['norm_std'] = metrics.get('Normal_std', metrics.get('norm_std', 'N/A'))
            metrics['norm_entropy'] = metrics.get('Normal_entropy', metrics.get('norm_entropy', 'N/A'))
            metrics['vp_lum'] = metrics.get('Viral Pneumonia_lum', metrics.get('vp_lum', 'N/A'))
            metrics['vp_std'] = metrics.get('Viral Pneumonia_std', metrics.get('vp_std', 'N/A'))
            metrics['vp_entropy'] = metrics.get('Viral Pneumonia_entropy', metrics.get('vp_entropy', 'N/A'))
    
    
    elif viz_id in ["viz5", "viz6"]:
        # Mask coverage
        if by_class and classes:
            metrics['covid_mask'] = f"{np.mean(by_class.get('COVID', {}).get('mask_coverages', [0])):.1f}" if by_class.get('COVID', {}).get('mask_coverages') else 'N/A'
            metrics['lo_mask'] = f"{np.mean(by_class.get('Lung_Opacity', {}).get('mask_coverages', [0])):.1f}" if by_class.get('Lung_Opacity', {}).get('mask_coverages') else 'N/A'
            metrics['norm_mask'] = f"{np.mean(by_class.get('Normal', {}).get('mask_coverages', [0])):.1f}" if by_class.get('Normal', {}).get('mask_coverages') else 'N/A'
            metrics['vp_mask'] = f"{np.mean(by_class.get('Viral Pneumonia', {}).get('mask_coverages', [0])):.1f}" if by_class.get('Viral Pneumonia', {}).get('mask_coverages') else 'N/A'
    
    
    return metrics


# ---------------- Utility viz helpers ----------------
def overlay_mask_on_image(img_path: Path, mask_path: Path, alpha: float = 0.45) -> Image.Image:
    """Superpose mask (en rouge) sur image en RGBA."""
    img = Image.open(img_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(img.size)
    # create colored mask
    mask_rgba = Image.new("RGBA", img.size, (255, 0, 0, 0))
    mask_arr = np.array(mask)
    alpha_layer = (mask_arr > 0).astype(np.uint8) * int(255 * alpha)
    alpha_img = Image.fromarray(alpha_layer, mode="L")
    red = Image.new("RGBA", img.size, (255, 0, 0, 0))
    red.putalpha(alpha_img)
    return Image.alpha_composite(img, red)

# ---------------- Full dataset scan (cached) ----------------
@st.cache_data(show_spinner=False)
def run_full_dataset_scan(root: Path, classes: List[str], include_masks: bool = True) -> Dict:
    """
    Balayage complet du dataset — retourne métriques agrégées par image et par classe.
    Résultat:
    {
      "per_image": [{"path": str(p), "class": cls, "metrics": {...}, "mask": mask_or_None, "mask_coverage": float_or_None}, ...],
      "by_class": {cls: {"count": int, "metrics": [...], "fake_pct": float, ...}, ...}
    }
    """
    results = {"per_image": [], "by_class": {}}
    for cls in classes:
        results["by_class"].setdefault(cls, {"count": 0, "metrics": [], "mask_coverages": [], "files": []})
        # try standard locations
        cls_dir = root / cls
        images_sub = cls_dir / "images"
        if images_sub.exists() and images_sub.is_dir():
            search_dir = images_sub
        else:
            # maybe root/images/<cls>
            maybe = root / "images" / cls
            if maybe.exists() and maybe.is_dir():
                search_dir = maybe
            else:
                search_dir = cls_dir
        files = sorted([p for p in search_dir.rglob("*") if _is_image_file(p)])
        results["by_class"][cls]["count"] = len(files)
        for p in files:
            try:
                im = Image.open(p).convert("RGB")
                metrics = compute_image_metrics(im)
                mask_path = None
                mask_cov = None
                if include_masks:
                    # common mask locations: same folder parent/"masks"/filename or cls_dir/"masks"/filename
                    candidate1 = p.parent.parent / "masks" / p.name  # if structure .../images/<class>/<file> and masks at .../<class>/masks/
                    candidate2 = p.parent / "masks" / p.name
                    candidate3 = cls_dir / "masks" / p.name
                    for cand in (candidate1, candidate2, candidate3):
                        if cand.exists():
                            mask_path = cand
                            break
                    if mask_path:
                        mask_cov = mask_coverage(mask_path)
                entry = {"path": str(p), "class": cls, "metrics": metrics, "mask": str(mask_path) if mask_path else None, "mask_coverage": mask_cov}
                results["per_image"].append(entry)
                results["by_class"][cls]["metrics"].append(metrics)
                if mask_cov is not None:
                    results["by_class"][cls]["mask_coverages"].append(mask_cov)
                results["by_class"][cls]["files"].append(str(p))
            except Exception:
                # skip unreadable
                continue
    # compute per-class summaries
    for cls, info in results["by_class"].items():
        ms = info["metrics"]
        if ms:
            fake_pct = 100.0 * sum(1 for m in ms if m.get("fake_rgb")) / len(ms)
            avg_lum = float(np.mean([m["luminosity_mean"] for m in ms]))
            avg_std = float(np.mean([m["contrast_std"] for m in ms]))
            avg_entropy = float(np.mean([m["entropy"] for m in ms]))
        else:
            fake_pct = avg_lum = avg_std = avg_entropy = 0.0
        info["fake_pct"] = fake_pct
        info["avg_lum"] = avg_lum
        info["avg_std"] = avg_std
        info["avg_entropy"] = avg_entropy
    return results


# ---------------- UI ----------------
def run():
    st.set_page_config(layout="wide")
    st.markdown("<style>.small-note{font-size:12px;color:#98a1b3}</style>", unsafe_allow_html=True)

    colored_header("1. Contexte & aperçu", "Extrait un petit échantillon, métriques d'image, export RTF+ZIP — +10 visualisations (plotly).", color_name="violet-70")
    st.divider()

    # ---------------- Chargement du dataset ----------------
    detected_root = None
    
    # 1) Vérifier session state (page 02_donnees)
    if "detected_root" in st.session_state:
        try:
            detected_root = Path(st.session_state["detected_root"])
            if detected_root.exists():
                st.caption(f"✅ Dataset chargé depuis session : `{detected_root}`")
        except Exception:
            detected_root = None
    
    # 2) Sinon, télécharger via KaggleHub
    if detected_root is None:
        with st.spinner("📥 Téléchargement du dataset via Kaggle..."):
            kaggle_path = get_kaggle_dataset_path(KAGGLE_SLUG)
            if kaggle_path:
                detected_root = find_dataset_root(kaggle_path)
                st.success(f"✅ Dataset téléchargé : `{detected_root}`")
            else:
                st.error(
                    "❌ Impossible de charger le dataset.\n\n"
                    "Assurez-vous d'avoir :\n"
                    "1. Exécuté la page **02_donnees** pour initialiser le dataset\n"
                    "2. Configuré vos credentials Kaggle dans `~/.kaggle/kaggle.json`"
                )
                return
    
    # Utiliser les classes hardcodées
    classes = CLASS_NAMES
    
    # Afficher info compacte
    with st.expander("ℹ️ Configuration du Dataset"):
        st.markdown(f"**Racine** : `{detected_root}`")
        st.markdown(f"**Classes** : {', '.join(classes)}")

    # ---------------- Sample selection & quick view ----------------
    colored_header("2. Sélection échantillon", "Choisir une classe et tirer jusqu'à 5 images.", color_name="violet-70")
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        choice = st.selectbox("Classe :", options=classes)
    with col2:
        n = st.number_input("N images (max 5) :", 1,5,5)
    with col3:
        if st.button("Resample / nouvel échantillon"):
            st.session_state.pop("viz_sample", None)

    sample_key = f"{choice}__{n}"
    sample = st.session_state.get("viz_sample", None)
    if sample is None or sample.get("key") != sample_key:
        imgs = sample_images_from_class(detected_root, choice, n)
        st.session_state["viz_sample"] = {"key": sample_key, "images": [str(p) for p in imgs]}
        sample = st.session_state["viz_sample"]
    img_paths = [Path(p) for p in sample.get("images", [])]

    # display thumbnails + metrics quick
    colored_header("3. Aperçu & métriques (rapide)", "Miniatures + luminosité, contraste, entropie, fake-RGB.", color_name="violet-70")
    if not img_paths:
        st.info("Aucun fichier image pour cette classe.")
    else:
        cols = st.columns(min(3, len(img_paths)))
        per_image_metrics = []
        for idx,p in enumerate(img_paths):
            with cols[idx % len(cols)]:
                try:
                    im = Image.open(p).convert("RGB")
                    im.thumbnail(THUMBNAIL_MAX)
                    st.image(im, use_column_width=True, caption=p.name)
                    metrics = compute_image_metrics(im)
                    per_image_metrics.append({"filename": p.name, **metrics, "path": str(p)})
                    st.markdown(
                        f"- Luminosité : **{metrics['luminosity_mean']:.2f}**  \n"
                        f"- Contraste (std) : **{metrics['contrast_std']:.2f}**  \n"
                        f"- Entropie : **{metrics['entropy']:.3f}**  \n"
                        f"- Fake-RGB : {metrics['fake_rgb']}"
                    )
                except Exception as e:
                    st.error(f"Erreur ouvrant {p}: {e}")
    st.divider()

    # ---------------- Run full scan and visualisations ----------------
    colored_header("4. Visualisations & analyses (full dataset)", "Clique 'Run full dataset scan' pour générer les 10 visualisations (op lourde).", color_name="violet-70")
    run_scan = st.button("Run full dataset scan (génère les 10 visualisations)")
    if "last_scan" not in st.session_state:
        st.session_state["last_scan"] = None

    if run_scan or st.session_state["last_scan"] is None:
        with st.spinner("Scanning dataset (peut prendre du temps)..."):
            scan = run_full_dataset_scan(detected_root, classes, include_masks=True)
            st.session_state["last_scan"] = scan
    else:
        scan = st.session_state["last_scan"]
    
    # --- Générer analyses automatiques pour les visualisations (1 fois après le scan)
    try:
        analyses = generate_visual_analysis(scan, classes)
        # stocker pour réutilisation (évite recalcul si re-rendu)
        st.session_state["last_scan_analyses"] = analyses
    except Exception as e:
        analyses = {}
        st.warning(f"génération analyses automatique échouée: {e}")


    # If scan empty, warn
    if not scan or not scan.get("per_image"):
        st.warning("Scan incomplet ou aucun fichier analysé — vérifie les permissions / structure des dossiers.")
        return

    per_image = scan["per_image"]
    by_class = scan["by_class"]

    # --- Visualization 2 : Histogramme luminosité & contraste par classe (small multiples)
    st.markdown("### 2) Histogramme de luminosité & contraste par classe")
    # Build dataframe for plotly
    import pandas as pd
    rows = []
    for entry in per_image:
        m = entry["metrics"]
        rows.append({"class": entry["class"], "lum": m["luminosity_mean"], "std": m["contrast_std"], "entropy": m["entropy"]})
    df_metrics = pd.DataFrame(rows)
    # Luminosity violin
    fig2 = px.violin(df_metrics, x="class", y="lum", box=True, points="all", labels={"lum":"Luminosité moyenne"}, title="Distribution de luminosité par classe")
    st.plotly_chart(fig2, use_container_width=True)
    # --- Afficher analyses automatiques (résumé: nom + interprétation)
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    fig3 = px.violin(df_metrics, x="class", y="std", box=True, points="all", labels={"std":"Contraste (std)"}, title="Distribution du contraste (std) par classe")
    st.plotly_chart(fig3, use_container_width=True)
    # --- Afficher analyses automatiques (résumé: nom + interprétation)
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan if 'scan' in locals() else None, viz_id="viz2")
        else:
            st.markdown("Aucune analyse automatique disponible.")


    # --- Visualization 5 : Overlay mask preview + mask coverage stats
    st.markdown("### 5) Aperçu masques (overlay) + statistiques de coverage")
    # show up to 3 examples with masks if available
    mask_examples = [e for e in per_image if e.get("mask")]
    if mask_examples:
        ex = mask_examples[:3]
        cols_m = st.columns(len(ex))
        for i,e in enumerate(ex):
            with cols_m[i]:
                try:
                    ip = Path(e["path"])
                    mp = Path(e["mask"])
                    overlay = overlay_mask_on_image(ip, mp, alpha=0.4)
                    overlay.thumbnail((320,320))
                    st.image(overlay, caption=f"{ip.name} (mask overlay)")
                    cov = e.get("mask_coverage")
                    st.markdown(f"- Mask coverage: {cov:.2f}%") if cov is not None else st.markdown("- Mask coverage: N/A")
                except Exception as excep:
                    st.write("Erreur affichage mask:", excep)
    else:
        st.info("Aucun mask détecté pour les échantillons (vérifie structure masks/).")

    # mask coverage aggregated
    mask_cov_rows = []
    for cls in classes:
        covs = by_class[cls].get("mask_coverages", [])
        if covs:
            mask_cov_rows.append({"class": cls, "mean_mask_cov": float(np.mean(covs)), "median_mask_cov": float(np.median(covs))})

    # --- Visualization 6 : Mask coverage distribution (box)
    if mask_cov_rows:
        # build long form
        long = []
        for cls in classes:
            for cov in by_class[cls].get("mask_coverages", []):
                long.append({"class": cls, "mask_cov": cov})
        if long:
            df_long = pd.DataFrame(long)
            fig6 = px.box(df_long, x="class", y="mask_cov", labels={"mask_cov":"Mask coverage %"}, title="Distribution mask coverage par classe")
            st.plotly_chart(fig6, use_container_width=True)
    # --- Afficher analyses automatiques dans un expander
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan if 'scan' in locals() else None, viz_id="viz6")
        else:
            st.markdown("Aucune analyse automatique disponible.")


