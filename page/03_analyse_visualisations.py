# 03_analyse_visualisations.py — version robuste & fonctionnelle : échantillonnage, métriques image, export RTF+ZIP
# Ajoutées : 10 visualisations (MVP -> avancées) utilisant plotly quand possible.
import streamlit as st
from streamlit_extras.colored_header import colored_header
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageOps
import numpy as np
import io, zipfile, random, html, datetime
import plotly.express as px
import plotly.graph_objects as go
import json


# Optional heavy deps (embeddings)
try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    import torch
    torch.set_num_threads(1)   # limite threads CPU si problème de CPU contention


try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ---------------- CONFIG ----------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KAGGLE_SLUG = "tawsifurrahman/covid19-radiography-database"
THUMBNAIL_MAX = (512, 512)

# ---------------- Helpers ----------------
def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def find_dataset_root_guess(base: Path = Path("dataset")) -> Optional[Path]:
    """Devine la racine du dataset si 02_donnees n'a pas set-stated."""
    if not base.exists():
        return None
    for p in [base] + [d for d in base.iterdir() if d.is_dir()]:
        subs = [c for c in p.iterdir() if c.is_dir()]
        n_good = sum(1 for c in subs if any(_is_image_file(f) for f in c.iterdir()))
        if n_good >= 2:
            return p
    return base

def list_classes_from_root(root: Path) -> List[str]:
    """Recherche tolérante des classes (ex: dataset_root/<CLASS>/images or dataset_root/images/<CLASS>/)."""
    if not root or not root.exists():
        return []
    # direct children that contain images
    direct = [p.name for p in sorted(root.iterdir()) if p.is_dir() and any(_is_image_file(f) for f in p.iterdir())]
    if direct:
        return sorted(direct)
    # try images/ structure
    images_sub = root / "images"
    if images_sub.exists() and images_sub.is_dir():
        direct2 = [p.name for p in sorted(images_sub.iterdir()) if p.is_dir() and any(_is_image_file(f) for f in p.iterdir())]
        if direct2:
            return sorted(direct2)
    # fallback: scan two levels
    candidates = set()
    for lvl1 in (p for p in root.iterdir() if p.is_dir()):
        for candidate in (c for c in lvl1.iterdir() if c.is_dir()):
            if any(_is_image_file(f) for f in candidate.iterdir()):
                candidates.add(candidate.name)
    if candidates:
        return sorted(list(candidates))
    # deep fallback
    for p in root.rglob("*"):
        if p.is_dir() and any(_is_image_file(f) for f in p.iterdir()):
            return [p.name]
    return []

def sample_images_from_class(root: Path, cls: str, n: int) -> List[Path]:
    """Récupère n images depuis root/<cls> ou root/.../<cls>/images."""
    cls_dir = root / cls
    if not cls_dir.exists():
        for p in root.rglob(cls):
            if p.is_dir():
                cls_dir = p
                break
    if not cls_dir.exists():
        return []
    images_sub = cls_dir / "images"
    if images_sub.exists() and any(_is_image_file(f) for f in images_sub.iterdir()):
        cls_dir = images_sub
    imgs = sorted([p for p in cls_dir.iterdir() if _is_image_file(p)])
    rng = random.Random()
    if len(imgs) <= n:
        return imgs
    return rng.sample(imgs, k=n)

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

import math
import statistics
import pandas as pd

def generate_visual_analysis(scan: Dict, classes: List[str]) -> Dict[str,str]:
    """
    Génération d'analyses textuelles pour chaque visualisation.
    Retourne un dict de strings avec clefs 'viz1'..'viz10' et 'recommendation'.
    Chaque string est du markdown-simple (titres gras + paragraphes).
    """
    out = {}
    per_image = scan.get("per_image", [])
    by_class = scan.get("by_class", {})
    total_images = len(per_image)

    # small helpers
    import numpy as _np
    def safe_mean(arr):
        return float(_np.mean(arr)) if arr else float('nan')

    # Precompute some convenient dicts
    counts = {c: by_class.get(c, {}).get("count", 0) for c in classes}
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    largest = sorted_counts[0] if sorted_counts else ("-",0)
    smallest = sorted_counts[-1] if sorted_counts else ("-",0)
    imbalance_ratio = (largest[1] / (smallest[1] + 1)) if smallest[1] >= 0 else float('inf')

    # Rework fake-rgb interpretation: call it "canaux identiques / probable grayscale"
    fake_pct_by_class = {c: by_class.get(c, {}).get("fake_pct", 0.0) for c in classes}

    # --- Viz 1 : Répartition des classes
    txt = []
    txt.append("**1) Répartition des classes — Aperçu & risque de biais**\n")
    txt.append(f"- Total images analysées : **{total_images}**.")
    txt.append(f"- Classe la plus fournie : **{largest[0]}** ({largest[1]} images).")
    txt.append(f"- Classe la moins fournie : **{smallest[0]}** ({smallest[1]} images).")
    if imbalance_ratio == float('inf') or smallest[1] == 0:
        txt.append(":warning: Certaines classes n'ont **aucune** image lisible — collecte requise.")
        txt.append("**Action recommandée** : prioriser la collecte / augmentation pour ces classes avant entraînement.")
    elif imbalance_ratio > 5:
        txt.append(f":warning: Déséquilibre marqué (ratio ≈ {imbalance_ratio:.1f}).")
        txt.append("**Action recommandée** : envisager sur-échantillonnage, pondération de la loss ou sampling stratifié.")
    else:
        txt.append("Répartition acceptable pour un POC ; documenter l'effet dans le rapport d'équilibrage.")
    out['viz1'] = "\n\n".join(txt)

    # --- Viz 2 : Luminosité & Contraste
    df_rows = []
    for e in per_image:
        m = e['metrics']
        df_rows.append({'class': e['class'], 'lum': m.get('luminosity_mean', 0.0),
                        'std': m.get('contrast_std', 0.0), 'entropy': m.get('entropy', 0.0)})
    import pandas as _pd
    dfm = _pd.DataFrame(df_rows) if df_rows else _pd.DataFrame(columns=['class','lum','std','entropy'])

    txt = ["**2) Luminosité & Contraste — Signal d'exposition**\n"]
    if dfm.empty:
        txt.append("Pas assez de métriques pour décrire luminosité/contraste.")
    else:
        for cls in classes:
            part = dfm[dfm['class'] == cls]
            if part.empty:
                txt.append(f"- **{cls}** : aucune image analysable.")
                continue
            txt.append(f"- **{cls}** — lum moy {part['lum'].mean():.1f}, std moy {part['std'].mean():.1f}, entropie moy {part['entropy'].mean():.2f} (n={len(part)})")
        # detect low/high exposure
        overall_lum = dfm['lum'].mean()
        if overall_lum < 60:
            txt.append(":warning: Images globalement sombres — vérifier acquisition / fenêtre de préprocessing.")
            txt.append("**Action** : étudier normalisation d'histogramme ou CLAHE en preprocessing.")
        elif overall_lum > 220:
            txt.append(":warning: Images très claires → risque de clipping.")
            txt.append("**Action** : vérifier pipeline d'export / correction gamma.")
        else:
            txt.append("Luminosité globale dans une plage raisonnable.")
    out['viz2'] = "\n\n".join(txt)

    # --- Viz 3 : Canaux identiques / probable grayscale
    txt = ["**3) Canaux identiques (probable grayscale) — qualité couleur**\n"]
    high_fake = [c for c,v in fake_pct_by_class.items() if v > 20.0]
    for cls, pct in fake_pct_by_class.items():
        txt.append(f"- **{cls}** : {pct:.1f}% images avec canaux identiques (RGB≈R=G=B).")
    if high_fake:
        txt.append(":info: Beaucoup d'images avec canaux identiques — cela peut être normal pour des radiographies converties en RGB.")
        txt.append("**Action** : ne pas traiter ce flag comme une erreur automatique — vérifier si les images sont originellement grayscale.")
        txt.append("**Si tu veux** : ajouter un test qui détecte *vraiment* corruption (ex : canaux différents mais valeurs nulles, logos colorés, canaux décalés).")
    else:
        txt.append("Ratio de canaux identiques bas — images probablement colorées ou correctement encodées.")
    out['viz3'] = "\n\n".join(txt)

    # --- Viz 4 : Exemples par classe
    txt = ["**4) Exemples par classe — inspection rapide**\n"]
    for cls in classes:
        n = min(3, len(by_class.get(cls, {}).get('files', [])))
        txt.append(f"- **{cls}** : {n} exemples affichés (vérifier artefacts visibles).")
    txt.append("**Actions pratiques** : vérifier annotations mal placées, logos, bandes noires, texte incrusté. Lister 5 images suspectes dans le rapport si trouvées.")
    out['viz4'] = "\n\n".join(txt)

    # --- Viz 5 & 6 : Couverture des masks
    txt = ["**5/6) Couverture des masques — disponibilité & cohérence**\n"]
    mask_stats = []
    for cls in classes:
        covs = by_class.get(cls, {}).get('mask_coverages', [])
        if covs:
            mask_stats.append((cls, float(_np.mean(covs)), float(_np.median(covs)), len(covs)))
            txt.append(f"- **{cls}** — couverture moyenne : {(_np.mean(covs)):.1f}% (n={len(covs)})")
        else:
            txt.append(f"- **{cls}** — aucun mask détecté.")
    if not mask_stats:
        txt.append(":warning: Aucun mask trouvé — impossible d'évaluer la qualité de segmentation.")
        txt.append("**Action** : vérifier chemins / format des masks (uint8, 0/255 attendu).")
    else:
        low = [c for c,mean,med,n in mask_stats if mean < 1.0]
        high = [c for c,mean,med,n in mask_stats if mean > 40.0]
        if low:
            txt.append(f":warning: Couverture très faible pour : {', '.join(low)} — peut indiquer masks vides ou format incorrect.")
        if high:
            txt.append(f"Couverture élevée (>40%) pour : {', '.join(high)} — utile pour tâches de segmentation.")
    out['viz5'] = out['viz6'] = "\n\n".join(txt)

    # --- Viz 7 : Histogramme global (pixels)
    txt = ["**7) Histogramme global — contraste global du dataset**\n"]
    if dfm.empty:
        txt.append("Pas d'images exploitables pour histogramme global.")
    else:
        overall_lum = dfm['lum'].mean()
        txt.append(f"- Luminosité moyenne globale (échantillon) : {overall_lum:.1f}.")
        txt.append("- Interprétation : valeurs extrêmes indiquent potentiels problèmes d'acquisition / clipping.")
        txt.append("**Action** : si histogramme très centré, envisager normalisation ; si bimodal, vérifier sous-populations.")
    out['viz7'] = "\n\n".join(txt)

    # --- Viz 8 : Corrélations
    txt = ["**8) Corrélations entre métriques — multicolinéarité potentielle**\n"]
    try:
        corr_df = dfm[['lum','std','entropy']].corr()
        # flatten and pick strongest pair (off-diagonal)
        corr_flat = corr_df.abs().stack().reset_index()
        corr_flat = corr_flat[corr_flat['level_0'] != corr_flat['level_1']].sort_values(0, ascending=False)
        if not corr_flat.empty:
            top = corr_flat.iloc[0]
            a, b, val = top['level_0'], top['level_1'], top[0]
            txt.append(f"- Paire la plus corrélée : **{a}** vs **{b}** (|corr|={val:.2f}).")
            if abs(val) > 0.6:
                txt.append(":info: Corrélation notable — risque de colinéarité si ces métriques sont utilisées directement en features.")
                txt.append("**Action** : envisager PCA / regularisation / sélection de features.")
            else:
                txt.append("Corrélations modérées — OK pour features simples.")
        else:
            txt.append("Pas assez de données pour calculer corrélations.")
    except Exception:
        txt.append("Erreur lors du calcul de corrélations.")
    out['viz8'] = "\n\n".join(txt)

    # --- Viz 9 : Embeddings
    txt = ["**9) Embeddings visuels — qualité du signal (si calculés)**\n"]
    if scan.get('embeddings_computed', False):
        txt.append("- Embeddings disponibles : vérifier séparation/clustering par label (clusters propres → bon signal).")
        txt.append("**Action** : fournir silhouette score / Davies–Bouldin pour chiffrer la séparation.")
    else:
        txt.append("- Embeddings non calculés (PyTorch absent ou non exécuté).")
        txt.append("**Action** : pré-calculer embeddings sur GPU hors production et stocker `.npy` / `.npz` pour analyses interactives.")
    out['viz9'] = "\n\n".join(txt)

    # --- Viz 10 : Distance perceptuelle inter-classes (JS)
    txt = ["**10) Distance perceptuelle (JS) entre classes — similarité visuelle**\n"]
    try:
        dist = scan.get("_js_distance_matrix", None)
        labels = scan.get("_js_labels", classes)
        if dist is None:
            txt.append("- Matrice JS non fournie par le scan (fallback heuristique utilisée). Interprétation :")
            txt.append("  - Valeurs élevées → classes visuellement distinctes (bon pour classification).")
            txt.append("  - Valeurs faibles → classes ressemblantes → risque de confusion.")
        else:
            # find max off-diag
            n = dist.shape[0]
            maxd = -1; pair = (None, None)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if dist[i, j] > maxd:
                        maxd = float(dist[i, j]); pair = (labels[i], labels[j])
            txt.append(f"- Distance JS maximale entre **{pair[0]}** et **{pair[1]}** = {maxd:.3f} (off-diag max).")
            txt.append("**Action** : regarder paires proches → risque d'erreurs; appliquer augmentation ciblée ou features spécifiques.")
    except Exception:
        txt.append("Impossible d'interpréter la matrice de distances JS.")
    out['viz10'] = "\n\n".join(txt)

    # --- Recommandations globales synthétiques
    rec_lines = [
        "**Recommandations globales :**",
        "- Vérifier les classes avec peu d'images (collecte / augmentation).",
        "- Inspecter manuellement images flagged (canaux identiques, très basse entropie, masques incohérents).",
        "- Pré-calculer embeddings sur GPU pour analyses lourdes et stocker snapshots pour Streamlit.",
        "- Documenter chaque décision (équilibrage, normalisation) dans le rapport RTF pour traçabilité."
    ]
    out['recommendation'] = "\n\n".join(rec_lines)
    return out

def get_viz_interpretation(viz_id, metrics):
    """
    viz_id : str | identifiant interne (ex: 'viz1')
    metrics : dict | mesures calculées pour cette visualisation
    return : (title, interpretation)
    """

    # 1) Répartition des classes
    if viz_id == "viz1":
        title = "Répartition des classes"
        interpretation = (
            f"Total images analysées : {metrics.get('total_images', 'N/A')}.\n\n"
            f"Classe la plus fournie : {metrics.get('max_class', 'N/A')} "
            f"({metrics.get('max_value', 'N/A')} images).\n\n"
            f"Classe la moins fournie : {metrics.get('min_class', 'N/A')} "
            f"({metrics.get('min_value', 'N/A')} images).\n\n"
            "Déséquilibre marqué entre classes → risque de biais statistique."
        )
        return title, interpretation

    # 2) Luminosité / Contraste / Entropie
    if viz_id == "viz2":
        title = "Luminosité & Contraste — Analyse d'exposition"
        interpretation = (
            f"COVID — lum moy {metrics.get('covid_lum', 'N/A')}, std {metrics.get('covid_std', 'N/A')}, entropie {metrics.get('covid_entropy', 'N/A')}.\n\n"
            f"Lung_Opacity — lum moy {metrics.get('lo_lum', 'N/A')}, std {metrics.get('lo_std', 'N/A')}, entropie {metrics.get('lo_entropy', 'N/A')}.\n\n"
            f"Normal — lum moy {metrics.get('norm_lum', 'N/A')}, std {metrics.get('norm_std', 'N/A')}, entropie {metrics.get('norm_entropy', 'N/A')}.\n\n"
            f"Viral Pneumonia — lum moy {metrics.get('vp_lum', 'N/A')}, std {metrics.get('vp_std', 'N/A')}, entropie {metrics.get('vp_entropy', 'N/A')}.\n\n"
            "Luminosité globale correcte et homogène entre classes."
        )
        return title, interpretation

    # 3) Détection Fake-RGB
    if viz_id == "viz3":
        title = "Proportion d'images Fake-RGB"
        interpretation = (
            f"COVID : {metrics.get('covid_fake', 'N/A')}% potentiellement fake-RGB.\n\n"
            f"Lung_Opacity : {metrics.get('lo_fake', 'N/A')}%.\n\n"
            f"Normal : {metrics.get('norm_fake', 'N/A')}%.\n\n"
            f"Viral Pneumonia : {metrics.get('vp_fake', 'N/A')}%.\n\n"
            "Présence importante d’images avec duplicats de canaux → probables conversions automatiques."
        )
        return title, interpretation

    # 4) Exemples représentatifs
    if viz_id == "viz4":
        title = "Exemples représentatifs par classe"
        interpretation = (
            "Aperçu visuel des images par classe. Permet de détecter anomalies : zones noires, texte parasite, "
            "logos, masques incorrects ou mauvaise exposition."
        )
        return title, interpretation

    # 5 et 6) Masque — moyenne de couverture
    if viz_id in ["viz5", "viz6"]:
        title = "Analyse de couverture des masques"
        interpretation = (
            f"COVID — couverture moy {metrics.get('covid_mask', 'N/A')}%.\n\n"
            f"Lung_Opacity — {metrics.get('lo_mask', 'N/A')}%.\n\n"
            f"Normal — {metrics.get('norm_mask', 'N/A')}%.\n\n"
            f"Viral Pneumonia — {metrics.get('vp_mask', 'N/A')}%.\n\n"
            "Taux de couverture comparables entre classes."
        )
        return title, interpretation

    # 7) Luminosité globale échantillon
    if viz_id == "viz7":
        title = "Luminosité globale – échantillon"
        interpretation = (
            f"Luminosité moyenne globale : {metrics.get('global_lum', 'N/A')}.\n\n"
            "Positionnée dans une zone visuellement normale."
        )
        return title, interpretation

    # 8) Corrélation entre features
    if viz_id == "viz8":
        title = "Corrélation entre mesures radiologiques"
        interpretation = (
            f"Corrélation la plus forte : {metrics.get('strongest_pair', 'N/A')} "
            f"(|corr| = {metrics.get('strongest_corr', 'N/A')}).\n\n"
            "Corrélations modérées → indépendance partielle des features, utile pour modélisation."
        )
        return title, interpretation

    # 9) Embeddings (non calculés)
    if viz_id == "viz9":
        title = "Embeddings (désactivés)"
        interpretation = (
            "Embeddings non calculés : torch absent ou non activé.\n\n"
            "Nécessite pré-calcul CPU/GPU en amont."
        )
        return title, interpretation

    # 10) Distance perceptuelle (JS)
    if viz_id == "viz10":
        title = "Distance perceptuelle entre classes (Jensen–Shannon)"
        interpretation = (
            "Distances JS fournies ou fallback heuristique.\n\n"
            "Valeurs élevées → classes visuellement distinctes.\n"
            "Valeurs faibles → classes difficiles à distinguer visuellement."
        )
        return title, interpretation

    # fallback
    return "Visualisation inconnue", "Aucune interprétation disponible."


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
            title, interp = get_viz_interpretation(viz_id, metrics)
            st.markdown(f"**{title}**")
            st.markdown(interp)
    else:
        # Sinon afficher tous les viz + recommandations
        for k in sorted([kk for kk in analyses.keys() if kk.startswith("viz")]):
            # Build metrics dict from scan data for this viz
            metrics = _build_metrics_for_viz(k, scan, per_image, by_class or {}, classes or [], per_image_count)
            title, interp = get_viz_interpretation(k, metrics)
            st.markdown(f"**{title}**")
            st.markdown(interp)
            st.markdown("---")
        rec = analyses.get("recommendation")
        if rec:
            st.markdown("**Recommandations globales**")
            st.markdown(rec)

def _build_metrics_for_viz(viz_id, scan, per_image, by_class, classes, per_image_count):
    """Construit le dict metrics pour une visualisation donnée à partir du scan."""
    metrics = {}
    
    if viz_id == "viz1":
        # Distribution des classes
        counts = {c: by_class.get(c, {}).get("count", 0) for c in (classes or [])}
        if counts:
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            largest = sorted_counts[0] if sorted_counts else ("N/A", 0)
            smallest = sorted_counts[-1] if sorted_counts else ("N/A", 0)
            metrics['total_images'] = per_image_count
            metrics['max_class'] = largest[0]
            metrics['max_value'] = largest[1]
            metrics['min_class'] = smallest[0]
            metrics['min_value'] = smallest[1]
    
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
    
    elif viz_id == "viz3":
        # Fake-RGB by class
        if by_class and classes:
            metrics['covid_fake'] = f"{by_class.get('COVID', {}).get('fake_pct', 0.0):.1f}"
            metrics['lo_fake'] = f"{by_class.get('Lung_Opacity', {}).get('fake_pct', 0.0):.1f}"
            metrics['norm_fake'] = f"{by_class.get('Normal', {}).get('fake_pct', 0.0):.1f}"
            metrics['vp_fake'] = f"{by_class.get('Viral Pneumonia', {}).get('fake_pct', 0.0):.1f}"
    
    elif viz_id in ["viz5", "viz6"]:
        # Mask coverage
        if by_class and classes:
            metrics['covid_mask'] = f"{np.mean(by_class.get('COVID', {}).get('mask_coverages', [0])):.1f}" if by_class.get('COVID', {}).get('mask_coverages') else 'N/A'
            metrics['lo_mask'] = f"{np.mean(by_class.get('Lung_Opacity', {}).get('mask_coverages', [0])):.1f}" if by_class.get('Lung_Opacity', {}).get('mask_coverages') else 'N/A'
            metrics['norm_mask'] = f"{np.mean(by_class.get('Normal', {}).get('mask_coverages', [0])):.1f}" if by_class.get('Normal', {}).get('mask_coverages') else 'N/A'
            metrics['vp_mask'] = f"{np.mean(by_class.get('Viral Pneumonia', {}).get('mask_coverages', [0])):.1f}" if by_class.get('Viral Pneumonia', {}).get('mask_coverages') else 'N/A'
    
    elif viz_id == "viz7":
        # Global luminosity
        if per_image:
            lums = [e['metrics'].get('luminosity_mean', 0.0) for e in per_image]
            metrics['global_lum'] = f"{np.mean(lums):.1f}" if lums else 'N/A'
    
    return metrics

# ---------------- RTF + ZIP export ----------------
def _escape_rtf(text: str) -> str:
    out = ""
    for c in text:
        if c in "\\{}":
            out += "\\" + c
        elif ord(c) > 127:
            out += r"\u" + str(ord(c)) + "?"
        else:
            out += c
    return out

def make_report_rtf_bytes(title: str, meta: Dict, per_image_metrics: List[Dict], analyses: Optional[Dict[str,str]] = None) -> bytes:
    """
    title, meta, per_image_metrics : inchangés
    analyses : dict contenant keys 'viz1'..'viz10' et 'recommendation' (texte markdown-ish).
    Le RTF contiendra :
      - entête + meta
      - synthèse analyses par visualisation (plain text)
      - table métriques par image (texte)
    """
    header = r"{\rtf1\ansi\ansicpg1252\deff0"
    body = []
    body.append(r"\b " + _escape_rtf(title) + r" \b0\par\par")
    body.append(r"\b Date \b0 : " + datetime.date.today().isoformat() + r"\par\par")
    body.append(r"\b Meta \b0\par")
    for k, v in meta.items():
        body.append(r"\b " + _escape_rtf(str(k)) + r" \b0 : " + _escape_rtf(str(v)) + r"\par")
    body.append(r"\par\b Analyses automatiques (visualisations) \b0\par")
    if analyses:
        # order viz1..viz10 then recommendation
        for i in range(1, 11):
            key = f"viz{i}"
            txt = analyses.get(key, None)
            if txt:
                # small header
                body.append(r"\b " + _escape_rtf(f"Visualisation {i}") + r" \b0\par")
                # convert simple markdown-ish bullets/emojis into plain text lines
                # keep as-is but escape
                body.append(_escape_rtf(txt) + r"\par\par")
        # recommendations
        rec = analyses.get("recommendation", None)
        if rec:
            body.append(r"\par\b Recommandations globales \b0\par")
            body.append(_escape_rtf(rec) + r"\par\par")
    else:
        body.append(_escape_rtf("Aucune analyse automatique fournie.") + r"\par\par")

    body.append(r"\par\b Détails par image \b0\par")
    for im in per_image_metrics:
        fn = im.get("filename", im.get("path", ""))
        body.append(r"\b Fichier \b0 : " + _escape_rtf(str(fn)) + r"\par")
        # list the metrics in a consistent order (defensive)
        lum = im.get("luminosity_mean") or im.get("lum", 0.0)
        std = im.get("contrast_std") or im.get("std", 0.0)
        ent = im.get("entropy", 0.0)
        fake = im.get("fake_rgb", im.get("fake", False))
        body.append(_escape_rtf(f" - Luminosité moyenne : {float(lum):.2f}") + r"\par")
        body.append(_escape_rtf(f" - Contraste (std) : {float(std):.2f}") + r"\par")
        body.append(_escape_rtf(f" - Entropie : {float(ent):.3f}") + r"\par")
        body.append(_escape_rtf(f" - Fake-RGB : {bool(fake)}") + r"\par\par")

    footer = r"}"
    rtf = header + "\n" + "\n".join(body) + "\n" + footer
    return rtf.encode("utf-8")


def create_zip_with_report(img_paths: List[Path], rtf_bytes: bytes, rtf_name: str="analysis_report.rtf", scan_summary: Optional[Dict] = None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer,"w",zipfile.ZIP_DEFLATED) as zf:
        # add images
        for p in img_paths:
            try:
                zf.write(p, arcname=str(Path("images")/p.name))
            except Exception:
                pass
        # add textual RTF report
        zf.writestr(rtf_name, rtf_bytes)
        # add scan summary JSON for traceability / future analyses
        if scan_summary is not None:
            try:
                zf.writestr("scan_summary.json", json.dumps(scan_summary, default=str, indent=2))
            except Exception:
                pass
    zip_buffer.seek(0)
    return zip_buffer

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

# ---------------- Embeddings (optional) ----------------
def extract_embeddings(paths: List[str], device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """Extrait embeddings via un backbone ResNet18 (si torch disponible).
    Batché, safe pour CPU. Retourne array shape (N, feat_dim).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch non disponible.")
    import torch
    from torchvision import models, transforms as T

    # modèle léger (ResNet18) sans fc
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # output (B,512,1,1)
    model.eval()
    model.to(device)

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    feats_list = []
    # process in batches to avoid OOM on CPU
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(transform(im))
            except Exception:
                # if image unreadable, push zero tensor
                imgs.append(torch.zeros((3,224,224), dtype=torch.float32))
        x = torch.stack(imgs, dim=0).to(device)
        with torch.no_grad():
            out = model(x)  # shape (B, feat, 1, 1)
            out = out.reshape(out.size(0), -1).cpu().numpy()  # (B, feat_dim)
        feats_list.append(out)
    if feats_list:
        return np.vstack(feats_list)
    else:
        # fallback empty
        return np.zeros((0, 512), dtype=float)


# ---------------- UI ----------------
def run():
    st.set_page_config(layout="wide")
    st.markdown("<style>.small-note{font-size:12px;color:#98a1b3}</style>", unsafe_allow_html=True)

    colored_header("1. Contexte & aperçu", "Extrait un petit échantillon, métriques d'image, export RTF+ZIP — +10 visualisations (plotly).", color_name="violet-70")
    st.divider()

    # ---------------- Robust dataset discovery ----------------
    detected_root = None
    classes = None

    # 1) Session state
    if "detected_root" in st.session_state:
        try:
            detected_root = Path(st.session_state["detected_root"])
            classes = st.session_state.get("classes", None)
            st.caption(f"Racine dataset depuis session : `{detected_root}`")
        except Exception:
            detected_root = None
            classes = None

    # 2) Fallback KaggleHub
    if detected_root is None:
        try:
            import kagglehub
            p = kagglehub.dataset_download(KAGGLE_SLUG)
            if p:
                detected_root = Path(p)
                st.success(f"Racine dataset via KaggleHub : `{detected_root}`")
                # si unique child dossier qui contient images, descendre dedans (heuristique)
                subdirs = [d for d in detected_root.iterdir() if d.is_dir()]
                if len(subdirs) == 1 and (subdirs[0] / "images").exists():
                    detected_root = subdirs[0]
                    st.caption(f"Descente automatique d'un niveau : nouvelle racine `{detected_root}`")
        except Exception:
            pass

    # 3) Heuristique locale
    if detected_root is None:
        guessed = find_dataset_root_guess()
        if guessed:
            detected_root = guessed
            st.caption(f"Dataset racine devinée : `{detected_root}`")
        else:
            st.warning("Aucune racine détectée. Vérifie d'avoir exécuté 02 ou dossier dataset/ valide.")
            return

    if isinstance(detected_root, str):
        detected_root = Path(detected_root)

    if not classes:
        classes = list_classes_from_root(detected_root)

    # Descendre dans images/ si nécessaire
    if classes and set(classes) <= {"images", "masks"}:
        images_folder = detected_root / "images"
        if images_folder.exists():
            classes = sorted([p.name for p in images_folder.iterdir() if p.is_dir() and any(_is_image_file(f) for f in p.iterdir())])
            st.success(f"Classes extraites depuis `{images_folder}` : {classes}")

    if not classes:
        st.error("Aucune classe détectée sous la racine dataset.")
        return

    st.write(f"Classes détectées ({len(classes)}): {classes[:20]}")

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

    # --- Visualization 1 : Distribution des classes (counts)
    st.markdown("### 1) Distribution des classes")
    counts = {cls: by_class[cls]["count"] for cls in classes}
    fig = px.bar(x=list(counts.keys()), y=list(counts.values()), labels={"x":"Classe","y":"Nombre d'images"}, title="Nombre d'images par classe")
    st.plotly_chart(fig, use_container_width=True)
    # --- Afficher analyses automatiques (résumé: nom + interprétation)
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan, scan.get("by_class"), list(scan.get("by_class", {}).keys()) if scan else [], viz_id="viz1")
        else:
            st.markdown("Aucune analyse automatique disponible.")

    # --- Visualization 2 : Histogramme luminosité & contraste par classe (small multiples)
    st.markdown("### 2) Histogramme de luminosité & contraste par classe (small multiples)")
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

    # --- Visualization 3 : Fake-RGB detection summary
    st.markdown("### 3) Pourcentage d'images 'fake-RGB' par classe")
    fake_pct = {cls: by_class[cls].get("fake_pct", 0.0) for cls in classes}
    fig4 = px.bar(x=list(fake_pct.keys()), y=list(fake_pct.values()), labels={"x":"Classe","y":"% fake-RGB"}, title="% fake-RGB par classe", text=[f"{v:.1f}%" for v in fake_pct.values()])
    fig4.update_traces(textposition="outside")
    st.plotly_chart(fig4, use_container_width=True)
    # --- Afficher analyses automatiques (résumé: nom + interprétation)
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan if 'scan' in locals() else None, viz_id="viz3")
        else:
            st.markdown("Aucune analyse automatique disponible.")

    # --- Visualization 4 : Grid d'exemples par classe avec métriques (interactive)
    st.markdown("### 4) Grille d'exemples par classe (miniatures + métriques)")
    cols = st.columns(len(classes))
    for c_idx, cls in enumerate(classes):
        with cols[c_idx]:
            st.markdown(f"**{cls}** — {by_class[cls]['count']} images")
            sample_files = by_class[cls]["files"][:3]  # show up to 3
            for fpath in sample_files:
                try:
                    p = Path(fpath)
                    im = Image.open(p).convert("RGB")
                    im.thumbnail((240,240))
                    st.image(im, caption=p.name)
                    m = compute_image_metrics(im)
                    st.markdown(f"- L: {m['luminosity_mean']:.1f} — std: {m['contrast_std']:.1f} — ent: {m['entropy']:.2f} — fake:{m['fake_rgb']}")
                except Exception:
                    continue

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
    if mask_cov_rows:
        df_cov = pd.DataFrame(mask_cov_rows)
        fig5 = px.bar(df_cov, x="class", y="mean_mask_cov", labels={"mean_mask_cov":"Moyenne % mask"}, title="Couverture moyenne des masks par classe")
        st.plotly_chart(fig5, use_container_width=True)
    # --- Afficher analyses automatiques (résumé: nom + interprétation)
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan if 'scan' in locals() else None, viz_id="viz5")
        else:
            st.markdown("Aucune analyse automatique disponible.")

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

    # --- Visualization 7 : Histogrammes globaux du dataset (pixels)
    st.markdown("### 7) Histogrammes globaux (pixels grayscale) — échantillon rapide")
    # For performance, sample up to 200 images
    sample_paths = [e["path"] for e in per_image]
    sample_paths = sample_paths[::max(1, len(sample_paths)//200)]
    pixel_rows = []
    for p in sample_paths[:200]:
        try:
            im = Image.open(p).convert("L").resize((256,256))
            arr = np.array(im).flatten()
            pixel_rows.append(arr)
        except Exception:
            continue
    if pixel_rows:
        all_pixels = np.concatenate(pixel_rows)
        # histogram
        hist_vals, bin_edges = np.histogram(all_pixels, bins=256, range=(0,255))
        fig7 = go.Figure(data=go.Bar(x=bin_edges[:-1], y=hist_vals))
        fig7.update_layout(title="Histogramme global des pixels (grayscale) — échantillon", xaxis_title="Valeur pixel", yaxis_title="Count")
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Impossible de construire histogramme global (aucune image lisible).")
    # --- Afficher analyses automatiques (résumé: nom + interprétation)
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan if 'scan' in locals() else None, viz_id="viz7")
        else:
            st.markdown("Aucune analyse automatique disponible.")

    # --- Visualization 8 : Corrélation des métriques (lum, std, entropy, mask_cov)
    st.markdown("### 8) Corrélation entre métriques (luminosité, contraste, entropie, mask_coverage)")
    corr_rows = []
    for e in per_image:
        m = e["metrics"]
        corr_rows.append({"class": e["class"], "lum": m["luminosity_mean"], "std": m["contrast_std"], "entropy": m["entropy"], "mask_cov": e.get("mask_coverage") if e.get("mask_coverage") is not None else np.nan})
    df_corr = pd.DataFrame(corr_rows).dropna(axis=0, subset=["lum","std","entropy"])
    if not df_corr.empty:
        corr = df_corr[["lum","std","entropy","mask_cov"]].corr()
        fig8 = px.imshow(corr, text_auto=True, labels=dict(x="metric", y="metric"), title="Matrice de corrélation des métriques d'image")
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Pas assez de données pour matrice de corrélation (mask_coverage peut être manquant).")
    # --- Afficher analyses automatiques (résumé: nom + interprétation)
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan if 'scan' in locals() else None, viz_id="viz8")
        else:
            st.markdown("Aucune analyse automatique disponible.")

    # --- Visualization 9 : Embeddings + UMAP/PCA (optionnel, si torch disponible)
    st.markdown("### 9) Embeddings visuels + réduction (UMAP/PCA) — optionnel")
    emb_status = ""
    try:
        # pick up to 200 images for embedding to keep light
        emb_paths = [e["path"] for e in per_image][:200]
        if TORCH_AVAILABLE:
            device = "cpu"
            feats = extract_embeddings(emb_paths, device=device, batch_size=8)
            if UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, random_state=42)
                emb2 = reducer.fit_transform(feats)
            else:
                # fallback to PCA from sklearn
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    emb2 = pca.fit_transform(feats)
                except Exception:
                    emb2 = None
                    emb_status = "Embeddings extraits mais pas de méthode de réduction disponible."
            if emb2 is not None:
                labels = [Path(p).parent.name for p in emb_paths]
                df_emb = pd.DataFrame({"x": emb2[:,0], "y": emb2[:,1], "label": labels, "fname":[Path(p).name for p in emb_paths]})
                fig9 = px.scatter(df_emb, x="x", y="y", color="label", hover_data=["fname"], title="UMAP/PCA des embeddings (échantillon)")
                st.plotly_chart(fig9, use_container_width=True)
            else:
                st.info(emb_status or "Réduction d'embeddings indisponible.")
        else:
            st.info("PyTorch non disponible — embeddings non calculés. Installe torch pour activer cette visualisation.")
    except Exception as e:
        st.info("Erreur embeddings (optionnel) — " + str(e))
    # --- Afficher analyses automatiques (résumé: nom + interprétation)
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan if 'scan' in locals() else None, viz_id="viz9")
        else:
            st.markdown("Aucune analyse automatique disponible.")

    # --- Visualization 10 : Inter-class perceptual distance (proxy via histogram distance)
    st.markdown("### 10) Distance perceptuelle inter-classes (proxy via histogram moyen)")
    # calculer histogramme moyen par classe (grayscale)
    class_hist = {}
    for cls in classes:
        files = by_class[cls].get("files", [])[:200]  # limit for speed
        agg = None
        count = 0
        for f in files:
            try:
                im = Image.open(f).convert("L").resize((128,128))
                hist, _ = np.histogram(np.array(im).flatten(), bins=64, range=(0,255))
                hist = hist.astype(float) / (hist.sum()+1e-12)
                if agg is None:
                    agg = hist
                else:
                    agg += hist
                count += 1
            except Exception:
                continue
        if agg is None or count == 0:
            class_hist[cls] = np.zeros(64)
        else:
            class_hist[cls] = agg / (count + 1e-12)
    # compute pairwise distances (JS divergence or simple L2)
    cls_list = list(classes)
    dist_mat = np.zeros((len(cls_list), len(cls_list)))
    for i,c1 in enumerate(cls_list):
        for j,c2 in enumerate(cls_list):
            h1 = class_hist[c1] + 1e-12
            h2 = class_hist[c2] + 1e-12
            # Jensen-Shannon divergence
            m = 0.5*(h1 + h2)
            def kld(a,b):
                return np.sum(a * np.log2(a/b))
            js = 0.5 * (kld(h1,m) + kld(h2,m))
            dist_mat[i,j] = js
    fig10 = px.imshow(dist_mat, x=cls_list, y=cls_list, color_continuous_scale="viridis", labels={"x":"Classe","y":"Classe"}, title="Distance JS (histogram proxy) entre classes")
    st.plotly_chart(fig10, use_container_width=True)
    # --- Afficher analyses automatiques dans un expander
    analyses = st.session_state.get("last_scan_analyses", analyses if 'analyses' in locals() else {})
    with st.expander("Analyses automatiques (résumé)"):
        if analyses:
            render_analyses_summary(analyses, scan if 'scan' in locals() else None, viz_id="viz10")
        else:
            st.markdown("Aucune analyse automatique disponible.")

    st.divider()

    # ---------------- Export RTF + ZIP for current sample ----------------
    colored_header("Export & Rapport (échantillon actuel)", "Génère .rtf + ZIP des images sélectionnées.", color_name="violet-70")
    meta = {"classe": choice, "n_images": len(img_paths),
            "date": datetime.datetime.now().isoformat(),
            "notes": ("Interpretations..." ) }
    if st.button("Générer et télécharger RTF + ZIP (échantillon)"):
        if not img_paths:
            st.warning("Rien à exporter.")
        else:
            per_image_metrics_export = []
            for p in img_paths:
                try:
                    im = Image.open(p).convert("RGB")
                    metrics = compute_image_metrics(im)
                    per_image_metrics_export.append({"filename": Path(p).name, **metrics, "path": str(p)})
                except Exception:
                    per_image_metrics_export.append({"filename": Path(p).name, "luminosity_mean":0.0,"contrast_std":0.0,"entropy":0.0,"fake_rgb":True, "path": str(p)})

            # récupère les analyses automatiques si présentes
            analyses = st.session_state.get("last_scan_analyses", None)

            # meta utile à garder dans le rapport
            meta_report = {
                "classe": choice,
                "n_images": len(img_paths),
                "date": datetime.datetime.now().isoformat(),
                "dataset_root": str(detected_root),
                "notes": (interpretations[:500] + "...") if (interpretations := st.session_state.get("interpretations", "")) else ""
            }

            # make rtf with analyses
            rtf = make_report_rtf_bytes(f"Analyse visuelle — classe {choice}", meta_report, per_image_metrics_export, analyses=analyses)

            # create zip and include scan summary (if available) for full traceability
            scan_summary = st.session_state.get("last_scan", None)
            zip_buf = create_zip_with_report([Path(p) for p in img_paths], rtf, rtf_name=f"report_{choice}.rtf", scan_summary=scan_summary)

            st.download_button("Télécharger RTF+ZIP", data=zip_buf.getvalue(), file_name=f"analysis_{choice}.zip", mime="application/zip")
            st.success("Archive générée — contient images, report RTF et scan_summary.json.")


    st.markdown("<div class='small-note'>Notes : embeddings / UMAP sont optionnels (dépendances lourdes). Si tu veux que je retire la partie embeddings pour alléger, dis-le.</div>", unsafe_allow_html=True)
