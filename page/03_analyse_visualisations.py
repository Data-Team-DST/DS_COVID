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

def make_report_rtf_bytes(title: str, meta: Dict, per_image_metrics: List[Dict]) -> bytes:
    header = r"{\rtf1\ansi\ansicpg1252\deff0"
    body = [r"\b " + _escape_rtf(title) + r" \b0\par\par",
            r"\b Date \b0 : " + datetime.date.today().isoformat() + r"\par\par",
            r"\b Meta \b0\par"]
    for k,v in meta.items():
        body.append(r"\b " + _escape_rtf(str(k)) + r" \b0 : " + _escape_rtf(str(v)) + r"\par")
    body.append(r"\par\b Analyses par image \b0\par")
    for im in per_image_metrics:
        body.append(r"\b Fichier \b0 : " + _escape_rtf(im.get("filename","")) + r"\par")
        body.append(_escape_rtf(f" - Luminosité moyenne : {im.get('luminosity_mean'):.2f}") + r"\par")
        body.append(_escape_rtf(f" - Contraste (std) : {im.get('contrast_std'):.2f}") + r"\par")
        body.append(_escape_rtf(f" - Entropie : {im.get('entropy'):.3f}") + r"\par")
        body.append(_escape_rtf(f" - Fake-RGB : {im.get('fake_rgb')}") + r"\par\par")
    rtf = header + "\n" + "\n".join(body) + "\n}"
    return rtf.encode("utf-8")

def create_zip_with_report(img_paths: List[Path], rtf_bytes: bytes, rtf_name: str="analysis_report.rtf"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in img_paths:
            try:
                zf.write(p, arcname=str(Path("images")/p.name))
            except Exception:
                pass
        zf.writestr(rtf_name, rtf_bytes)
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
    fig3 = px.violin(df_metrics, x="class", y="std", box=True, points="all", labels={"std":"Contraste (std)"}, title="Distribution du contraste (std) par classe")
    st.plotly_chart(fig3, use_container_width=True)

    # --- Visualization 3 : Fake-RGB detection summary
    st.markdown("### 3) Pourcentage d'images 'fake-RGB' par classe")
    fake_pct = {cls: by_class[cls].get("fake_pct", 0.0) for cls in classes}
    fig4 = px.bar(x=list(fake_pct.keys()), y=list(fake_pct.values()), labels={"x":"Classe","y":"% fake-RGB"}, title="% fake-RGB par classe", text=[f"{v:.1f}%" for v in fake_pct.values()])
    fig4.update_traces(textposition="outside")
    st.plotly_chart(fig4, use_container_width=True)

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
                    per_image_metrics_export.append({"filename": Path(p).name, **metrics})
                except Exception:
                    per_image_metrics_export.append({"filename": Path(p).name, "luminosity_mean":0.0,"contrast_std":0.0,"entropy":0.0,"fake_rgb":True})
            rtf = make_report_rtf_bytes(f"Analyse visuelle — classe {choice}", meta, per_image_metrics_export)
            zip_buf = create_zip_with_report(img_paths, rtf, rtf_name=f"report_{choice}.rtf")
            st.download_button("Télécharger RTF+ZIP", data=zip_buf.getvalue(), file_name=f"analysis_{choice}.zip", mime="application/zip")
            st.success("Archive générée — contient images et rapport.")

    st.markdown("<div class='small-note'>Notes : embeddings / UMAP sont optionnels (dépendances lourdes). Si tu veux que je retire la partie embeddings pour alléger, dis-le.</div>", unsafe_allow_html=True)
