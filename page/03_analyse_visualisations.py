# 03_analyse_visualisations.py — version robuste & fonctionnelle : échantillonnage, métriques, export RTF+ZIP
import streamlit as st
from streamlit_extras.colored_header import colored_header
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import numpy as np
import io, zipfile, random, html, datetime
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KAGGLE_SLUG = "tawsifurrahman/covid19-radiography-database"
THUMBNAIL_MAX = (512,512)

# ---------------- Helpers ----------------
def _is_image_file(p: Path):
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
    if not root or not root.exists():
        return []
    direct = [p.name for p in sorted(root.iterdir()) if p.is_dir() and any(_is_image_file(f) for f in p.iterdir())]
    if direct:
        return sorted(direct)
    candidates = set()
    for lvl1 in (p for p in root.iterdir() if p.is_dir()):
        images_sub = lvl1 / "images"
        if images_sub.exists() and any(_is_image_file(f) for f in images_sub.iterdir()):
            subdirs = [d for d in lvl1.iterdir() if d.is_dir()]
            if subdirs:
                for s in subdirs:
                    if any(_is_image_file(f) for f in s.iterdir()):
                        candidates.add(s.name)
            else:
                candidates.add(lvl1.name)
        else:
            for candidate in (c for c in lvl1.iterdir() if c.is_dir()):
                if any(_is_image_file(f) for f in candidate.iterdir()):
                    candidates.add(candidate.name)
    if not candidates:
        for p in root.rglob("*"):
            if p.is_dir() and any(_is_image_file(f) for f in p.iterdir()):
                candidates.add(p.name)
    return sorted(list(candidates))

def sample_images_from_class(root: Path, cls: str, n: int) -> List[Path]:
    cls_dir = root / cls
    if not cls_dir.exists():
        for p in root.rglob(cls):
            if p.is_dir():
                cls_dir = p
                break
    if not cls_dir.exists():
        return []

    # Descendre dans images/ si présent
    images_sub = cls_dir / "images"
    if images_sub.exists() and any(_is_image_file(f) for f in images_sub.iterdir()):
        cls_dir = images_sub

    imgs = sorted([p for p in cls_dir.iterdir() if _is_image_file(p)])
    rng = random.Random()
    if len(imgs) <= n:
        return imgs
    return rng.sample(imgs, k=n)


def compute_image_metrics(img: Image.Image) -> Dict:
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r,g,b = arr[...,0], arr[...,1], arr[...,2]
    L_channel = 0.299*r + 0.587*g + 0.114*b
    fake_rgb = bool(np.allclose(r,g) and np.allclose(g,b))
    mean_lum = float(np.mean(L_channel))
    std_lum = float(np.std(L_channel))
    hist, _ = np.histogram(L_channel.flatten(), bins=256, range=(0,255))
    probs = hist/(hist.sum()+1e-12)
    probs = probs[probs>0]
    entropy = float(-(probs*np.log2(probs)).sum()) if probs.size>0 else 0.0
    return {"luminosity_mean": mean_lum, "contrast_std": std_lum, "entropy": entropy, "fake_rgb": fake_rgb}

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
        body.append(r"\b "+_escape_rtf(str(k))+r" \b0 : "+_escape_rtf(str(v))+r"\par")
    body.append(r"\par\b Analyses par image \b0\par")
    for im in per_image_metrics:
        body.append(r"\b Fichier \b0 : "+_escape_rtf(im.get("filename",""))+r"\par")
        body.append(_escape_rtf(f" - Luminosité moyenne : {im.get('luminosity_mean'):.2f}")+r"\par")
        body.append(_escape_rtf(f" - Contraste (std) : {im.get('contrast_std'):.2f}")+r"\par")
        body.append(_escape_rtf(f" - Entropie : {im.get('entropy'):.3f}")+r"\par")
        body.append(_escape_rtf(f" - Fake-RGB : {im.get('fake_rgb')}")+r"\par\par")
    rtf = header + "\n" + "\n".join(body) + "\n}"
    return rtf.encode("utf-8")

def create_zip_with_report(img_paths: List[Path], rtf_bytes: bytes, rtf_name: str="analysis_report.rtf"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in img_paths:
            try:
                zf.write(p, arcname=str(Path("images")/p.name))
            except: pass
        zf.writestr(rtf_name, rtf_bytes)
    zip_buffer.seek(0)
    return zip_buffer

# ---------------- UI ----------------
def run():
    st.markdown("<style>.small-note{font-size:12px;color:#98a1b3}</style>", unsafe_allow_html=True)
    
    colored_header("1. Contexte & aperçu", "Extrait un petit échantillon, métriques d'image, export RTF+ZIP.", color_name="violet-70")
    st.divider()

    # ---------------- Robust dataset discovery ----------------
    detected_root = None
    classes = None

    # 1) Session state
    if "detected_root" in st.session_state:
        detected_root = Path(st.session_state["detected_root"])
        classes = st.session_state.get("classes", None)
        st.caption(f"Racine dataset depuis session : `{detected_root}`")

    # 2) Fallback KaggleHub
    if detected_root is None:
        try:
            import kagglehub
            p = kagglehub.dataset_download(KAGGLE_SLUG)
            if p:
                detected_root = Path(p)
                st.success(f"Racine dataset via KaggleHub : `{detected_root}`")

                # ---------------- Descente automatique ----------------
                subdirs = [d for d in detected_root.iterdir() if d.is_dir()]
                if len(subdirs) == 1 and (subdirs[1] / "images").exists():
                    detected_root = subdirs[1]
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
    if classes and set(classes) <= {"images","masks"}:
        images_folder = detected_root / "images"
        if images_folder.exists():
            classes = sorted([p.name for p in images_folder.iterdir() if p.is_dir() and any(_is_image_file(f) for f in p.iterdir())])
            st.success(f"Classes extraites depuis `{images_folder}` : {classes}")

    if not classes:
        st.error("Aucune classe détectée sous la racine dataset.")
        return

    st.write(f"Classes détectées ({len(classes)}): {classes[:20]}")

    # ---------------- Sample UI ----------------
    colored_header("2. Sélection échantillon", "Choisir une classe et tirer jusqu'à 5 images.", color_name="violet-70")
    col1,col2,col3 = st.columns([3,1,1])
    with col1: choice = st.selectbox("Classe :", options=classes)
    with col2: n = st.number_input("N images (max 5) :", 1,5,5)
    with col3:
        if st.button("Resample / nouvel échantillon"):
            st.session_state.pop("viz_sample", None)

    sample_key = f"{choice}__{n}"
    sample = st.session_state.get("viz_sample", None)
    if sample is None or sample.get("key") != sample_key:
        imgs = sample_images_from_class(detected_root, choice, n)
        st.session_state["viz_sample"] = {"key": sample_key, "images":[str(p) for p in imgs]}
        sample = st.session_state["viz_sample"]

    img_paths = [Path(p) for p in sample.get("images", [])]

    # ---------------- Display & metrics ----------------
    colored_header("3. Aperçu & métriques", "Miniatures + luminosité, contraste, entropie, fake-RGB.", color_name="violet-70")
    if not img_paths:
        st.info("Aucun fichier image pour cette classe.")
    else:
        cols = st.columns(min(3,len(img_paths)))
        per_image_metrics = []
        for idx,p in enumerate(img_paths):
            with cols[idx%len(cols)]:
                try:
                    im = Image.open(p).convert("RGB")
                    im.thumbnail(THUMBNAIL_MAX)
                    st.image(im,use_column_width=True,caption=p.name)
                    metrics = compute_image_metrics(im)
                    per_image_metrics.append({"filename": p.name, **metrics})
                    st.markdown(
                        f"- Luminosité : **{metrics['luminosity_mean']:.2f}**  \n"
                        f"- Contraste (std) : **{metrics['contrast_std']:.2f}**  \n"
                        f"- Entropie : **{metrics['entropy']:.3f}**  \n"
                        f"- Fake-RGB : {metrics['fake_rgb']}"
                    )
                except Exception as e:
                    st.error(f"Erreur ouvrant {p}: {e}")

        # histogrammes
        lums = [m["luminosity_mean"] for m in per_image_metrics]
        stds = [m["contrast_std"] for m in per_image_metrics]
        st.divider()
        st.markdown("### Histogrammes échantillon")
        fig1,ax1 = plt.subplots()
        if lums: ax1.hist(lums,bins=12)
        ax1.set_title("Luminosité moyenne")
        ax1.set_xlabel("Luminosité")
        ax1.set_ylabel("Nb images")
        st.pyplot(fig1)

        fig2,ax2 = plt.subplots()
        if stds: ax2.hist(stds,bins=12)
        ax2.set_title("Contraste (std)")
        ax2.set_xlabel("Contraste")
        ax2.set_ylabel("Nb images")
        st.pyplot(fig2)

    st.divider()

    # ---------------- Interprétation ----------------
    colored_header("4. Interprétation & notes", "Saisis tes interprétations par figure.", color_name="violet-70")
    interpretations = st.text_area("Interprétations pour l'échantillon / figures", height=160)
    st.divider()

    # ---------------- Export RTF + ZIP ----------------
    colored_header("5. Export / Rapport", "Génère .rtf + ZIP des images sélectionnées.", color_name="violet-70")
    meta = {"classe": choice, "n_images": len(img_paths),
            "date": datetime.datetime.now().isoformat(),
            "notes": (interpretations[:500]+"...") if interpretations else ""}
    if st.button("Générer et télécharger RTF + ZIP"):
        if not img_paths:
            st.warning("Rien à exporter.")
        else:
            per_image_metrics = []
            for p in img_paths:
                try:
                    im = Image.open(p).convert("RGB")
                    metrics = compute_image_metrics(im)
                    per_image_metrics.append({"filename": p.name, **metrics})
                except:
                    per_image_metrics.append({"filename": p.name, "luminosity_mean":0.0,"contrast_std":0.0,"entropy":0.0,"fake_rgb":True})
            meta["interpretation_head"] = (interpretations[:1000]+"...") if interpretations else "—"
            rtf = make_report_rtf_bytes(f"Analyse visuelle — classe {choice}", meta, per_image_metrics)
            zip_buf = create_zip_with_report(img_paths, rtf, rtf_name=f"report_{choice}.rtf")
            st.download_button("Télécharger RTF+ZIP", data=zip_buf.getvalue(), file_name=f"analysis_{choice}.zip", mime="application/zip")
            st.success("Archive générée — contient images et rapport.")

    st.divider()
    st.markdown("<div class='small-note'>Astuce : pour un scan complet du dataset et histogrammes globaux, ajouter une action 'Run full dataset scan' (opération lourde).</div>", unsafe_allow_html=True)
