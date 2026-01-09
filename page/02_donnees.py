# 02_donnees.py — version améliorée : UI harmonisée, preview + ZIP, robust
import io, random, zipfile
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
from PIL import Image
import html

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

DEFAULT_CLASS_COUNTS = {"COVID-19": 3616, "Normal": 10192, "Viral Pneumonia": 1345, "Lung Opacity": 6012}
DEFAULT_TOTAL = sum(DEFAULT_CLASS_COUNTS.values())

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

def safe_listdir_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])

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

def find_dataset_root(base: Path):
    children = [p for p in base.iterdir() if p.is_dir()]
    if children:
        n_good = sum(1 for c in children if looks_like_images(c) or (c / "images").exists())
        if n_good >= 2:
            return base
    for depth in range(1, 4):
        for p in base.rglob("*"):
            if not p.is_dir(): continue
            if len(p.relative_to(base).parts) != depth: continue
            subs = [c for c in p.iterdir() if c.is_dir()]
            n_good = sum(1 for c in subs if looks_like_images(c) or (c / "images").exists())
            if n_good >= 2:
                return p
    candidate = base / base.name
    return candidate if candidate.exists() else base

def build_sample_map_simple(root: Path, targets: List[str], n: int, include_masks: bool):
    rng = random.Random(42)
    sample_map: Dict[str, List[dict]] = {}
    for cls in targets:
        cls_dir = root / cls
        images_dir = cls_dir / "images"
        if not images_dir.exists():
            images_dir = cls_dir
        imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}]
        chosen = rng.sample(imgs, k=min(n,len(imgs))) if imgs else []
        entries = [{"image": str(img), "mask": str(cls_dir/"masks"/img.name) if include_masks and (cls_dir/"masks"/img.name).exists() else None} for img in chosen]
        sample_map[cls] = entries
    return sample_map

def create_zip_bytes_from_sample(sample_map: Dict[str,List[dict]]) -> io.BytesIO:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls_name, items in sample_map.items():
            for entry in items:
                imgp = Path(entry.get("image"))
                if imgp.exists():
                    zf.write(imgp, arcname=f"{cls_name}/images/{imgp.name}")
                m = entry.get("mask")
                if m and Path(m).exists():
                    zf.write(m, arcname=f"{cls_name}/masks/{Path(m).name}")
    zip_buffer.seek(0)
    return zip_buffer

# ---------------- UI ----------------
def run():
    st.markdown(_CSS, unsafe_allow_html=True)
    
    # Header narratif développé
    header_text = (
        "Le dataset COVID-19 Radiography Database rassemble plusieurs milliers d'images de radiographies thoraciques, "
        "classées par type de pathologie : COVID-19, Normal, Viral Pneumonia et Lung Opacity. "
        "Ces images permettent d'illustrer les capacités d'analyse et de modélisation dans le cadre d'un POC. "
        "Chaque image est potentiellement accompagnée d'un masque de segmentation, utile pour les modèles supervisés. "
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

    # 1. Rôle & périmètre
    _render_section(
        "1. Rôle des données & périmètre",
        "Le dataset COVID-19 Radiography Dataset sert pour démonstration du POC et la validation de modèles ML/DL. "
        "Il inclut des images thoraciques classées (COVID-19, Normal, Viral Pneumonia, Lung Opacity) ainsi que les masques correspondants lorsque disponibles. "
        "Cette section permet de visualiser le volume de données et leur organisation, tout en garantissant un accès reproductible aux échantillons. "
        "Le dataset est pré-traité pour uniformiser les formats et préparer les jeux d'entrainement/validation/test."
    )

    # 2. Inventaire & volumétrie
    _render_section(
        "2. Inventaire & volumétrie",
        f"Dataset : {DATASET_SLUG}\nTotal images référencées : {DEFAULT_TOTAL}\n"
        "Les images sont réparties selon les classes suivantes, permettant une visualisation claire de la disponibilité des données par catégorie :"
    )
    table_md = "| Classe | Images |\n|---:|---:|\n"
    for k,v in DEFAULT_CLASS_COUNTS.items(): table_md += f"| {k} | {v} |\n"
    st.markdown(table_md)
    
    # 3. Import & aperçu rapide
    st.markdown("## 3. Import & aperçu rapide (Kaggle)")
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

    # UI controls
    col1, col2, col3 = st.columns([2,1,1])
    with col1: choice = st.selectbox("Classe :", ["all"]+classes, index=0)
    with col2: n = st.number_input("N images / classe :", 1,200,N_PER_CLASS_DEFAULT)
    with col3: include_masks = st.checkbox("Inclure masks", value=False)

    if st.button("Préparer échantillon"):
        targets = classes if choice=="all" else [choice]
        sample_map = build_sample_map_simple(detected_root, targets, n, include_masks)
        st.session_state["sample_map"] = sample_map
        st.success(f"Échantillon préparé ({len(sample_map)} classes).")
    st.divider()

    # 4. Aperçu échantillons
    st.markdown("## 4. Aperçu échantillons")
    sample_map = st.session_state.get("sample_map", {})
    if not sample_map: st.info("Clique sur 'Préparer échantillon' pour voir les images.")
    else:
        total = 0
        for cls_name, entries in sample_map.items():
            st.markdown(f"### {cls_name} — {len(entries)} exemples")
            if not entries: st.write("— aucun fichier —"); continue
            cols = st.columns(3)
            for idx, entry in enumerate(entries):
                with cols[idx%3]:
                    img_path = Path(entry["image"])
                    if img_path.exists():
                        im = Image.open(img_path).convert("RGB"); im.thumbnail(THUMBNAIL_MAX)
                        st.image(im, caption=img_path.name, use_column_width=True)
                    mask_path = entry.get("mask")
                    if include_masks and mask_path and Path(mask_path).exists():
                        m_im = Image.open(mask_path); m_im.thumbnail(THUMBNAIL_MAX)
                        st.image(m_im, caption=f"mask: {Path(mask_path).name}", use_column_width=True)
                    total += 1
        st.success(f"{total} images affichées.")
    st.divider()

