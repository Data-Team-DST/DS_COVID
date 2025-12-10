# page/02_donnees.py (Kaggle-only, version réparée)
# Présentation des données + import KaggleHub (Kaggle public) + preview images/masks + ZIP sample
# Minimal, fiable — section 3 réécrite à partir de ton streamlit_app fonctionnel.

import io
import random
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import os

import streamlit as st
from PIL import Image

# kagglehub (utilisé pour télécharger les datasets publics Kaggle)
try:
    import kagglehub
except Exception:
    kagglehub = None

# streamlit-extras (optionnel)
try:
    from streamlit_extras.colored_header import colored_header
except Exception:
    colored_header = None
try:
    from streamlit_extras.badges import badge
except Exception:
    badge = None

# ---------------- CONFIG (Kaggle) ----------------
DATASET_SLUG = "tawsifurrahman/covid19-radiography-database"  # dataset Kaggle public
DEFAULT_DATASET_NAME = "COVID-19_Radiography_Dataset (Kaggle)"
DEFAULT_CLASS_COUNTS = {"COVID-19": 3616, "Normal": 10192, "Viral Pneumonia": 1345, "Lung Opacity": 6012}
DEFAULT_TOTAL = sum(DEFAULT_CLASS_COUNTS.values())
DATASET_DIR = Path("dataset")
N_PER_CLASS_DEFAULT = 6
THUMBNAIL_MAX = (512, 512)

# ---------------- UI CSS ----------------
_CSS = """
<style>
.section-card { background: linear-gradient(90deg, rgba(12,18,30,0.95), rgba(8,12,20,0.95)); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); color:#cfe8ff; margin-bottom:12px; }
.label { font-weight:700; color:#cfe8ff; margin-bottom:6px; }
.card { background:#131416; padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.04); width:100%; max-width:260px; box-shadow:0 6px 14px rgba(0,0,0,0.35); margin-bottom:8px; }
.kv { font-size:12px; color:#98a1b3; }
.small-note { font-size:12px; color:#98a1b3; }
</style>
"""

def _render_section(title: str, body: str):
    st.markdown(f"<div class='section-card'><div class='label'>{title}</div><div>{body}</div></div>", unsafe_allow_html=True)

# ---------------- Helpers ----------------
@st.cache_resource
def get_kaggle_dataset_path(dataset_slug: str) -> Optional[Path]:
    """Télécharge (ou récupère depuis cache) le dataset Kaggle public via kagglehub.
    Retourne le chemin local (Path) vers la racine du dataset (versions/x) ou None si erreur.
    """
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

def find_classes(root: Path) -> List[str]:
    """Détecte les classes — prend les sous-dossiers directs. 
    Si aucun sous-dossier direct, tente un niveau descendant (cas double-name extrait par KaggleHub)."""
    if not root or not root.exists():
        return []
    subs = safe_listdir_dirs(root)
    if subs:
        if len(subs) == 1 and subs[0].name == root.name:
            subs2 = safe_listdir_dirs(subs[0])
            if subs2:
                return [p.name for p in subs2]
        return [p.name for p in subs]
    return []

def build_sample_map_simple(root: Path, targets: List[str], n: int, include_masks: bool) -> Dict[str, List[Dict[str, Optional[str]]]]:
    rng = random.Random(42)
    sample_map: Dict[str, List[Dict[str, Optional[str]]]] = {}
    for cls in targets:
        cls_dir = root / cls
        images_dir = cls_dir / "images"
        if not images_dir.exists():
            images_dir = cls_dir
        imgs = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")]
        if not imgs:
            sample_map[cls] = []
            continue
        k = min(n, len(imgs))
        chosen = rng.sample(imgs, k=k) if len(imgs) > k else imgs[:k]
        entries = []
        for img in chosen:
            ent = {"image": str(img)}
            ent["mask"] = None
            if include_masks:
                maybe_mask = cls_dir / "masks" / img.name
                if maybe_mask.exists():
                    ent["mask"] = str(maybe_mask)
            entries.append(ent)
        sample_map[cls] = entries
    return sample_map

def create_zip_bytes_from_sample(sample_map: Dict[str, List[Dict[str, Optional[str]]]]) -> io.BytesIO:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls_name, items in sample_map.items():
            for entry in items:
                imgp = Path(entry.get("image"))
                if imgp.exists():
                    zf.write(imgp, arcname=f"{cls_name}/images/{imgp.name}")
                m = entry.get("mask")
                if m:
                    mp = Path(m)
                    if mp.exists():
                        zf.write(mp, arcname=f"{cls_name}/masks/{mp.name}")
    zip_buffer.seek(0)
    return zip_buffer

# ---------------- UI ----------------
def run():
    st.markdown(_CSS, unsafe_allow_html=True)

    # Header
    if colored_header:
        try:
            colored_header(label="📦 Présentation des données", description="Volumétrie & aperçu d'échantillons (Kaggle).", color_name="blue-70")
        except Exception:
            st.markdown("### 📦 Présentation des données")
    else:
        st.markdown("### 📦 Présentation des données")
    st.divider()

    # 1. Rôle & périmètre
    st.markdown("## 1. Rôle des données & périmètre")
    st.markdown(
        "Dataset primaire (pré-rempli) : **COVID-19_Radiography_Dataset** — réunit les dossiers :\n\n"
        "- `COVID` / `Normal` / `Viral Pneumonia` / `Lung_Opacity` chacun pouvant contenir `images/` et `masks/`.\n\n"
        "Ce dataset sert pour démonstration du POC."
    )
    st.divider()

    # 2. Inventaire & volumétrie (résumé)
    st.markdown("## 2. Inventaire & volumétrie")
    _render_section("Inventaire (synthétique)", f"<strong>Dataset</strong> : {DEFAULT_DATASET_NAME} (Kaggle)<br><strong>Usage</strong> : classification COVID vs non-COVID.")
    st.markdown("### Répartition (référence)")
    table_md = "| Classe | Images |\n|---:|---:|\n"
    for k, v in DEFAULT_CLASS_COUNTS.items():
        table_md += f"| {k} | {v} |\n"
    table_md += f"| **Total (ref.)** | **{DEFAULT_TOTAL}** |\n"
    st.markdown(table_md)
    st.divider()

    #---------------- 3. Import & aperçu rapide (Kaggle) ----------------
    st.markdown("## 3. Import & aperçu rapide (Kaggle)")
    st.info(f"Utilisation automatique du dataset Kaggle : **{DATASET_SLUG}**")
    st.divider()

    # ---- heuristique pour détecter root et classes ----
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
                if not p.is_dir():
                    continue
                if len(p.relative_to(base).parts) != depth:
                    continue
                subs = [c for c in p.iterdir() if c.is_dir()]
                n_good = sum(1 for c in subs if looks_like_images(c) or (c / "images").exists())
                if n_good >= 2:
                    return p
        candidate = base / base.name
        if candidate.exists() and candidate.is_dir():
            return candidate
        return base

    # ---- récupération dataset Kaggle ----
    @st.cache_resource
    def get_dataset_path(slug: str):
        return Path(kagglehub.dataset_download(slug))

    try:
        dataset_root = get_dataset_path(DATASET_SLUG)
    except Exception as e:
        st.error(f"Impossible de récupérer le dataset Kaggle : {e}")
        st.stop()

    detected_root = find_dataset_root(dataset_root)
    st.write(f"Racine utilisée pour les classes : `{detected_root}`")

    # ---- construction liste classes ----
    classes = []
    for p in sorted(detected_root.iterdir()):
        if p.is_dir() and looks_like_images(p):
            classes.append(p.name)
        elif (p / "images").exists():
            classes.append(p.name)

    if not classes:
        st.error("Aucune classe détectée. Vérifie l'arborescence.")
        st.stop()

    classes.sort()
    st.write(f"Classes détectées : {classes}")

    # ---- UI controls pour build sample_map (section 3 ne fait que préparer, pas afficher) ----
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        choice = st.selectbox("Choisir la classe :", ["all"] + classes, index=0, key="class_select")
    with col2:
        n = st.number_input("Nombre d'images / classe :", min_value=1, max_value=200,
                            value=st.session_state.get("n_per_class", N_PER_CLASS_DEFAULT),
                            step=1, key="n_per_class")
    with col3:
        include_masks = st.checkbox("Inclure masks (préparation seulement)",
                                    value=st.session_state.get("include_masks", False),
                                    key="include_masks")

    if st.button("Préparer échantillon (section 3)"):
        targets = classes if choice == "all" else [choice]
        sample_map = build_sample_map_simple(detected_root, targets, n, include_masks)
        st.session_state["sample_map"] = sample_map
        st.success(f"Échantillon préparé pour {len(sample_map)} classes (section 3).")

    st.divider()

    # ---------------- 4. Aperçu échantillons (UI depuis session_state) ----------------
    st.markdown("## 4. Aperçu échantillons")
    sample_map = st.session_state.get("sample_map", {})
    if not sample_map:
        st.info("Aucun échantillon chargé — clique sur 'Charger échantillon' pour afficher des images.")
    else:
        total = 0
        for cls_name, entries in sample_map.items():
            st.markdown(f"### {cls_name} — {len(entries)} exemples")
            if not entries:
                st.write("— aucun fichier image disponible pour cette classe —")
                continue
            cols = st.columns(3)
            for idx, entry in enumerate(entries):
                with cols[idx % 3]:
                    img_path = Path(entry.get("image"))
                    if img_path.exists():
                        try:
                            im = Image.open(img_path).convert("RGB")
                            im.thumbnail(THUMBNAIL_MAX)
                            st.image(im, caption=img_path.name, width="stretch")
                        except Exception as e:
                            st.write(f"Erreur lecture image: {e}")
                    else:
                        st.write("Image manquante.")
                    mask_path = entry.get("mask")
                    if include_masks and mask_path:
                        mp = Path(mask_path)
                        if mp.exists():
                            try:
                                m_im = Image.open(mp)
                                m_im.thumbnail(THUMBNAIL_MAX)
                                st.image(m_im, caption=f"mask: {mp.name}", width="stretch")
                            except Exception:
                                st.write("Erreur lecture mask.")
                        else:
                            st.write("— mask introuvable —")
                total += 1
        st.success(f"{total} images affichées.")
    st.divider()

    # ---------------- 5. ZIP generation & download ----------------
    st.markdown("## 5. Télécharger l'échantillon (ZIP)")
    if st.button("Générer ZIP de l'échantillon visible"):
        sample_map = st.session_state.get("sample_map", {})
        if not sample_map:
            st.warning("Aucun échantillon à zipper. Charge d'abord un échantillon.")
        else:
            try:
                with st.spinner("Construction du ZIP en mémoire..."):
                    zip_buf = create_zip_bytes_from_sample(sample_map)
                    data_bytes = zip_buf.getvalue()
                st.success("ZIP prêt.")
                st.download_button(label="Télécharger le ZIP", data=data_bytes, file_name="sample_images_masks.zip", mime="application/zip")
            except Exception as e:
                st.error(f"Erreur création ZIP : {e}")
    st.divider()

    # ---------------- 6/7/8/9 — rest (conservés mais minimal) ----------------
    st.markdown("## 6. Jeux pour modélisation & logique de split")
    st.text_area("Logique de split & justification (chronological / patient-level)", value=st.session_state.get("data_split_logic", "Chronological / patient-level"), height=100, key="data_split_logic")
    st.divider()

    st.markdown("## 7. Contraintes & risques (détaillés)")
    st.markdown(
        "- **Biais d'échantillonnage** : sources multiples → répartition non homogène.\n"
        "- **Hétérogénéité des annotations** : labels provenant de différentes sources.\n"
        "- **Qualité d'image** : résolutions et artefacts variables.\n"
        "- **Conformité & anonymisation** : vérifier métadonnées DICOM / PII."
    )
    st.text_area("Notes contraintes / risques (à compléter)", value=st.session_state.get("data_constraints", ""), height=120, key="data_constraints")
    st.divider()

    st.markdown("## 8. Artefacts recommandés & intégration CI")
    st.text_area("Artefacts & jobs CI", value=st.session_state.get("data_artifacts", "schema.json\nsample_anonymized.csv\ndata_report.html"), height=80, key="data_artifacts")
    st.divider()

    st.markdown("## 9. Résumé & prochaines actions")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.text_input("Action 1 (haute)", value=st.session_state.get("data_next_1", "Fournir snapshot DVC"), key="data_next_1")
    with a2:
        st.text_input("Action 2 (moyenne)", value=st.session_state.get("data_next_2", "Documenter dictionnaire"), key="data_next_2")
    with a3:
        st.text_input("Action 3 (basse)", value=st.session_state.get("data_next_3", "Automatiser QA en CI"), key="data_next_3")

    st.markdown("<small class='small-note'>Status: remplis les champs et fournis un échantillon pour rendre la section reproductible.</small>", unsafe_allow_html=True)
