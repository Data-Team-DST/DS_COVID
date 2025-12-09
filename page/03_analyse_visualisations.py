# Theming metadata:
# - Dark theme inherited.
# - Extras activés : stylable_container, colored_header, badge pour structuration élégante.
# - Page = hub d’analyses visuelles : upload images, URLs, CSV bruts, annotations.

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header
from streamlit_extras.badges import badge

def run():

    colored_header(
        label="1. Topic overview & context",
        description="Résumé narratif + rôle des figures dans l’analyse métier.",
        color_name="violet-70"
    )

    st.markdown("> Ajoutez une courte explication pour chaque figure insérée — ce qu’elle montre et pourquoi elle compte.")
    st.divider()

    # Checklist
    with stylable_container(key="chk_container", css_styles="""
        {
            border: 1px solid #2a2d33;
            padding: 1em;
            border-radius: 12px;
            background-color: #131418;
        }
    """):
        st.markdown("### Checklist visuelle")
        c1, c2, c3 = st.columns(3)
        c1.checkbox("KPIs présents")
        c2.checkbox("Figures importées")
        c3.checkbox("Interprétations ajoutées")

    st.divider()


    colored_header(
        label="2. Data intro (sources, volume, structure)",
        description="Documenter la provenance et le périmètre utilisé.",
        color_name="violet-70"
    )

    st.text_input("Source utilisée (ex: snapshot parquet / table SQL)", key="viz_src_path")
    st.divider()


    # ============================
    # Section 3 — Visualisations
    # ============================

    colored_header(
        label="3. Data analysis & visualizations",
        description="Upload images, URLs, ou CSV brut.",
        color_name="violet-70"
    )

    with stylable_container(
        key="info_box",
        css_styles="""
            {
                background-color: #1e1f24;
                border: 1px solid #2b2c31;
                padding: 0.8rem 1rem;
                border-radius: 10px;
                color: #cfd2d9;
            }
        """
    ):
        st.markdown("Upload multimodal activé : images, URLs, CSV.")


    st.markdown("Interface modulable : choisissez image upload, URL ou CSV pour prévisualiser vos artefacts.")
    st.divider()

    # A — Upload Image
    st.markdown("### A — Upload d'image (PNG/JPG)")
    img_file = st.file_uploader("Déposer image", type=["png", "jpg", "jpeg"])

    if img_file:
        st.image(img_file, use_column_width=True, caption=f"Preview : {img_file.name}")

    st.divider()

    # B — Image via URL
    st.markdown("### B — URL d’image")
    img_url = st.text_input("Entrer URL d’une image")

    if img_url:
        try:
            st.image(img_url, use_column_width=True, caption="Preview via URL")
            st.markdown(f"_Source : {img_url}_")
        except:
            st.error("Erreur lors du chargement depuis l'URL.")

    st.divider()

    # C — Upload CSV brut
    st.markdown("### C — Upload CSV/TSV (aperçu brut)")
    data_file = st.file_uploader("Déposer CSV / TSV", type=["csv", "tsv", "txt"])

    if data_file:
        raw = data_file.getvalue().decode("utf-8", errors="replace")
        head_preview = "\n".join(raw.splitlines()[:40])
        st.text_area("Aperçu brut (head)", value=head_preview, height=240)


    # Snippet
    st.expander("Snippet pandas/plotly (si dispo dans env)").markdown(
        """```python
import pandas as pd
import plotly.express as px

df = pd.read_csv("votre_fichier.csv")
st.dataframe(df.head())
fig = px.line(df, x="date", y="target")
st.plotly_chart(fig)
```"""
    )

    st.divider()

    # ============================
    # Suite du template
    # ============================

    colored_header(
        label="4. Preprocessing per figure",
        description="Documenter filtres, agrégations, fenêtres.",
        color_name="violet-70"
    )
    st.text_area("Préprocessing", height=100)

    st.divider()


    colored_header(
        label="5. Model summary/results",
        description="Associer figures aux artefacts versionnés.",
        color_name="violet-70"
    )
    st.text_input("Artefact modèle (ex : model_v1.2)")

    st.divider()


    colored_header(
        label="6. Best model analysis",
        description="SHAP, PDP, importance features.",
        color_name="violet-70"
    )
    st.text_area("Notes best-model visuals", height=80)

    st.divider()


    colored_header(
        label="7. Conclusions & business relevance",
        description="Relier visuels → décision métier.",
        color_name="violet-70"
    )
    st.text_area("Conclusions par figure", height=100)

    st.divider()


    colored_header(
        label="8. Critique & future perspectives",
        description="Backlog visuel, données manquantes.",
        color_name="violet-70"
    )
    st.text_area("Critique & backlog", height=100)

    st.divider()


    colored_header(
        label="9. CI/CD pipeline overview",
        description="Artefacts, snapshots, reproductibilité.",
        color_name="violet-70"
    )
    st.markdown("- Versionner PNG/HTML.")
    st.markdown("- Stocker notebooks de génération.")

    st.markdown(
        "<small style='color:#98a1b3'>Status : version enrichie — structure avec extras, upload images + CSV OK.</small>",
        unsafe_allow_html=True,
    )
