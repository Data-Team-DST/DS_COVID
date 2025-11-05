"""
Frontend Streamlit pour l'inspection des features.
"""

from datetime import datetime

import pandas as pd  # type: ignore

import streamlit as st
from src.features.Inspector.Features_Core import (
    analyze_module_functions,
    get_all_functions_summary,
    get_features_files,
)

# -------------------- Constantes -------------------- #
NO_DOC = "Pas de documentation"


# -------------------- Helper affichage -------------------- #
def _render_function_content(func_info: dict) -> None:
    """Rend le contenu des détails d'une fonction."""
    st.code(f"def {func_info['name']}{func_info['signature']}", language="python")

    # Documentation
    doc = func_info.get("doc", "")
    if doc and doc != NO_DOC:
        color, border = "#f0f2f6", "#1f77b4"
        msg = doc
    else:
        color, border = "#ffe6e6", "#ff6b6b"
        msg = f"⚠️ {NO_DOC}"
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 15px;
            background-color: {color};
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid {border};
        ">
            <em>{msg}</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Paramètres
    if func_info.get("parameters"):
        st.markdown("**📝 Paramètres:**")
        for param in func_info["parameters"]:
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.write(f"**{param['name']}**")
            with col2:
                (
                    st.code(param["annotation"], language="python")
                    if param.get("annotation")
                    else st.write("_Type non spécifié_")
                )
            with col3:
                default = param.get("default")
                st.write(f"Défaut: `{default}`" if default else "_Requis_")


def show_function_details(func_info: dict, use_expander: bool = True) -> None:
    """Affiche les détails d'une fonction, avec ou sans expander."""
    if use_expander:
        with st.expander(f"🔧 {func_info['name']}", expanded=False):
            _render_function_content(func_info)
    else:
        st.markdown(f"### 🔧 {func_info['name']}")
        _render_function_content(func_info)


# -------------------- Fichiers Features -------------------- #
def show_features_files() -> None:
    """Affiche la liste des fichiers Python dans Features avec info."""
    st.subheader("📁 Fichiers dans le dossier Features:")
    files, features_dir, file_info = get_features_files()
    st.info(f"📂 Dossier analysé: `{features_dir}`")

    if not files:
        st.warning(f"⚠️ Aucun fichier Python trouvé dans `{features_dir}`")
        st.info(
            "💡 Suggestions:\n- Vérifiez l'existence du dossier"
            "\n- Vérifiez qu'il contient des `.py`\n- Vérifiez les permissions"
        )
        return

    st.success(f"✅ {len(files)} fichier(s) Python trouvé(s)")
    table_data = []

    for file in files:
        info = file_info.get(file.name, {})
        if "error" not in info:
            try:
                modified_date = datetime.fromtimestamp(info["modified"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                modified_date = "Inconnu"
            table_data.append(
                {
                    "Fichier": file.name,
                    "Taille (KB)": info.get("size_kb", "N/A"),
                    "Lignes": info.get("lines", "N/A"),
                    "Modifié": modified_date,
                    "Statut": "✅ OK" if info.get("exists", False) else "❌ Erreur",
                }
            )
        else:
            table_data.append(
                {
                    "Fichier": file.name,
                    "Taille (KB)": "Erreur",
                    "Lignes": "Erreur",
                    "Modifié": "Erreur",
                    "Statut": f"❌ {info['error']}",
                }
            )

    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    with st.expander("📋 Liste simple des fichiers"):
        for file in files:
            st.write(f"- 📄 {file.name}")


# -------------------- Analyse Functions -------------------- #
def _filter_functions(functions_info: list, filter_option: str) -> list:
    if filter_option == "Documentées uniquement":
        return [f for f in functions_info if f.get("doc", "") != NO_DOC]
    if filter_option == "Non documentées uniquement":
        return [f for f in functions_info if f.get("doc", "") == NO_DOC]
    return functions_info


def _display_functions(filtered_functions: list, display_mode: str) -> None:
    for i, func_info in enumerate(filtered_functions):
        if display_mode == "Liste compacte":
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(f"**🔧 {func_info['name']}**")
            with col2:
                st.write(f"📏 {func_info.get('source_lines', 0)} lignes")
            with col3:
                st.write("📖 ✅" if func_info.get("doc", "") != NO_DOC else "📖 ❌")
            with col4:
                key_show = f"show_details_{func_info['name']}"
                if st.button("👁️ Détails", key=f"details_{i}_{func_info['name']}"):
                    st.session_state[key_show] = True
            if st.session_state.get(key_show, False):
                with st.container():
                    show_function_details(func_info, use_expander=False)
                if st.button("🔼 Masquer", key=f"hide_{i}_{func_info['name']}"):
                    st.session_state[key_show] = False
                    st.rerun()
                st.markdown("---")
        else:
            show_function_details(func_info, use_expander=True)


def _analyze_single_file(file_path) -> None:
    """Analyse et affiche les fonctions d'un seul fichier."""
    try:
        with st.spinner("🔍 Analyse en cours..."):
            functions_info = analyze_module_functions(file_path)
        if not functions_info:
            st.info("ℹ️ Aucune fonction trouvée dans ce fichier")
            return

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔧 Fonctions", len(functions_info))
        with col2:
            st.metric(
                "📏 Lignes totales",
                sum(f.get("source_lines", 0) for f in functions_info),
            )
        with col3:
            documented = sum(1 for f in functions_info if f.get("doc", "") != NO_DOC)
            st.metric("📖 Documentées", f"{documented}/{len(functions_info)}")

        st.markdown("---")
        filter_option = st.radio(
            "🔍 Filtrer les fonctions:",
            ["Toutes", "Documentées uniquement", "Non documentées uniquement"],
            horizontal=True,
        )
        filtered_functions = _filter_functions(functions_info, filter_option)
        display_mode = st.radio(
            "📋 Mode d'affichage:",
            ["Liste compacte", "Détails complets"],
            horizontal=True,
        )
        _display_functions(filtered_functions, display_mode)

    except Exception as exc:
        st.error(f"❌ Erreur lors de l'analyse de {file_path.name}: {str(exc)}")
        with st.expander("🔍 Détails de l'erreur"):
            st.code(str(exc))


def _analyze_all_files() -> None:
    """Analyse toutes les fonctions de tous les fichiers Features."""
    try:
        with st.spinner("🔍 Analyse de tous les fichiers..."):
            summary = get_all_functions_summary()
        if not summary or not summary.get("all_functions"):
            st.info("ℹ️ Aucune fonction trouvée")
            return

        summary_data = [
            {
                "🔧 Fonction": f.get("name", "N/A"),
                "📄 Fichier": f.get("file", "N/A"),
                "📏 Lignes": f.get("source_lines", 0),
                "📖 Documentée": "✅" if f.get("doc", "") != NO_DOC else "❌",
                "📝 Paramètres": len(f.get("parameters", [])),
            }
            for f in summary["all_functions"]
        ]

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔧 Total fonctions", summary.get("total_functions", 0))
        with col2:
            st.metric("📄 Fichiers", summary.get("total_files", 0))
        with col3:
            st.metric("📏 Lignes totales", summary.get("total_lines", 0))
        with col4:
            st.metric(
                "📖 Documentation",
                f"{summary.get('documented_functions', 0)}"
                f" /{summary.get('total_functions', 0)}",
            )

        # Chart
        if len(summary.get("functions_by_file", {})) > 1:
            st.subheader("📊 Répartition des fonctions par fichier")
            chart_df = pd.DataFrame(
                [
                    {"Fichier": fname, "Nombre de fonctions": len(funcs)}
                    for fname, funcs in summary["functions_by_file"].items()
                ]
            )
            st.bar_chart(chart_df.set_index("Fichier"))

    except Exception as exc:
        st.error(f"❌ Erreur lors de l'analyse globale: {str(exc)}")
        with st.expander("🔍 Détails de l'erreur"):
            st.code(str(exc))


def show_features_functions_analysis() -> None:
    """Interface Streamlit pour analyser un fichier ou tous les fichiers."""
    st.header("🔍 Analyse des Fonctions Features")
    try:
        files, _, _ = get_features_files()
        if not files:
            st.warning("⚠️ Aucun fichier à analyser")
            return
        st.success(f"✅ {len(files)} fichier(s) trouvé(s) à analyser")

        selected_file = st.selectbox(
            "📁 Choisir un fichier:", options=[f.name for f in files]
        )
        if selected_file:
            file_path = next((f for f in files if f.name == selected_file), None)
            if file_path:
                st.markdown(f"### 📄 Analyse de `{selected_file}`")
                _analyze_single_file(file_path)

        st.markdown("---")
        if st.checkbox("🌍 Afficher toutes les fonctions de tous les fichiers"):
            st.markdown("## 🌍 Vue d'ensemble")
            _analyze_all_files()

    except Exception as exc:
        st.error(f"❌ Erreur lors de l'initialisation: {str(exc)}")
