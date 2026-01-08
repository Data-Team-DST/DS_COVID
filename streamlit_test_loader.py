"""Application Streamlit standalone pour tester le ImageLoader."""

import random
from pathlib import Path

import streamlit as st

from src.features.Pipelines.transformateurs.image_loaders import ImageLoader


def get_random_images_paths_and_labels(
    dataset_base: Path,
    class_names: list[str] | str,
    num_samples_per_class: int,
    seed: int = 42,
) -> tuple[list[Path], list[str]]:
    """
    Sélectionne aléatoirement des images du dataset avec distribution équilibrée.
    
    Args:
        dataset_base: Chemin racine du dataset
        class_names: Nom(s) de classe(s) ('COVID', 'Normal', etc. ou liste de ceux-ci, ou 'all' pour toutes les classes)
        num_samples_per_class: Nombre d'images à sélectionner par classe (distribution équilibrée)
        seed: Seed pour la génération aléatoire
    
    Returns:
        tuple: (liste des chemins d'images, liste des labels correspondants)
        Les images sont mélangées aléatoirement après sélection
    
    Examples:
        >>> # Une seule classe : 10 images COVID
        >>> paths, labels = get_random_images_paths_and_labels(base, "COVID", 10, seed=42)
        >>> # Plusieurs classes : 10 images de chaque = 20 total
        >>> paths, labels = get_random_images_paths_and_labels(base, ["COVID", "Normal"], 10, seed=42)
        >>> # Toutes les classes : 10 images de chaque = 40 total
        >>> paths, labels = get_random_images_paths_and_labels(base, "all", 10, seed=42)
    """
    
    
    # Normaliser l'entrée : toujours travailler avec une liste
    if isinstance(class_names, str):
        if class_names.lower() == "all":
            class_names = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
        else:
            class_names = [class_names]
    
    # Collecter les images PAR CLASSE pour garantir l'équilibre
    all_selected: list[tuple[Path, str]] = []
    
    for class_name in class_names:
        class_dir = dataset_base / class_name / "images"
        
        if not class_dir.exists():
            st.warning(f"⚠️ Classe '{class_name}' introuvable dans {class_dir}")
            continue
        
        images = list(class_dir.glob("*.png")) # Toutes les images de la classe 
        
        if not images:
            st.warning(f"⚠️ Aucune image trouvée pour la classe '{class_name}'")
            continue
        
        # Si num_samples_per_class est None, prendre TOUTES les images
        if num_samples_per_class is None:
            selected_for_class = images
            st.info(f"📊 Classe '{class_name}': {len(selected_for_class)} images (TOUTES)")
        else:
            # Sélectionner exactement num_samples_per_class images de cette classe
            num_to_sample = min(num_samples_per_class, len(images))
            selected_for_class = random.sample(images, num_to_sample)
            
            if num_to_sample < num_samples_per_class:
                st.warning(
                    f"⚠️ Classe '{class_name}': seulement {num_to_sample}/{num_samples_per_class} images disponibles"
                )
        
        # Ajouter (chemin, classe) pour chaque image sélectionnée
        all_selected.extend([(img, class_name) for img in selected_for_class])
    
    if not all_selected:
        raise ValueError(f"Aucune image trouvée pour les classes : {class_names}")
    
    # Mélanger aléatoirement l'ensemble pour éviter d'avoir toutes les classes groupées
    random.shuffle(all_selected)
    
    # Séparer chemins et classes
    image_paths = [item[0] for item in all_selected]
    image_classes = [item[1] for item in all_selected]
    
    return image_paths, image_classes


def main():
    """Application principale."""
    st.title("🖼️ Test du ImageLoader")
    st.markdown("Testez le chargement d'images avec notre transformer custom")

    # Sidebar : Configuration du loader
    st.sidebar.header("⚙️ Configuration")
    
    color_mode = st.sidebar.selectbox(
        "Mode couleur",
        options=["L", "RGB", "RGBA"],
        index=0,
        help="L = niveaux de gris, RGB = couleur",
    )
    
    validate_paths = st.sidebar.checkbox(
        "Valider les chemins", value=True, help="Vérifie l'existence des fichiers"
    )
    
    fail_on_error = st.sidebar.checkbox(
        "Arrêter sur erreur",
        value=False,
        help="Si coché, stoppe au premier échec",
    )
    
    verbose = st.sidebar.checkbox(
        "Mode verbeux", value=True, help="Affiche les logs détaillés"
    )

    # Sélection du dataset
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Sélection du dataset")
    
    dataset_class = st.sidebar.selectbox(
        "Classe d'images",
        options=["Toutes", "COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
        help="Choisissez la classe d'images à charger",
    )
    
    load_all = st.sidebar.checkbox(
        "🌍 Charger TOUT le dataset",
        value=False,
        help="Si coché, charge toutes les images disponibles (ignorera le slider)",
    )
    
    if load_all:
        nb_images_per_class = None
        if dataset_class == "Toutes":
            st.sidebar.warning("⚠️ Cela chargera les ~21,165 images complètes !")
        else:
            st.sidebar.info(f"ℹ️ Toutes les images de {dataset_class} seront chargées")
    else:
        if dataset_class == "Toutes":
            nb_images_per_class = st.sidebar.slider(
                "Images par classe",
                min_value=1,
                max_value=50,
                value=10,
                help="Nombre d'images à charger PAR CLASSE (4 classes × N images = 4N total)",
            )
            st.sidebar.info(f"📊 Total : {nb_images_per_class * 4} images (4 classes × {nb_images_per_class})")
        else:
            nb_images_per_class = st.sidebar.slider(
                "Nombre d'images",
                min_value=1,
                max_value=50,
                value=10,
                help=f"Nombre d'images de la classe {dataset_class}",
            )
    
    nb_cols = st.sidebar.slider(
        "Nombre de colonnes",
        min_value=1,
        max_value=5,
        value=3,
        help="Nombre de colonnes pour l'affichage",
    )
    
    random_seed = st.sidebar.number_input(
        "Seed aléatoire",
        min_value=0,
        max_value=9999,
        value=42,
        help="Changez pour obtenir d'autres images aléatoires",
    )

    # Chemin vers le dataset
    dataset_base = Path(__file__).parent / "data" / "raw" / "COVID-19_Radiography_Dataset" / "COVID-19_Radiography_Dataset"
    
    st.markdown("## 📦 Dataset COVID-19 Radiography")
    
    if load_all:
        if dataset_class == "Toutes":
            st.warning("⚠️ **Mode COMPLET** : Chargement de toutes les images du dataset (~21,165 images)")
        else:
            st.info(f"**Mode COMPLET** : Chargement de toutes les images de la classe {dataset_class}")
    else:
        if dataset_class == "Toutes":
            total_images = nb_images_per_class * 4
            st.info(f"**Classe sélectionnée** : {dataset_class} | **{nb_images_per_class} images/classe** → **{total_images} images total**")
        else:
            st.info(f"**Classe sélectionnée** : {dataset_class} | **{nb_images_per_class} images**")

    # Bouton de traitement
    if st.button("🚀 Charger les images avec ImageLoader", type="primary"):
        with st.spinner("Chargement en cours..."):
            try:
                # Utiliser la fonction refactorisée avec distribution équilibrée
                if dataset_class == "Toutes":
                    selected_images, image_classes = get_random_images_paths_and_labels(
                        dataset_base, "all", nb_images_per_class, seed=random_seed
                    )
                    # Compter les images par classe pour vérification
                    from collections import Counter
                    class_counts = Counter(image_classes)
                    count_str = ", ".join([f"{cls}: {cnt}" for cls, cnt in sorted(class_counts.items())])
                    st.info(f"📁 {len(selected_images)} images chargées | Distribution: {count_str}")
                else:
                    selected_images, image_classes = get_random_images_paths_and_labels(
                        dataset_base, dataset_class, nb_images_per_class, seed=random_seed
                    )
                    st.info(f"📁 {len(selected_images)} images de la classe {dataset_class}")

                # Initialisation du loader
                loader = ImageLoader(
                    color_mode=color_mode,
                    validate_paths=validate_paths,
                    fail_on_error=fail_on_error,
                    verbose=verbose,
                )

                # Fit (no-op mais respecte l'API sklearn)
                st.write("**Étape 1/2** : Fit du transformer...")
                loader.fit(selected_images)
                
                # Transform : charge les images
                st.write("**Étape 2/2** : Transform (chargement)...")
                loaded_images = loader.transform(selected_images)

                # Résultats
                st.markdown("---")
                st.success("✅ Chargement terminé !")
                
                # Métriques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Images chargées", loader.n_images_loaded_)
                with col2:
                    success_rate = (
                        loader.n_images_loaded_ / len(selected_images)
                    ) * 100
                    st.metric("Taux de succès", f"{success_rate:.1f}%")
                with col3:
                    st.metric("Échecs", len(loader.failed_images_))

                # Affichage des images
                if loaded_images:
                    st.markdown("## 🖼️ Images chargées")
                    
                    # Création de la grille
                    for i in range(0, len(loaded_images), nb_cols):
                        cols = st.columns(nb_cols)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(loaded_images):
                                img = loaded_images[idx]
                                img_class = image_classes[idx]
                                with col:
                                    st.image(
                                        img,
                                        caption=f"{img_class} #{idx+1}\n{img.size[0]}×{img.size[1]} - {img.mode}",
                                        use_container_width=True,
                                    )
                                    
                                    # Détails de l'image
                                    with st.expander("ℹ️ Détails"):
                                        st.write(f"**Classe** : {img_class}")
                                        st.write(f"**Format** : {img.format}")
                                        st.write(f"**Mode** : {img.mode}")
                                        st.write(f"**Taille** : {img.size}")
                                        st.write(f"**Fichier** : {selected_images[idx].name}")
                                        st.write(f"**Chemin** : {selected_images[idx]}")

                # Images échouées
                if loader.failed_images_:
                    st.markdown("---")
                    st.error(f"⚠️ {len(loader.failed_images_)} image(s) échouée(s)")
                    with st.expander("Voir les détails"):
                        for failed_path in loader.failed_images_:
                            st.text(f"❌ {failed_path}")

            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {e}")
                st.exception(e)

    # Documentation
    with st.expander("📚 Documentation du ImageLoader"):
        st.markdown(
            """
        ## 🎯 Fonctionnalités
        
        Le **ImageLoader** est un transformer sklearn qui :
        
        - ✅ Charge des images depuis des chemins de fichiers
        - ✅ Convertit au format PIL Image
        - ✅ Gère la conversion de mode couleur (L, RGB, RGBA)
        - ✅ Valide l'existence des fichiers
        - ✅ Gestion d'erreurs robuste (skip ou raise)
        - ✅ Tracking du taux de succès
        
        ## ⚙️ Paramètres
        
        | Paramètre | Type | Défaut | Description |
        |-----------|------|--------|-------------|
        | `color_mode` | str | 'L' | 'L' (gris), 'RGB' (couleur), 'RGBA' (avec alpha) |
        | `validate_paths` | bool | True | Vérifie l'existence des fichiers avant chargement |
        | `fail_on_error` | bool | False | Stoppe tout si erreur (False = skip les images invalides) |
        | `verbose` | bool | True | Affiche les logs et barres de progression |
        
        ## 🔧 Exemple d'utilisation
        
        ```python
        from src.features.Pipelines.transformateurs.image_loaders import ImageLoader
        
        # Initialisation
        loader = ImageLoader(color_mode='L', validate_paths=True)
        
        # Fit (no-op, mais respecte l'API sklearn)
        loader.fit(image_paths)
        
        # Transform : charge les images
        images = loader.transform(image_paths)
        
        # Vérifier le résultat
        print(f"Images chargées : {loader.n_images_loaded_}")
        print(f"Échecs : {len(loader.failed_images_)}")
        ```
        
        ## 📦 Pipeline sklearn
        
        ```python
        from sklearn.pipeline import Pipeline
        from src.features.Pipelines.transformateurs.image_loaders import ImageLoader
        from src.features.Pipelines.transformateurs.image_preprocessing import (
            ImageResizer, ImageNormalizer
        )
        
        pipeline = Pipeline([
            ('loader', ImageLoader(color_mode='L')),
            ('resizer', ImageResizer(img_size=(256, 256))),
            ('normalizer', ImageNormalizer(method='minmax'))
        ])
        
        # Fit + transform en une fois
        processed_images = pipeline.fit_transform(image_paths)
        ```
        """
        )


if __name__ == "__main__":
    main()
