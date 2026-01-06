"""
Transformateurs pour le chargement de données.

Contient ImagePathLoader et TupleToDataFrame pour charger et convertir
les chemins d'images et leurs métadonnées.
"""

import os
import glob
from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import streamlit as st
    import plotly.express as px
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from .base import BaseTransform


class ImagePathLoader(BaseTransform):
    """
    Charge les chemins des images depuis le système de fichiers.
    
    Ce transformateur parcourt une structure de répertoires pour collecter
    les chemins des images et leurs masques associés, organisés par label.
    
    ⚠️ IMPORTANT: Ce transformateur est STATEFUL pour compatibilité pickle.
    Il charge les paths une seule fois lors du fit(), puis les retourne
    lors du transform().
    
    Structure attendue :
        root_dir/
            ├── COVID/
            │   ├── images/*.png
            │   └── masks/*.png
            ├── Normal/
            │   ├── images/*.png
            │   └── masks/*.png
            └── ...
    
    Usage dans Pipeline:
        loader = ImagePathLoader(root_dir="data/raw/...")
        pipeline = Pipeline([
            ('loader', loader),
            ('converter', TupleToDataFrame()),
            ...
        ])
        pipeline.fit(None)  # Charge les paths
        result = pipeline.transform(None)  # Retourne les données
    """
    
    def __init__(self, root_dir: str, **kwargs):
        """
        Initialise le chargeur de chemins.
        
        Args:
            root_dir: Chemin vers le répertoire racine des données
            **kwargs: Paramètres de BaseTransform (verbose, use_streamlit)
        """
        super().__init__(**kwargs)
        self.root_dir = root_dir
        # État du transformateur (pour pickle)
        self.image_paths_ = None
        self.mask_paths_ = None
        self.labels_ = None
    
    def _fit(self, X: Any, y: Optional[Any] = None) -> None:
        """
        Charge les chemins depuis le filesystem pendant fit().
        
        Pattern sklearn: fit = apprentissage/chargement des métadonnées.
        
        Args:
            X: Ignoré (peut être None)
            y: Ignoré
        """
        self._log(f"Chargement des chemins depuis {self.root_dir}")
        
        image_paths = []
        mask_paths = []
        labels = []
        
        # Liste des labels (répertoires avec sous-dossier 'images')
        labels_list = [
            label for label in os.listdir(self.root_dir) 
            if os.path.isdir(os.path.join(self.root_dir, label, 'images'))
        ]
        
        self._log(f"Labels trouvés : {labels_list}")
        
        # Parcourir chaque label avec progression
        total_labels = len(labels_list)
        if self.use_streamlit and self._progress_bar is not None:
            # Mode Streamlit
            for idx, label in enumerate(labels_list):
                class_img_dir = os.path.join(self.root_dir, label, 'images')
                class_mask_dir = os.path.join(self.root_dir, label, 'masks')
                
                for img_path in glob.glob(os.path.join(class_img_dir, '*.png')):
                    filename = os.path.basename(img_path)
                    mask_path = os.path.join(class_mask_dir, filename)
                    if os.path.exists(mask_path):
                        image_paths.append(img_path)
                        mask_paths.append(mask_path)
                        labels.append(label.lower())
                
                progress = (idx + 1) / total_labels
                self._update_progress(progress, f"Scanné {idx + 1}/{total_labels} labels")
            self._clear_progress()
        else:
            # Mode console avec tqdm
            for label in tqdm(labels_list, desc=f"[{self.__class__.__name__}] Scan labels", 
                             disable=not self.verbose):
                class_img_dir = os.path.join(self.root_dir, label, 'images')
                class_mask_dir = os.path.join(self.root_dir, label, 'masks')
                
                for img_path in glob.glob(os.path.join(class_img_dir, '*.png')):
                    filename = os.path.basename(img_path)
                    mask_path = os.path.join(class_mask_dir, filename)
                    if os.path.exists(mask_path):
                        image_paths.append(img_path)
                        mask_paths.append(mask_path)
                        labels.append(label.lower())
        
        # Stocker l'état (attributs avec underscore = fitted)
        self.image_paths_ = image_paths
        self.mask_paths_ = mask_paths
        self.labels_ = labels
        
        self._log(f"Total chargé : {len(self.image_paths_)} images")
        
        # Afficher la distribution si Streamlit
        if self.use_streamlit:
            self._plot_distribution()
    
    def _process(self, X: Any) -> Tuple[list, list, list]:
        """
        Retourne les chemins chargés lors du fit().
        
        Pattern sklearn: transform = application de la transformation.
        
        Args:
            X: Ignoré
        
        Returns:
            tuple: (image_paths, mask_paths, labels)
        
        Raises:
            RuntimeError: Si fit() n'a pas été appelé
        """
        if self.image_paths_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.fit() doit être appelé avant transform()"
            )
        
        self._log(f"Retour de {len(self.image_paths_)} chemins")
        return (self.image_paths_, self.mask_paths_, self.labels_)
    
    def _plot_distribution(self) -> None:
        """Affiche la distribution des labels (Streamlit uniquement)."""
        if not HAS_STREAMLIT:
            return
            
        try:
            # Compter les labels
            df = pd.DataFrame(self.labels_, columns=['Label'])
            label_counts = df['Label'].value_counts().reset_index()
            label_counts.columns = ['Label', 'Count']
            
            # Palette de couleurs
            unique_labels = label_counts['Label'].unique()
            default_colors = px.colors.qualitative.Plotly
            color_map = {
                label: default_colors[i % len(default_colors)] 
                for i, label in enumerate(unique_labels)
            }
            
            # Graphiques
            bar_fig = px.bar(
                label_counts, x='Label', y='Count', 
                title='Distribution des labels',
                color='Label', color_discrete_map=color_map
            )
            
            pie_fig = px.pie(
                label_counts, names='Label', values='Count', 
                title='Répartition des classes',
                color='Label', color_discrete_map=color_map
            )
            
            # Affichage
            st.subheader("📊 Distribution des données")
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                st.write("**Comptage:**")
                st.dataframe(label_counts, use_container_width=True)
            with col2:
                st.plotly_chart(bar_fig, use_container_width=True)
            with col3:
                st.plotly_chart(pie_fig, use_container_width=True)
        except Exception as e:
            self._log(f"Erreur lors de l'affichage Streamlit: {e}", level="warning")


class TupleToDataFrame(BaseTransform):
    """
    Convertit le tuple (image_paths, mask_paths, labels) en DataFrame.
    
    Ce transformateur prend le tuple retourné par ImagePathLoader
    et le convertit en DataFrame pandas pour faciliter la manipulation.
    
    Usage:
        converter = TupleToDataFrame()
        df = converter.fit_transform((image_paths, mask_paths, labels))
    """
    
    def _process(self, X: Tuple[list, list, list]) -> pd.DataFrame:
        """
        Convertit un tuple en DataFrame.
        
        Args:
            X: Tuple (image_paths, mask_paths, labels)
        
        Returns:
            DataFrame avec colonnes ['image_path', 'mask_path', 'label']
        
        Raises:
            ValueError: Si X n'est pas un tuple de 3 éléments
        """
        if not isinstance(X, tuple) or len(X) != 3:
            raise ValueError(
                f"Attendu un tuple de 3 éléments (image_paths, mask_paths, labels), "
                f"reçu: {type(X)}"
            )
        
        image_paths, mask_paths, labels = X
        
        self._log(f"Conversion de {len(image_paths)} éléments en DataFrame")
        
        df = pd.DataFrame({
            'image_path': image_paths,
            'mask_path': mask_paths,
            'label': labels
        })
        
        self._log(f"DataFrame créé: {len(df)} lignes, {len(df.columns)} colonnes")
        
        return df
