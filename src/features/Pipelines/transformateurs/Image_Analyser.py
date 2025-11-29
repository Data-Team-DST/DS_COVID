import os
import glob
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from .BaseTransform import BaseTransform


class ImageAnalyser(BaseTransform):
    """
    Transformateur pour charger et analyser les images.
    Fournit des statistiques détaillées sur les images et leurs masques.
    """
    
    def __init__(self, verbose=True, use_streamlit=True, load_images=True, analyze_masks=True):
        """
        Args:
            verbose: Active les logs
            use_streamlit: Utilise Streamlit pour l'affichage
            load_images: Charge les images en mémoire
            analyze_masks: Analyse également les masques
        """
        super().__init__(verbose=verbose, use_streamlit=use_streamlit)
        self.load_images = load_images
        self.analyze_masks = analyze_masks
        self.stats = {}
        
    def fit(self, X=None, y=None):
        """
        Analyse les métadonnées des images sans les charger.
        
        Args:
            X: DataFrame avec colonnes ['image_path', 'mask_path', 'label']
        """
        if self.use_streamlit :
            st.divider()
            st.info("🔄 Analyse des métadonnées des images...")
        if X is None or len(X) == 0:
            self.log("Aucune donnée à analyser.")
            return self
            
        self.log(f"Analyse de {len(X)} images...")
        
        # Statistiques de base
        self.stats['total_images'] = len(X)
        self.stats['labels_distribution'] = X['label'].value_counts().to_dict()
        
        # Analyse des dimensions et propriétés des images
        image_dims = []
        mask_dims = []
        file_sizes = []
        
        for idx, row in self.progress(X.iterrows(), desc="Analyse des images", total=len(X)):
            try:
                # Analyse de l'image
                with Image.open(row['image_path']) as img:
                    image_dims.append(img.size)  # (width, height)
                    file_sizes.append(os.path.getsize(row['image_path']))
                
                # Analyse du masque si demandé
                if self.analyze_masks and 'mask_path' in row and pd.notna(row['mask_path']):
                    with Image.open(row['mask_path']) as mask:
                        mask_dims.append(mask.size)
                        
            except Exception as e:
                self.log(f"Erreur lors de l'analyse de {row['image_path']}: {e}")
        
        # Statistiques des dimensions
        self.stats['image_dimensions'] = Counter(image_dims)
        self.stats['unique_dimensions'] = len(set(image_dims))
        self.stats['most_common_size'] = Counter(image_dims).most_common(1)[0][0] if image_dims else None
        
        if mask_dims:
            self.stats['mask_dimensions'] = Counter(mask_dims)
        
        # Statistiques des tailles de fichiers
        if file_sizes:
            self.stats['avg_file_size_mb'] = np.mean(file_sizes) / (1024 * 1024)
            self.stats['min_file_size_mb'] = np.min(file_sizes) / (1024 * 1024)
            self.stats['max_file_size_mb'] = np.max(file_sizes) / (1024 * 1024)
        else:
            self.stats['avg_file_size_mb'] = 0
            self.stats['min_file_size_mb'] = 0
            self.stats['max_file_size_mb'] = 0
        
        self._display_statistics()
        
        return super().fit(X, y)

    def transform(self, X):
        """
        Charge les images en mémoire et effectue une analyse approfondie.
        
        Args:
            X: DataFrame avec colonnes ['image_path', 'mask_path', 'label']
            
        Returns:
            DataFrame enrichi avec colonnes supplémentaires d'analyse
        """
        if self.use_streamlit :
            st.divider()
            st.info("🔄 Chargement et analyse des images...")
        self.log("Chargement et analyse des images...")
        
        if not self.load_images:
            self.log("Chargement des images désactivé, retour des données originales.")
            return X
        
        # Copie du DataFrame pour ne pas modifier l'original
        X_transformed = X.copy()
        
        # Listes pour stocker les analyses
        images_array = []
        masks_array = []
        mean_intensities = []
        std_intensities = []
        channels_info = []
        
        for idx, row in self.progress(X.iterrows(), desc="Chargement des images", total=len(X)):
            try:
                # Charger l'image
                img = Image.open(row['image_path'])
                img_array = np.array(img)
                images_array.append(img_array)
                
                # Statistiques de l'image
                mean_intensities.append(img_array.mean())
                std_intensities.append(img_array.std())
                channels_info.append(img_array.shape[-1] if len(img_array.shape) == 3 else 1)
                
                # Charger le masque si disponible
                if self.analyze_masks and 'mask_path' in row and pd.notna(row['mask_path']):
                    mask = Image.open(row['mask_path'])
                    masks_array.append(np.array(mask))
                else:
                    masks_array.append(None)
                    
            except Exception as e:
                self.log(f"Erreur lors du chargement de {row['image_path']}: {e}")
                images_array.append(None)
                masks_array.append(None)
                mean_intensities.append(np.nan)
                std_intensities.append(np.nan)
                channels_info.append(np.nan)
        
        # Ajouter les colonnes d'analyse
        X_transformed['image_array'] = images_array
        X_transformed['mask_array'] = masks_array
        X_transformed['mean_intensity'] = mean_intensities
        X_transformed['std_intensity'] = std_intensities
        X_transformed['channels'] = channels_info
        
        # Analyse par label
        self._analyze_by_label(X_transformed)
        
        self.log(f"Transformation terminée: {len(X_transformed)} images chargées.")
        
        return X_transformed
    
    def _display_statistics(self):
        """Affiche les statistiques collectées."""
        if not self.use_streamlit:
            # Affichage console
            print("\n" + "="*50)
            print("STATISTIQUES DES IMAGES")
            print("="*50)
            for key, value in self.stats.items():
                print(f"{key}: {value}")
            return
        
        # Affichage Streamlit
        st.subheader("📊 Statistiques des Images")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", self.stats['total_images'])
        with col2:
            st.metric("Dimensions Uniques", self.stats['unique_dimensions'])
        with col3:
            st.metric("Taille Moyenne", f"{self.stats['avg_file_size_mb']:.2f} MB")
        with col4:
            size = self.stats['most_common_size']
            st.metric("Taille Commune", f"{size[0]}x{size[1]}" if size else "N/A")

    
    def _analyze_by_label(self, X_transformed):
        """Analyse les statistiques par label."""
        if not self.use_streamlit:
            return
        
        st.subheader("🔍 Analyse par Label")
        
        col1,col2 = st.columns([1,2])

       
        with col1:
            # Statistiques d'intensité par label
            intensity_stats = X_transformed.groupby('label').agg({
                'mean_intensity': ['mean', 'std'],
                'std_intensity': ['mean', 'std']
            }).round(2)
            st.write("")
            st.subheader("Statistiques d'Intensité par Label")
            st.space()
            st.dataframe(intensity_stats, width='content')
        
        with col2:
            # Box plot des intensités moyennes par label
            fig = px.box(X_transformed, x='label', y='mean_intensity',
                     title="Distribution de l'Intensité Moyenne par Label",
                     color='label')
            st.plotly_chart(fig, width='content')
    
    def get_statistics(self):
        """Retourne les statistiques calculées."""
        return self.stats

