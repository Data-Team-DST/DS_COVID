"""
RGB_to_L - Conversion RGB vers niveaux de gris.

Transformateur pour convertir des images couleur en niveaux de gris.
"""

from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from .base import BaseTransform


class RGB_to_L(BaseTransform):
    """
    Convertit des images RGB en niveaux de gris (Luminance).
    
    Ce transformateur prend des images couleur (RGB) et les convertit en 
    niveaux de gris en utilisant la conversion standard PIL/Pillow.
    
    Formule de conversion : L = 0.299*R + 0.587*G + 0.114*B
    
    Pattern sklearn: Transformation stateless (sans apprentissage).
    
    Formats d'entrée supportés:
        - DataFrame avec colonne 'image_array' (numpy arrays)
        - Liste d'images (numpy arrays ou PIL Images)
        - Numpy array 4D (batch d'images)
    
    Sortie:
        - DataFrame avec colonne 'image_array' mise à jour
        - Liste d'images en niveaux de gris
        - Numpy array 3D (batch d'images grises)
    
    Usage:
        # Avec DataFrame
        converter = RGB_to_L()
        df_gray = converter.fit_transform(df)
        
        # Avec liste d'images
        images_gray = converter.fit_transform(images_list)
    """
    
    def _process(self, X: Any) -> Any:
        """
        Convertit les images RGB en niveaux de gris.
        
        Args:
            X: DataFrame, liste d'images, ou numpy array
        
        Returns:
            Même format que l'entrée, avec images converties
        
        Raises:
            ValueError: Si le format d'entrée n'est pas supporté
        """
        # Cas 1: DataFrame avec colonne 'image_array'
        if isinstance(X, pd.DataFrame):
            if 'image_array' not in X.columns:
                raise ValueError("DataFrame doit contenir une colonne 'image_array'")
            
            self._log(f"Conversion de {len(X)} images (DataFrame)")
            
            X_transformed = X.copy()
            gray_images = []
            total = len(X)
            
            if self.use_streamlit and self._progress_bar is not None:
                # Mode Streamlit
                for idx, (_, row) in enumerate(X.iterrows()):
                    img = row['image_array']
                    if img is not None:
                        gray_images.append(self._convert_image(img))
                    else:
                        gray_images.append(None)
                    
                    if idx % 100 == 0 or idx == total - 1:
                        progress = (idx + 1) / total
                        self._update_progress(progress, f"Converti {idx + 1}/{total}")
                self._clear_progress()
            else:
                # Mode console avec tqdm
                for idx, row in tqdm(X.iterrows(), total=total,
                                    desc=f"[{self.__class__.__name__}] RGB → L",
                                    disable=not self.verbose):
                    img = row['image_array']
                    if img is not None:
                        gray_images.append(self._convert_image(img))
                    else:
                        gray_images.append(None)
            
            X_transformed['image_array'] = gray_images
            
            # Mettre à jour le nombre de canaux
            if 'channels' in X_transformed.columns:
                X_transformed['channels'] = 1
            
            # Visualisation en mode Streamlit
            if self.use_streamlit:
                self._plot_conversion_effect(X['image_array'].tolist()[:6], gray_images[:6])
            
            return X_transformed
        
        # Cas 2: Liste d'images
        elif isinstance(X, list):
            self._log(f"Conversion de {len(X)} images (liste)")
            
            gray_images = []
            for img in tqdm(X, desc=f"[{self.__class__.__name__}] RGB → L",
                           disable=not self.verbose):
                gray_images.append(self._convert_image(img))
            
            return gray_images
        
        # Cas 3: Numpy array (batch d'images)
        elif isinstance(X, np.ndarray):
            if X.ndim != 4:
                raise ValueError(
                    f"Array doit être 4D (batch, height, width, channels), reçu: {X.shape}"
                )
            
            self._log(f"Conversion de {X.shape[0]} images (numpy array)")
            
            gray_images = []
            for img in tqdm(X, desc=f"[{self.__class__.__name__}] RGB → L",
                           disable=not self.verbose):
                gray_images.append(self._convert_image(img))
            
            return np.array(gray_images)
        
        else:
            raise ValueError(
                f"Format non supporté. Attendu: DataFrame, list ou ndarray, "
                f"reçu: {type(X)}"
            )
    
    def _convert_image(self, image: Any) -> np.ndarray:
        """
        Convertit une seule image en niveaux de gris.
        
        Args:
            image: Image à convertir (numpy array ou PIL Image)
        
        Returns:
            Image en niveaux de gris (numpy array 2D)
        """
        # Si déjà en niveaux de gris (2D)
        if isinstance(image, np.ndarray) and image.ndim == 2:
            return image
        
        # Conversion en PIL Image si nécessaire
        if isinstance(image, np.ndarray):
            # Gérer les images normalisées [0, 1]
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Conversion en niveaux de gris
        l_image = pil_image.convert('L')
        
        return np.array(l_image)
    
    def _plot_conversion_effect(self, images_before, images_after, n_samples=6):
        """Visualise l'effet de la conversion RGB → Grayscale avec Plotly."""
        n = min(n_samples, len(images_before), len(images_after))
        
        if self.use_streamlit and HAS_STREAMLIT:
            st.markdown("### 📊 Conversion RGB → Niveaux de Gris")
            
            # Créer grille de comparaison
            fig = make_subplots(
                rows=2, cols=n,
                row_heights=[0.5, 0.5],
                subplot_titles=[f"RGB #{i+1}" for i in range(n)] + 
                               [f"Grayscale #{i+1}" for i in range(n)],
                specs=[[{'type': 'image'}]*n, [{'type': 'image'}]*n],
                vertical_spacing=0.05,
                horizontal_spacing=0.02
            )
            
            for i in range(n):
                img_before = images_before[i]
                img_after = images_after[i]
                
                # Image RGB (avant)
                if len(img_before.shape) == 2:
                    # Si déjà grayscale, convertir en RGB pour affichage
                    img_rgb = np.stack([img_before]*3, axis=-1)
                else:
                    img_rgb = img_before
                    
                # Normaliser pour affichage si nécessaire
                if img_rgb.max() <= 1.0:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                    
                fig.add_trace(go.Image(z=img_rgb), row=1, col=i+1)
                
                # Image Grayscale (après) - convertir en RGB pour affichage
                if img_after.max() <= 1.0:
                    img_gray_display = (img_after * 255).astype(np.uint8)
                else:
                    img_gray_display = img_after.astype(np.uint8)
                    
                img_gray_rgb = np.stack([img_gray_display]*3, axis=-1)
                fig.add_trace(go.Image(z=img_gray_rgb), row=2, col=i+1)
            
            # Mise en forme
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig.update_layout(
                height=600,
                title_text="Effet de la Conversion RGB → Grayscale",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            st.markdown("**📊 Informations:**")
            col1, col2 = st.columns(2)
            with col1:
                if len(images_before[0].shape) == 3:
                    st.metric("Canaux avant", "3 (RGB)", help="Image couleur")
                else:
                    st.metric("Canaux avant", "1 (déjà Grayscale)", help="Image déjà en niveaux de gris")
            with col2:
                st.metric("Canaux après", "1 (Grayscale)", help="Image en niveaux de gris")
