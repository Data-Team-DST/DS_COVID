"""
ImageNormalizer - Normalisation d'images.

Transformateur pour normaliser les pixels entre 0 et 1.
"""

from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import BaseTransform


class ImageNormalizer(BaseTransform):
    """
    Normalise les images pixel-wise entre 0 et 1.
    
    Ce transformateur convertit les pixels d'images (typiquement en [0, 255])
    en valeurs flottantes normalisées entre 0 et 1.
    
    Pattern sklearn: Transformation stateless (sans apprentissage).
    
    Formats d'entrée supportés:
        - Liste d'images (numpy arrays)
        - Numpy array 4D (batch d'images)
        - DataFrame avec colonne 'image_array'
    
    Sortie:
        - Images normalisées avec type float32
        - Même format que l'entrée
    
    Usage:
        normalizer = ImageNormalizer()
        images_norm = normalizer.fit_transform(images)
    """
    
    def _process(self, X: Any) -> Any:
        """
        Normalise les images entre 0 et 1.
        
        Args:
            X: Liste d'images, numpy array ou DataFrame
        
        Returns:
            Images normalisées (float32, valeurs entre 0 et 1)
        """
        # Cas 1: DataFrame
        if isinstance(X, pd.DataFrame):
            if 'image_array' not in X.columns:
                raise ValueError("DataFrame doit contenir une colonne 'image_array'")
            
            self._log(f"Normalisation de {len(X)} images (DataFrame)")
            
            X_transformed = X.copy()
            norm_images = []
            
            for idx, row in tqdm(X.iterrows(), total=len(X),
                                desc=f"[{self.__class__.__name__}] Normalisation",
                                disable=not self.verbose):
                img = row['image_array']
                if img is not None:
                    norm_images.append(self._normalize_image(img))
                else:
                    norm_images.append(None)
            
            X_transformed['image_array'] = norm_images
            
            return X_transformed
        
        # Cas 2: Liste ou numpy array
        else:
            data_array = np.array(X)
            self._log(f"Normalisation de {len(data_array)} images")
            
            # Normalisation directe pour l'ensemble
            X_norm = data_array.astype(np.float32)
            
            # Vérifier si déjà normalisé
            if X_norm.max() > 1.0:
                X_norm = X_norm / 255.0
            
            self._log("Normalisation terminée")
            
            return X_norm
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise une seule image.
        
        Args:
            image: Image à normaliser (numpy array)
        
        Returns:
            Image normalisée (float32, valeurs entre 0 et 1)
        """
        img_norm = image.astype(np.float32)
        
        # Vérifier si déjà normalisé
        if img_norm.max() > 1.0:
            img_norm = img_norm / 255.0
        
        return img_norm
    
    def visualize(self, X_before: Any, X_after: Any, n_samples: int = 3) -> None:
        """Visualise la normalisation avec histogrammes."""
        import matplotlib.pyplot as plt
        
        # Extraire les images
        if isinstance(X_before, pd.DataFrame):
            images_before = [X_before['image_array'].iloc[i] for i in range(min(n_samples, len(X_before)))]
            images_after = [X_after['image_array'].iloc[i] for i in range(min(n_samples, len(X_after)))]
        else:
            images_before = X_before[:n_samples]
            images_after = X_after[:n_samples]
        
        fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 12))
        fig.suptitle('ImageNormalizer: Normalisation 0-1', fontsize=16, fontweight='bold')
        
        for i in range(n_samples):
            # Avant
            axes[0, i].imshow(images_before[i], cmap='gray' if len(images_before[i].shape)==2 else None)
            axes[0, i].set_title(f'Avant: [{images_before[i].min():.3f}, {images_before[i].max():.3f}]')
            axes[0, i].axis('off')
            
            # Après
            axes[1, i].imshow(images_after[i], cmap='gray' if len(images_after[i].shape)==2 else None)
            axes[1, i].set_title(f'Après: [{images_after[i].min():.3f}, {images_after[i].max():.3f}]')
            axes[1, i].axis('off')
            
            # Histogrammes
            axes[2, i].hist(images_before[i].ravel(), bins=50, alpha=0.5, label='Avant', color='blue')
            axes[2, i].hist(images_after[i].ravel(), bins=50, alpha=0.5, label='Après', color='orange')
            axes[2, i].set_xlabel('Intensité')
            axes[2, i].set_ylabel('Fréquence')
            axes[2, i].legend()
            axes[2, i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
