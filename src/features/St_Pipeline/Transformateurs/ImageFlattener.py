"""
ImageFlattener - Aplatissement d'images en vecteurs 1D.

Transformateur pour convertir des images 2D/3D en vecteurs 1D.
"""

from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import BaseTransform


class ImageFlattener(BaseTransform):
    """
    Aplatit les images en vecteurs 1D.
    
    Ce transformateur convertit des images 2D ou 3D en vecteurs 1D,
    nécessaire pour l'utilisation avec certains algorithmes ML traditionnels.
    
    Pattern sklearn: Transformation stateless (sans apprentissage).
    
    Formats d'entrée supportés:
        - Numpy array 3D/4D (grayscale/color images)
        - DataFrame avec colonne 'image_array'
    
    Sortie:
        - Numpy array 2D (n_samples, n_features)
        - DataFrame avec colonne 'image_array' mise à jour
    
    Usage:
        flattener = ImageFlattener()
        X_flat = flattener.fit_transform(images)  # Shape: (n_samples, height*width*channels)
    """
    
    def _process(self, X: Any) -> Any:
        """
        Aplatit les images en vecteurs 1D.
        
        Args:
            X: Numpy array ou DataFrame
        
        Returns:
            Images aplaties (2D array ou DataFrame)
        """
        # Cas 1: DataFrame
        if isinstance(X, pd.DataFrame):
            if 'image_array' not in X.columns:
                raise ValueError("DataFrame doit contenir une colonne 'image_array'")
            
            self._log(f"Aplatissement de {len(X)} images (DataFrame)")
            
            X_transformed = X.copy()
            flat_images = []
            
            for idx, row in tqdm(X.iterrows(), total=len(X),
                                desc=f"[{self.__class__.__name__}] Aplatissement",
                                disable=not self.verbose):
                img = row['image_array']
                if img is not None:
                    flat_images.append(img.flatten())
                else:
                    flat_images.append(None)
            
            X_transformed['image_array'] = flat_images
            
            return X_transformed
        
        # Cas 2: Numpy array
        else:
            data_array = np.array(X)
            n_samples = data_array.shape[0]
            
            self._log(f"Aplatissement de {n_samples} images de shape {data_array.shape}")
            
            # Aplatir en conservant la première dimension (n_samples)
            X_flat = []
            for img in tqdm(data_array, desc=f"[{self.__class__.__name__}] Aplatissement",
                           disable=not self.verbose):
                X_flat.append(img.flatten())
            
            X_flat = np.array(X_flat)
            
            self._log(f"Aplatissement terminé. Shape: {X_flat.shape}")
            
            return X_flat
