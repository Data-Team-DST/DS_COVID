"""
ImageStandardScaler - Standardisation d'images.

Transformateur pour standardiser les images pixel-wise (mean=0, std=1).
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseTransform


class ImageStandardScaler(BaseTransform):
    """
    Applique un StandardScaler pixel-wise sur les images.
    
    Ce transformateur standardise les images en soustrayant la moyenne
    et en divisant par l'écart-type pour chaque pixel.
    
    Pattern sklearn: Transformation stateful (le scaler doit être fit).
    
    Usage:
        scaler = ImageStandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    """
    
    def __init__(self, **kwargs):
        """
        Initialise le StandardScaler.
        
        Args:
            **kwargs: Paramètres de BaseTransform (verbose, use_streamlit)
        """
        super().__init__(**kwargs)
        self.scaler = StandardScaler()
        self.original_shape_ = None
    
    def _fit(self, X: Any, y: Optional[Any] = None) -> None:
        """
        Apprend les statistiques (moyenne, std) sur les données.
        
        Args:
            X: Images d'entraînement
            y: Labels (unused)
        """
        # Préparer les données
        X_flat, original_shape = self._prepare_data(X, return_shape=True)
        self.original_shape_ = original_shape
        
        self._log(f"Apprentissage StandardScaler sur {X_flat.shape}")
        
        # Fit scaler
        self.scaler.fit(X_flat)
        
        self._log("StandardScaler fitted")
    
    def _process(self, X: Any) -> Any:
        """
        Applique la standardisation aux images.
        
        Args:
            X: Images à transformer
        
        Returns:
            Images standardisées (même format que l'entrée)
        """
        # Préparer les données
        X_flat, original_shape = self._prepare_data(X, return_shape=True)
        
        self._log(f"Standardisation de {X_flat.shape}")
        
        # Transform
        X_scaled = self.scaler.transform(X_flat)
        
        # Reshape si numpy array (pas DataFrame)
        if not isinstance(X, pd.DataFrame):
            X_scaled = X_scaled.reshape(original_shape)
        
        self._log(f"Standardisation terminée. Shape: {X_scaled.shape}")
        
        return X_scaled
    
    def _prepare_data(self, X: Any, return_shape: bool = False):
        """
        Prépare les données pour StandardScaler (aplatit les images).
        
        Args:
            X: Images (numpy array, liste ou DataFrame)
            return_shape: Si True, retourne aussi la forme originale
        
        Returns:
            Numpy array 2D (n_samples, n_features) et optionnellement la forme
        """
        # Cas 1: DataFrame
        if isinstance(X, pd.DataFrame):
            if 'image_array' not in X.columns:
                raise ValueError("DataFrame doit contenir une colonne 'image_array'")
            
            X_flat = []
            for idx, row in X.iterrows():
                img = row['image_array']
                if img is not None:
                    X_flat.append(img.flatten())
                else:
                    raise ValueError(f"Image None à l'index {idx}")
            
            X_flat = np.array(X_flat)
            
            if return_shape:
                return X_flat, X_flat.shape
            return X_flat
        
        # Cas 2: Numpy array ou liste
        else:
            data_array = np.array(X)
            original_shape = data_array.shape
            n_samples = data_array.shape[0]
            X_flat = data_array.reshape(n_samples, -1)
            
            if return_shape:
                return X_flat, original_shape
            return X_flat
