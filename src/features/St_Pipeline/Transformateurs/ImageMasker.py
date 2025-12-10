"""
ImageMasker - Application de masques binaires.

Transformateur pour appliquer des masques aux images pour isoler les régions d'intérêt.
"""

from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from .base import BaseTransform


class ImageMasker(BaseTransform):
    """
    Applique des masques binaires aux images.
    
    Ce transformateur multiplie les images par leurs masques associés pour
    isoler les régions d'intérêt (ROI).
    
    Pattern sklearn: Transformation stateful (les mask_paths sont stockés).
    
    Formats d'entrée supportés:
        - Liste d'images avec mask_paths en paramètre
        - Numpy array avec mask_paths en paramètre
        - DataFrame avec colonnes 'image_array' et 'mask_path'
    
    Usage:
        masker = ImageMasker(mask_paths=mask_list)
        images_masked = masker.fit_transform(images)
        
        # Ou avec DataFrame
        masker = ImageMasker()
        df_masked = masker.fit_transform(df)  # utilise df['mask_path']
    """
    
    def __init__(self, mask_paths=None, **kwargs):
        """
        Initialise le masqueur d'images.
        
        Args:
            mask_paths: Liste des chemins vers les masques (optionnel si DataFrame)
            **kwargs: Paramètres de BaseTransform (verbose, use_streamlit)
        """
        super().__init__(**kwargs)
        self.mask_paths = mask_paths
    
    def _process(self, X: Any) -> Any:
        """
        Applique les masques aux images.
        
        Args:
            X: Liste d'images, numpy array ou DataFrame
        
        Returns:
            Images masquées dans le même format
        
        Raises:
            ValueError: Si mask_paths n'est pas fourni et DataFrame sans 'mask_path'
        """
        # Cas 1: DataFrame avec colonnes 'image_array' et 'mask_path'
        if isinstance(X, pd.DataFrame):
            if 'image_array' not in X.columns or 'mask_path' not in X.columns:
                raise ValueError("DataFrame doit contenir 'image_array' et 'mask_path'")
            
            self._log(f"Application des masques sur {len(X)} images (DataFrame)")
            
            X_transformed = X.copy()
            masked_images = []
            
            for idx, row in tqdm(X.iterrows(), total=len(X),
                                desc=f"[{self.__class__.__name__}] Masquage",
                                disable=not self.verbose):
                img = row['image_array']
                mask_path = row['mask_path']
                
                if img is not None and pd.notna(mask_path):
                    masked_images.append(self._apply_mask(img, mask_path))
                else:
                    masked_images.append(img)
            
            X_transformed['image_array'] = masked_images
            
            return X_transformed
        
        # Cas 2: Liste ou numpy array avec mask_paths fourni
        else:
            if self.mask_paths is None:
                raise ValueError("mask_paths doit être fourni pour les listes/arrays")
            
            data_array = np.array(X)
            
            if len(data_array) != len(self.mask_paths):
                raise ValueError(
                    f"Nombre d'images ({len(data_array)}) != nombre de masques ({len(self.mask_paths)})"
                )
            
            self._log(f"Application des masques sur {len(data_array)} images")
            
            masked = []
            iterator = zip(data_array, self.mask_paths)
            
            for img, mask_path in tqdm(iterator, total=len(data_array),
                                      desc=f"[{self.__class__.__name__}] Masquage",
                                      disable=not self.verbose):
                masked.append(self._apply_mask(img, mask_path))
            
            return np.array(masked)
    
    def _apply_mask(self, image: np.ndarray, mask_path: str) -> np.ndarray:
        """
        Applique un masque à une image.
        
        Args:
            image: Image à masquer (numpy array)
            mask_path: Chemin vers le masque
        
        Returns:
            Image masquée (numpy array)
        """
        # Charger le masque
        mask = Image.open(mask_path).convert('L')
        
        # Redimensionner le masque à la taille de l'image
        if len(image.shape) == 3:
            mask = mask.resize((image.shape[1], image.shape[0]))
        else:
            mask = mask.resize((image.shape[1], image.shape[0]))
        
        # Convertir en array binaire
        mask_arr = np.array(mask) > 0
        
        # Appliquer le masque
        if len(image.shape) == 3:
            # Image couleur: élargir le masque
            mask_arr = mask_arr[:, :, np.newaxis]
        
        return image * mask_arr
