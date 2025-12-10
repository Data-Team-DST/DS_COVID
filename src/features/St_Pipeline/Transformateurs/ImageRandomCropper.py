"""
ImageRandomCropper - Crop aléatoire d'images.

Transformateur pour effectuer un crop aléatoire sur les images.
"""

from typing import Any, Optional
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import BaseTransform


class ImageRandomCropper(BaseTransform):
    """
    Effectue un crop aléatoire sur chaque image.
    
    Ce transformateur extrait une région rectangulaire aléatoire de chaque image.
    Utile pour l'augmentation de données et l'entraînement de modèles.
    
    Pattern sklearn: Transformation stateless avec seed pour reproductibilité.
    
    Usage:
        cropper = ImageRandomCropper(crop_size=(224, 224), seed=42)
        images_cropped = cropper.fit_transform(images)
    """
    
    def __init__(self, crop_size=(224, 224), seed=None, **kwargs):
        """
        Initialise le cropper aléatoire.
        
        Args:
            crop_size: Tuple (height, width) pour la taille du crop
            seed: Graine pour reproductibilité
            **kwargs: Paramètres de BaseTransform (verbose, use_streamlit)
        """
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.seed = seed
        self.rng_ = None
    
    def _fit(self, X: Any, y: Optional[Any] = None) -> None:
        """
        Initialise le générateur de nombres aléatoires.
        
        Args:
            X: Données (unused)
            y: Labels (unused)
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.rng_ = random
    
    def _process(self, X: Any) -> Any:
        """
        Applique le random crop aux images.
        
        Args:
            X: Images à cropper
        
        Returns:
            Images croppées
        """
        if self.rng_ is None:
            self._fit(X)
        
        # Cas 1: DataFrame
        if isinstance(X, pd.DataFrame):
            if 'image_array' not in X.columns:
                raise ValueError("DataFrame doit contenir une colonne 'image_array'")
            
            self._log(f"Random crop de {len(X)} images en {self.crop_size}")
            
            X_transformed = X.copy()
            cropped_images = []
            
            for idx, row in tqdm(X.iterrows(), total=len(X),
                                desc=f"[{self.__class__.__name__}] RandomCrop",
                                disable=not self.verbose):
                img = row['image_array']
                if img is not None:
                    cropped_images.append(self._crop_image(img))
                else:
                    cropped_images.append(None)
            
            X_transformed['image_array'] = cropped_images
            
            return X_transformed
        
        # Cas 2: Liste ou numpy array
        else:
            data_array = np.array(X)
            self._log(f"Random crop de {len(data_array)} images")
            
            cropped = []
            for img in tqdm(data_array, desc=f"[{self.__class__.__name__}] RandomCrop",
                           disable=not self.verbose):
                cropped.append(self._crop_image(img))
            
            result = np.array(cropped)
            self._log(f"Random crop terminé. Shape: {result.shape}")
            
            return result
    
    def _crop_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop une seule image aléatoirement.
        
        Args:
            img: Image à cropper (numpy array)
        
        Returns:
            Image croppée (numpy array)
        """
        h, w = img.shape[:2]
        ch, cw = self.crop_size
        
        # Si l'image est plus petite que le crop, la retourner telle quelle
        if h < ch or w < cw:
            self._log(f"Image trop petite ({h}x{w}) pour crop ({ch}x{cw}), ignoré", level="warning")
            return img
        
        # Position aléatoire du crop
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        
        # Crop
        return img[top:top+ch, left:left+cw]
