"""
ImageHistogram - Extraction d'histogrammes d'intensité.

Transformateur pour calculer les histogrammes d'intensité des images.
"""

from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import BaseTransform


class ImageHistogram(BaseTransform):
    """
    Calcule l'histogramme d'intensité pour chaque image.
    
    Ce transformateur extrait la distribution des intensités de pixels
    sous forme d'histogramme, créant ainsi un vecteur de features.
    
    Pattern sklearn: Transformation stateless (sans apprentissage).
    
    Usage:
        histogram = ImageHistogram(bins=32)
        features = histogram.fit_transform(images)  # Shape: (n_samples, bins)
    """
    
    def __init__(self, bins=32, **kwargs):
        """
        Initialise l'extracteur d'histogrammes.
        
        Args:
            bins: Nombre de bins pour l'histogramme
            **kwargs: Paramètres de BaseTransform (verbose, use_streamlit)
        """
        super().__init__(**kwargs)
        self.bins = bins
    
    def _process(self, X: Any) -> Any:
        """
        Calcule les histogrammes des images.
        
        Args:
            X: Images (numpy array, liste ou DataFrame)
        
        Returns:
            Numpy array 2D (n_samples, bins) avec les histogrammes
        """
        # Cas 1: DataFrame
        if isinstance(X, pd.DataFrame):
            if 'image_array' not in X.columns:
                raise ValueError("DataFrame doit contenir une colonne 'image_array'")
            
            self._log(f"Calcul des histogrammes ({self.bins} bins) pour {len(X)} images")
            
            histos = []
            for idx, row in tqdm(X.iterrows(), total=len(X),
                                desc=f"[{self.__class__.__name__}] Histogrammes",
                                disable=not self.verbose):
                img = row['image_array']
                if img is not None:
                    histo = np.histogram(img.flatten(), bins=self.bins, range=(0, 1))[0]
                    histos.append(histo)
                else:
                    histos.append(np.zeros(self.bins))
            
            return np.array(histos)
        
        # Cas 2: Numpy array ou liste
        else:
            data_array = np.array(X)
            self._log(f"Calcul des histogrammes ({self.bins} bins)")
            
            histos = []
            for img in tqdm(data_array, desc=f"[{self.__class__.__name__}] Histogrammes",
                           disable=not self.verbose):
                histo = np.histogram(img.flatten(), bins=self.bins, range=(0, 1))[0]
                histos.append(histo)
            
            return np.array(histos)
    
    def visualize(self, X_before: Any, X_after: Any, n_samples: int = 3) -> None:
        """Visualise les histogrammes extraits."""
        import matplotlib.pyplot as plt
        
        # Extraire les images originales
        if isinstance(X_before, pd.DataFrame):
            images = [X_before['image_array'].iloc[i] for i in range(min(n_samples, len(X_before)))]
        else:
            images = X_before[:n_samples]
        
        # X_after contient les histogrammes (n_samples, bins)
        histograms = X_after[:n_samples]
        
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
        fig.suptitle(f'ImageHistogram: Extraction de features ({self.bins} bins)', fontsize=16, fontweight='bold')
        
        for i in range(n_samples):
            # Image originale
            axes[0, i].imshow(images[i], cmap='gray' if len(images[i].shape)==2 else None)
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            # Histogramme extrait
            axes[1, i].bar(range(self.bins), histograms[i], color='steelblue', alpha=0.7)
            axes[1, i].set_xlabel('Bin')
            axes[1, i].set_ylabel('Fréquence')
            axes[1, i].set_title(f'Histogramme ({self.bins} bins)')
            axes[1, i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 Statistiques histogrammes:")
        print(f"   - Nombre de bins: {self.bins}")
        print(f"   - Shape output: {X_after.shape}")
        print(f"   - Compression: {images[0].size} pixels → {self.bins} features")
