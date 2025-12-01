import pandas as pd
import streamlit as st
from .BaseTransform import BaseTransform

class RGB_to_L(BaseTransform):
    """
    Convertit une image RGB en niveaux de gris (L).
    """
    
    def __init__(self, verbose=True, use_streamlit=True):

        super().__init__(verbose=verbose, use_streamlit=use_streamlit)

    
    def fit(self, X, y=None):
        """Pas d'ajustement nécessaire."""
        self.log("RGB_to_L: fit appelé")
        return super().fit(X, y)
    
    def transform(self, X):
        """
        Convertit une image RGB en niveaux de gris (L).
        """

        if self.use_streamlit:
            st.divider()
            st.info("🔄 Transformation des images RGB en niveaux de gris (L)...")
        
        # Si X est un DataFrame avec une colonne 'image_array', convertir chaque image
        if isinstance(X, pd.DataFrame) and 'image_array' in X.columns:
            X_transformed = X.copy()
            converted_images = []
            
            for idx, row in self.progress(X.iterrows(), desc="Conversion des images", total=len(X)):
                if row['image_array'] is not None:
                    l_img = self.rgb_to_l(row['image_array'])
                    converted_images.append(l_img)
                else:
                    converted_images.append(None)
            
            X_transformed['image_array'] = converted_images
            return X_transformed
        
        # Si X est une liste d'images
        elif isinstance(X, list):
            X_converted = []
            for img in self.progress(X, desc="Conversion des images", total=len(X)):
                l_img = self.rgb_to_l(img)
                X_converted.append(l_img)
            return X_converted
        
        # Sinon, retourner X tel quel
        else:
            self.log("Format de données non reconnu pour RGB_to_L, retour de X tel quel")
            return X
    
    def rgb_to_l(self, image):
        """
        Convertit une image RGB en niveaux de gris (L).
        
        Args:
            image: image au format RGB (numpy array ou autre)
            
        Returns:
            image convertie en niveaux de gris (L)
        """
        from PIL import Image
        import numpy as np

        # Conversion avec PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        l_image = pil_image.convert('L')
        
        return np.array(l_image)