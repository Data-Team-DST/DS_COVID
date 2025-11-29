import pandas as pd
import streamlit as st
from .BaseTransform import BaseTransform

class TupleToDataFrame(BaseTransform):
    """
    Convertit un tuple (image_paths, mask_paths, labels) en DataFrame.
    """
    
    def __init__(self, verbose=True, use_streamlit=True):
        super().__init__(verbose=verbose, use_streamlit=use_streamlit)
    
    def fit(self, X, y=None):
        """Pas d'ajustement nécessaire."""
        self.log("TupleToDataFrame: fit appelé")
        return super().fit(X, y)
    
    def transform(self, X):
        """
        Convertit le tuple en DataFrame.
        
        Args:
            X: tuple (image_paths, mask_paths, labels) ou DataFrame
            
        Returns:
            DataFrame avec colonnes ['image_path', 'mask_path', 'label']
        """
        if self.use_streamlit:
            st.divider()
            st.info("🔄 Conversion du tuple en DataFrame...")

        # Si c'est déjà un DataFrame, on le retourne tel quel
        if isinstance(X, pd.DataFrame):
            self.log("Données déjà en DataFrame, pas de conversion nécessaire")
            return X
        
        # Conversion du tuple en DataFrame
        if isinstance(X, tuple) and len(X) == 3:
            image_paths, mask_paths, labels = X
            
            df = pd.DataFrame({
                'image_path': image_paths,
                'mask_path': mask_paths,
                'label': labels
            })
            
            # Forcer les types de colonnes pour éviter les problèmes de conversion Arrow
            df['image_path'] = df['image_path'].astype(str)
            df['mask_path'] = df['mask_path'].astype(str)
            df['label'] = df['label'].astype(str)
            
            self.log(f"Conversion en DataFrame : {len(df)} lignes, {len(df.columns)} colonnes")
            
            if self.use_streamlit:
                st.subheader("✅ Aperçu des données converties")
                # Afficher seulement les statistiques de base sans les chemins
                st.write(f"**Nombre total d'entrées:** {len(df)}")
                st.write(f"**Distribution des labels:**")
                st.dataframe(df)
                st.success(f"✅ Conversion réussie : {len(df)} entrées")

            return df
        else:
            raise ValueError(f"Format d'entrée non supporté. Attendu: tuple de 3 éléments, reçu: {type(X)}")