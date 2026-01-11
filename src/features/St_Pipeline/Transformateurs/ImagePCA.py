"""
ImagePCA - Réduction de dimension par ACP.

Transformateur pour appliquer une analyse en composantes principales.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from .base import BaseTransform


class ImagePCA(BaseTransform):
    """
    Réduction de dimension par ACP (PCA) sur les images aplaties.
    
    Ce transformateur applique une analyse en composantes principales (PCA)
    pour réduire la dimensionnalité des images tout en préservant
    la variance maximale.
    
    Pattern sklearn: Transformation stateful (le PCA doit être fit).
    
    Usage:
        pca = ImagePCA(n_components=50)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
    """
    
    def __init__(self, n_components=50, **kwargs):
        """
        Initialise le transformateur PCA.
        
        Args:
            n_components: Nombre de composantes principales à conserver
            **kwargs: Paramètres de BaseTransform (verbose, use_streamlit)
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
    
    def _fit(self, X: Any, y: Optional[Any] = None) -> None:
        """
        Apprend les composantes principales sur les données.
        
        Args:
            X: Images d'entraînement
            y: Labels (unused)
        """
        # Préparer les données
        X_flat = self._prepare_data(X)
        
        self._log(f"Apprentissage PCA avec {self.n_components} composantes sur {X_flat.shape}")
        
        # Fit PCA
        self.pca.fit(X_flat)
        
        variance_explained = self.pca.explained_variance_ratio_.sum()
        self._log(f"PCA fitted. Variance expliquée: {variance_explained:.2%}")
        
        # Visualisation en mode Streamlit
        if self.use_streamlit:
            self._plot_pca_variance()
    
    def _plot_pca_variance(self):
        """Visualise la variance expliquée par PCA avec Plotly."""
        if self.use_streamlit and HAS_STREAMLIT:
            st.markdown("### 📊 Variance Expliquée PCA")
            
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Variance cumulée
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=list(range(1, len(cumsum)+1)),
                    y=cumsum,
                    mode='lines+markers',
                    name='Variance cumulée',
                    line=dict(color='blue', width=2)
                ))
                fig1.add_hline(y=0.95, line_dash="dash", line_color="red", 
                              annotation_text="95%")
                fig1.add_hline(y=0.90, line_dash="dash", line_color="orange",
                              annotation_text="90%")
                fig1.update_layout(
                    title="Variance Cumulée",
                    xaxis_title="Composantes",
                    yaxis_title="Variance Expliquée",
                    height=300
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Variance par composante (top 20)
                n_show = min(20, len(self.pca.explained_variance_ratio_))
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(1, n_show+1)),
                    y=self.pca.explained_variance_ratio_[:n_show],
                    marker_color='steelblue'
                ))
                fig2.update_layout(
                    title=f"Variance par Composante (Top {n_show})",
                    xaxis_title="Composante",
                    yaxis_title="Variance",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            st.info(f"📊 Total variance expliquée: **{cumsum[-1]:.2%}** avec {self.n_components} composantes")
    
    def _process(self, X: Any) -> np.ndarray:
        """
        Applique la transformation PCA aux images.
        
        Args:
            X: Images à transformer
        
        Returns:
            Features PCA (numpy array 2D)
        """
        # Préparer les données
        X_flat = self._prepare_data(X)
        
        self._log(f"Application PCA sur {X_flat.shape}")
        
        # Transform
        X_pca = self.pca.transform(X_flat)
        
        self._log(f"PCA terminé. Shape: {X_pca.shape}")
        
        return X_pca
    
    def _prepare_data(self, X: Any) -> np.ndarray:
        """
        Prépare les données pour PCA (aplatit les images).
        
        Args:
            X: Images (numpy array, liste ou DataFrame)
        
        Returns:
            Numpy array 2D (n_samples, n_features)
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
            
            return np.array(X_flat)
        
        # Cas 2: Numpy array ou liste
        else:
            data_array = np.array(X)
            n_samples = data_array.shape[0]
            return data_array.reshape(n_samples, -1)
    
    def _plot_pca_variance(self):
        """Visualise la variance expliquée par PCA avec Plotly."""
        if self.use_streamlit and HAS_STREAMLIT:
            st.markdown("### 📊 Variance Expliquée PCA")
            
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Variance cumulée
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=list(range(1, len(cumsum)+1)),
                    y=cumsum,
                    mode='lines+markers',
                    name='Variance cumulée',
                    line=dict(color='blue', width=2)
                ))
                fig1.add_hline(y=0.95, line_dash="dash", line_color="red", 
                              annotation_text="95%")
                fig1.add_hline(y=0.90, line_dash="dash", line_color="orange",
                              annotation_text="90%")
                fig1.update_layout(
                    title="Variance Cumulée",
                    xaxis_title="Composantes",
                    yaxis_title="Variance Expliquée",
                    height=300
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Variance par composante (top 20)
                n_show = min(20, len(self.pca.explained_variance_ratio_))
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(range(1, n_show+1)),
                    y=self.pca.explained_variance_ratio_[:n_show],
                    marker_color='steelblue'
                ))
                fig2.update_layout(
                    title=f"Variance par Composante (Top {n_show})",
                    xaxis_title="Composante",
                    yaxis_title="Variance",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            st.info(f"📊 Total variance expliquée: **{cumsum[-1]:.2%}** avec {self.n_components} composantes")
    
    def visualize(self, X_before: Any, X_after: Any, n_samples: int = 3) -> None:
        """Visualise la variance expliquée par PCA."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'ImagePCA: Réduction à {self.n_components} composantes', fontsize=16, fontweight='bold')
        
        # Variance expliquée cumulée
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        axes[0].plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2)
        axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[0].axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
        axes[0].set_xlabel('Nombre de composantes')
        axes[0].set_ylabel('Variance expliquée cumulée')
        axes[0].set_title('Variance expliquée par les composantes')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        # Variance par composante
        axes[1].bar(range(1, min(20, len(self.pca.explained_variance_ratio_))+1), 
                   self.pca.explained_variance_ratio_[:20], color='steelblue')
        axes[1].set_xlabel('Composante')
        axes[1].set_ylabel('Variance expliquée')
        axes[1].set_title('Variance par composante (top 20)')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 Statistiques PCA:")
        print(f"   - Dimensions avant: {X_before.shape if hasattr(X_before, 'shape') else 'N/A'}")
        print(f"   - Dimensions après: {X_after.shape}")
        print(f"   - Variance totale expliquée: {cumsum[-1]:.2%}")
        
        # Calcul du taux de compression
        if hasattr(X_before, 'shape') and len(X_before.shape) == 2:
            compression_ratio = X_after.shape[1] / X_before.shape[1]
            print(f"   - Compression: {compression_ratio:.2%} ({X_before.shape[1]} → {X_after.shape[1]} features)")
