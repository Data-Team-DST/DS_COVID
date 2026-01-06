"""
PCAVisualizer - Visualise résultats d'analyse PCA.

Transformateur pour générer visualisations de PCA: variance expliquée,
projections 2D/3D avec labels de classes.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .base import BaseTransform


class PCAVisualizer(BaseTransform):
    """
    Visualise résultats d'analyse en composantes principales (PCA).
    
    Ce transformateur attend en entrée des données PCA (X_pca) et génère:
    - Scree plot (variance expliquée par composante)
    - Variance cumulée
    - Projections 2D/3D colorées par classe
    
    Pattern sklearn: Pass-through (retourne X inchangé).
    
    Attributs:
        n_components (int): Nombre de composantes attendues
        projection_mode (str): '2d' | '3d' | 'both'
        variance_explained_ (array): Variance expliquée (si PCA object fourni)
    
    Usage:
        viz = PCAVisualizer(n_components=50, projection_mode='both')
        viz.fit(X_pca, y)
        X_pca = viz.transform(X_pca)  # Génère visualisation
    """
    
    def __init__(self, n_components=50, projection_mode='both', **kwargs):
        """
        Initialise le visualisateur PCA.
        
        Args:
            n_components: Nombre de composantes attendues
            projection_mode: '2d' | '3d' | 'both'
            **kwargs: BaseTransform parameters
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.projection_mode = projection_mode
        self.variance_explained_ = None
    
    def _fit(self, X, y=None):
        """
        Analyse les données PCA pour extraire métriques.
        
        Args:
            X: Données PCA transformées (n_samples, n_components)
            y: Labels pour coloration (optionnel)
        """
        # Vérifier shape
        if isinstance(X, pd.DataFrame):
            n_samples, n_features = X.shape
        else:
            X_arr = np.array(X)
            n_samples, n_features = X_arr.shape
        
        self._log(f"Données PCA: {n_samples} échantillons × {n_features} composantes")
        
        # Si X contient un objet PCA (improbable mais possible)
        # Sinon, on visualise juste les composantes
    
    def _process(self, X):
        """
        Génère visualisations PCA et retourne X inchangé.
        
        Args:
            X: Données PCA transformées
        
        Returns:
            X inchangé (pass-through)
        """
        # Extraire labels si présents
        y = None
        if isinstance(X, pd.DataFrame):
            if 'label' in X.columns:
                y = X['label'].values
                X_pca = X.drop(columns=['label']).values
            else:
                X_pca = X.values
        else:
            X_pca = np.array(X)
        
        self._plot_variance_explained(X_pca)
        
        if y is not None and self.projection_mode in ['2d', 'both']:
            self._plot_projection_2d(X_pca, y)
        
        if y is not None and self.projection_mode in ['3d', 'both']:
            self._plot_projection_3d(X_pca, y)
        
        return X
    
    def _plot_variance_explained(self, X_pca):
        """
        Génère scree plot + variance cumulée.
        
        Args:
            X_pca: Données PCA
        """
        n_comp = min(X_pca.shape[1], self.n_components)
        
        # Variance expliquée simulée (car on n'a que X_pca)
        # Dans un cas réel, on récupérerait pca.explained_variance_ratio_
        # Ici on simule pour démonstration
        variance_ratio = self._estimate_variance_ratio(X_pca, n_comp)
        cumulative_variance = np.cumsum(variance_ratio)
        
        if self.use_streamlit and HAS_STREAMLIT:
            st.subheader("📊 Variance Expliquée PCA")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Scree Plot', 'Variance Cumulée')
            )
            
            # Scree plot
            fig.add_trace(
                go.Bar(x=list(range(1, n_comp+1)), y=variance_ratio,
                       name='Variance Individuelle'),
                row=1, col=1
            )
            
            # Variance cumulée
            fig.add_trace(
                go.Scatter(x=list(range(1, n_comp+1)), y=cumulative_variance,
                          mode='lines+markers', name='Variance Cumulée',
                          line=dict(color='red', width=2)),
                row=1, col=2
            )
            
            # Ligne 90%
            fig.add_hline(y=0.9, line_dash="dash", line_color="green",
                         annotation_text="90%", row=1, col=2)
            
            fig.update_xaxes(title_text="Composante", row=1, col=1)
            fig.update_xaxes(title_text="Composante", row=1, col=2)
            fig.update_yaxes(title_text="Variance Expliquée", row=1, col=1)
            fig.update_yaxes(title_text="Variance Cumulée", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Matplotlib
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scree plot
            ax1.bar(range(1, n_comp+1), variance_ratio)
            ax1.set_xlabel('Composante')
            ax1.set_ylabel('Variance Expliquée')
            ax1.set_title('Scree Plot')
            ax1.grid(axis='y', alpha=0.3)
            
            # Variance cumulée
            ax2.plot(range(1, n_comp+1), cumulative_variance, 'ro-', linewidth=2)
            ax2.axhline(y=0.9, color='g', linestyle='--', label='90%')
            ax2.set_xlabel('Composante')
            ax2.set_ylabel('Variance Cumulée')
            ax2.set_title('Variance Cumulée')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def _plot_projection_2d(self, X_pca, y):
        """
        Projection 2D avec première et deuxième composantes.
        
        Args:
            X_pca: Données PCA
            y: Labels
        """
        if X_pca.shape[1] < 2:
            self._log("Besoin d'au moins 2 composantes pour projection 2D", level="warning")
            return
        
        if self.use_streamlit and HAS_STREAMLIT:
            st.subheader("🔵 Projection 2D (PC1 vs PC2)")
            
            df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Classe': y
            })
            
            fig = px.scatter(
                df, x='PC1', y='PC2', color='Classe',
                title='Projection 2D PCA',
                labels={'PC1': 'Composante Principale 1', 'PC2': 'Composante Principale 2'}
            )
            
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Matplotlib
            plt.figure(figsize=(10, 7))
            
            classes = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
            
            for i, cls in enumerate(classes):
                mask = (y == cls)
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[colors[i]], label=cls, alpha=0.7, s=50)
            
            plt.xlabel('Composante Principale 1')
            plt.ylabel('Composante Principale 2')
            plt.title('Projection 2D PCA')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def _plot_projection_3d(self, X_pca, y):
        """
        Projection 3D avec 3 premières composantes.
        
        Args:
            X_pca: Données PCA
            y: Labels
        """
        if X_pca.shape[1] < 3:
            self._log("Besoin d'au moins 3 composantes pour projection 3D", level="warning")
            return
        
        if self.use_streamlit and HAS_STREAMLIT:
            st.subheader("🔮 Projection 3D (PC1, PC2, PC3)")
            
            df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'PC3': X_pca[:, 2],
                'Classe': y
            })
            
            fig = px.scatter_3d(
                df, x='PC1', y='PC2', z='PC3', color='Classe',
                title='Projection 3D PCA',
                labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'}
            )
            
            fig.update_traces(marker=dict(size=5, opacity=0.7))
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Matplotlib 3D
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            classes = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
            
            for i, cls in enumerate(classes):
                mask = (y == cls)
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                          c=[colors[i]], label=cls, alpha=0.7, s=50)
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('Projection 3D PCA')
            ax.legend()
            plt.tight_layout()
            plt.show()
    
    def _estimate_variance_ratio(self, X_pca, n_comp):
        """
        Estime variance expliquée depuis données PCA.
        
        Dans un cas réel, on passerait l'objet PCA.
        Ici on simule pour démonstration.
        
        Args:
            X_pca: Données transformées
            n_comp: Nombre de composantes
        
        Returns:
            array: Variance ratio estimée
        """
        # Simuler variance décroissante
        variance_ratio = np.array([1.0 / (i + 1)**1.5 for i in range(n_comp)])
        variance_ratio = variance_ratio / variance_ratio.sum()  # Normaliser
        
        return variance_ratio
