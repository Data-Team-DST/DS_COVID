"""
Module de base pour les transformateurs de pipeline.

Contient les classes UIHandler, TransformLogger et BaseTransform qui servent
de fondation pour tous les transformateurs du pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


# ==============================================================================
# UIHandler - Gestionnaire d'interface utilisateur (Streamlit)
# ==============================================================================

class UIHandler:
    """
    Gestionnaire d'interface pour afficher des messages.
    
    Bascule automatiquement entre Streamlit et la console selon le paramètre use_streamlit.
    Permet d'utiliser le même code en notebook et dans une application Streamlit.
    """
    
    def __init__(self, use_streamlit: bool = False):
        """
        Initialise le gestionnaire d'UI.
        
        Args:
            use_streamlit: Active l'affichage Streamlit si True
        """
        self.use_streamlit = use_streamlit
        
        # Import conditionnel de Streamlit
        if use_streamlit:
            try:
                import streamlit as st
                self.st = st
            except ImportError:
                self.use_streamlit = False
                self.st = None
    
    def info(self, message: str) -> None:
        """Affiche un message d'information."""
        if self.use_streamlit and self.st:
            self.st.info(message)
        else:
            print(f"[INFO] {message}")
    
    def success(self, message: str) -> None:
        """Affiche un message de succès."""
        if self.use_streamlit and self.st:
            self.st.success(message)
        else:
            print(f"[SUCCESS] {message}")
    
    def warning(self, message: str) -> None:
        """Affiche un message d'avertissement."""
        if self.use_streamlit and self.st:
            self.st.warning(message)
        else:
            print(f"[WARNING] {message}")
    
    def error(self, message: str) -> None:
        """Affiche un message d'erreur."""
        if self.use_streamlit and self.st:
            self.st.error(message)
        else:
            print(f"[ERROR] {message}")


# ==============================================================================
# TransformLogger - Gestionnaire de logs
# ==============================================================================

class TransformLogger:
    """
    Gestionnaire de logs pour les transformateurs.
    
    Centralise la journalisation des événements pendant les transformations.
    Utilise UIHandler pour l'affichage des messages.
    """
    
    def __init__(self, name: str, verbose: bool = True, use_streamlit: bool = False):
        """
        Initialise le logger.
        
        Args:
            name: Nom du transformateur (pour identification dans les logs)
            verbose: Active les logs détaillés si True
            use_streamlit: Utilise Streamlit pour l'affichage si True
        """
        self.name = name
        self.verbose = verbose
        self.ui = UIHandler(use_streamlit)
    
    def info(self, message: str) -> None:
        """Log un message d'information."""
        if self.verbose:
            full_message = f"[{self.name}] {message}"
            self.ui.info(full_message)
    
    def warning(self, message: str) -> None:
        """Log un avertissement."""
        if self.verbose:
            full_message = f"[{self.name}] {message}"
            self.ui.warning(full_message)
    
    def error(self, message: str) -> None:
        """Log une erreur."""
        full_message = f"[{self.name}] {message}"
        self.ui.error(full_message)


# ==============================================================================
# BaseTransform - Classe de base pour tous les transformateurs
# ==============================================================================

class BaseTransform(ABC):
    """
    Classe de base pour créer des transformateurs compatibles sklearn.
    
    Principe: Un transformateur implémente UNE méthode abstraite _process()
    qui contient la logique métier. Le reste (logs, UI) est géré automatiquement.
    
    Compatible avec:
    - sklearn.pipeline.Pipeline
    - Sérialisation pickle
    - Streamlit (optionnel)
    
    Utilisation:
        class MyTransform(BaseTransform):
            def _process(self, X):
                # Votre logique de transformation ici
                return X_transformed
    """
    
    def __init__(self, verbose: bool = True, use_streamlit: bool = False):
        """
        Initialise le transformateur.
        
        Args:
            verbose: Active les logs détaillés si True
            use_streamlit: Active l'interface Streamlit si True
        """
        self.verbose = verbose
        self.use_streamlit = use_streamlit
        self._logger = None
    
    @property
    def logger(self) -> TransformLogger:
        """Lazy initialization du logger pour compatibilité pickle."""
        if self._logger is None:
            self._logger = TransformLogger(
                name=self.__class__.__name__,
                verbose=self.verbose,
                use_streamlit=self.use_streamlit
            )
        return self._logger
    
    def fit(self, X: Any, y: Optional[Any] = None) -> 'BaseTransform':
        """
        Phase d'apprentissage (pattern sklearn).
        
        Appelle _fit() pour permettre l'apprentissage optionnel.
        Retourne self pour permettre le chaînage.
        """
        self._fit(X, y)
        return self
    
    def transform(self, X: Any) -> Any:
        """
        Phase de transformation (pattern sklearn).
        
        Appelle _process() qui contient la logique métier.
        """
        return self._process(X)
    
    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Any:
        """
        Phase fit + transform en une seule étape.
        
        Args:
            X: Données d'entrée
            y: Labels optionnels
            
        Returns:
            Données transformées
        """
        return self.fit(X, y).transform(X)
    
    def _fit(self, X: Any, y: Optional[Any] = None) -> None:
        """
        Méthode d'apprentissage optionnelle.
        
        À surcharger si le transformateur nécessite un apprentissage.
        Par défaut, ne fait rien (transformateur stateless).
        """
        pass
    
    @abstractmethod
    def _process(self, X: Any) -> Any:
        """
        Méthode abstraite contenant la logique de transformation.
        
        DOIT être implémentée par les classes filles.
        C'est ici que vous mettez votre code de transformation.
        
        Args:
            X: Données à transformer
        
        Returns:
            Données transformées
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doit implémenter _process()"
        )
    
    def _log(self, message: str, level: str = "info") -> None:
        """
        Helper pour logger des messages.
        
        Args:
            message: Message à logger
            level: Niveau de log ("info", "warning", "error")
        """
        if level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)
    
    def visualize(self, X_before: Any, X_after: Any, n_samples: int = 3) -> None:
        """
        Visualise les données avant/après transformation.
        
        À surcharger dans les classes filles pour des visualisations spécifiques.
        
        Args:
            X_before: Données avant transformation
            X_after: Données après transformation
            n_samples: Nombre d'échantillons à visualiser
        """
        print(f"\n📊 Visualisation de {self.__class__.__name__}")
        print(f"Input shape: {np.array(X_before).shape if hasattr(X_before, '__len__') else 'N/A'}")
        print(f"Output shape: {np.array(X_after).shape if hasattr(X_after, '__len__') else 'N/A'}")
