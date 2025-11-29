import logging
from typing import Any, Dict, Optional, List, Iterable
from abc import abstractmethod,ABC
import sys 

import streamlit as st
from tqdm import tqdm

import plotly.express as px
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class BaseTransform(BaseEstimator, TransformerMixin):
    """
    Classe de base pour tous les transformateurs du pipeline.
    """

    def __init__(self,verbose: bool = True, use_streamlit: bool = True, **kwargs: Any) -> None:
        """
        Initialise le transformateur avec des paramètres optionnels.
        """
        self.verbose = verbose
        self.use_streamlit = use_streamlit
        self.params = kwargs
        self.logger = logging.getLogger(__name__)
        self._progress_bar = None
        self._status_text = None
        self.__is__fitted = False
        
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
            self.log("Transformer initialized with parameters: {}".format(self.params))
        st.divider()
        st.info(f"{self.__class__.__name__} initialisé avec paramètres : {self.params}")

    # @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None) -> "BaseTransform":
        """
        Ajuste le transformateur aux données.
        """
        with st.spinner(f"Ajustement du {self.__class__.__name__} en cours..."):
            self.log(f"Ajustement du {self.__class__.__name__} en cours...")
            self.__is__fitted = True
        st.success(f"{self.__class__.__name__} ajusté avec succès!")
        return self

    # @abstractmethod
    def transform(self, X: {Any}) -> Any:
        """
        Transforme les données en utilisant le transformateur ajusté.
        """
        with st.spinner(f"Transformation {self.__class__.__name__} en cours..."):
            self.log(f"Transformation {self.__class__.__name__} en cours...")
        st.success(f"Transformation {self.__class__.__name__} réussie!")
        return X

    def progress(self, iterable: Iterable, desc: str = "Processing", total: Optional[int] = None) -> Iterable:
        """
        Crée une barre de progression pour l'itération (tqdm ou Streamlit).
        
        Args:
            iterable: L'itérable à encapsuler
            desc: Texte de description pour la barre de progression
            total: Nombre total d'éléments (optionnel, déduit de l'itérable si possible)
        
        Yields:
            Éléments de l'itérable
        """
        if not self.verbose:
            yield from iterable
            return
        
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        
        if self.use_streamlit and total is not None:
            self._progress_bar = st.progress(0)
            self._status_text = st.empty()
            
            for i, item in enumerate(iterable):
                progress = (i + 1) / total
                self._progress_bar.progress(progress)
                self._status_text.text(f"{desc}: {i + 1}/{total}")
                yield item
            
            self._status_text.text(f"{desc}: Terminé!")
            self._progress_bar.empty()
            self._status_text.empty()
        else:
            # Utilise tqdm par défaut
            yield from tqdm(iterable, desc=desc, total=total)

    def log(self,msg:str) -> None:
        if self.verbose:
            self.logger.info(msg)

    def st_log(self, msg: str) -> None:
        """
        Affiche des messages dans Streamlit si verbose est activé.
        """
        if self.verbose and self.use_streamlit:
            st.info(msg)

    def plot(self, plot_func, *args, **kwargs) -> None:
        """
        Affiche un graphique dans Streamlit si verbose est activé.
        """
        if self.verbose and self.use_streamlit:
            fig = plot_func(*args, **kwargs)
            st.pyplot(fig)
