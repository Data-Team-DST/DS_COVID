"""
DS-COVID: Package d'analyse de radiographies COVID-19
=====================================================

Ce package fournit des outils pour l'analyse de radiographies pulmonaires
et la classification COVID-19 en utilisant ML et DL avancés.
"""

__version__ = "0.1.0"
__author__ = "Rafael Cepa, Cirine, Steven Moire"
__email__ = "rafael.cepa@example.fr"

# Imports principaux
from .config import Settings

# Crée une instance globale (optionnel)
settings = Settings()

__all__ = ["Settings", "settings", "__version__"]
