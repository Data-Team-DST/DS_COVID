"""
Module d'interface graphique (GUI) pour le projet DS_COVID.

Ce module fournit une interface utilisateur simple basée sur Tkinter
permettant de sélectionner un dossier et de lancer une analyse automatique
des images et métadonnées.
"""

import tkinter as tk
from tkinter import filedialog, ttk


class GUI:
    """
    Interface graphique principale pour l'analyse d'images et métadonnées.

    Cette interface permet :
      - la sélection d'un dossier racine contenant les données à analyser,
      - le lancement du processus d'analyse,
      - l'affichage de la progression et du statut.
    """

    def __init__(self, root: tk.Tk, on_analyse_callback):
        """
        Initialise l'interface graphique principale.

        Args:
            root (tk.Tk): Fenêtre principale Tkinter.
            on_analyse_callback (callable): Fonction appelée lors du clic sur
                le bouton "Lancer l’analyse".
        """
        self.root = root
        self.root.title("Analyse automatique des images + metadata")

        # Zone de sélection du dossier
        tk.Label(
            root,
            text="📁 Dossier principal :"
        ).grid(row=0, column=0, sticky="w", padx=5, pady=10)

        self.entry_root = tk.Entry(root, width=60)
        self.entry_root.grid(row=0, column=1, padx=5)

        tk.Button(
            root, text="Parcourir", command=self.choisir_dossier
        ).grid(row=0, column=2)

        # Bouton de lancement d’analyse
        self.btn_analyse = tk.Button(
            root,
            text="🚀 Lancer l’analyse",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            command=on_analyse_callback
        )
        self.btn_analyse.grid(row=1, column=1, pady=10)

        # Barre de progression
        self.progress_bar = ttk.Progressbar(
            root,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.progress_bar.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

        # Label de statut
        self.label_status = tk.Label(root, text="", fg="gray")
        self.label_status.grid(row=3, column=0, columnspan=3, pady=5)

    def choisir_dossier(self):
        """Ouvre un sélecteur de dossier et renseigne le champ de saisie."""
        path = filedialog.askdirectory(title="Choisir le dossier principal")
        if path:
            self.entry_root.delete(0, tk.END)
            self.entry_root.insert(0, path)

    def get_dossier_path(self) -> str:
        """
        Retourne le chemin du dossier actuellement sélectionné.

        Returns:
            str: Chemin du dossier principal choisi par l'utilisateur.
        """
        return self.entry_root.get()
