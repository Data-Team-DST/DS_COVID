"""
Module de chargement et de traitement des métadonnées d'images.

Ce module permet de lire un fichier Excel contenant les informations
liées aux images (nom, taille, etc.) et de préparer les données pour
l'analyse automatisée.
"""

import pandas as pd  # type: ignore


def parse_size(size_str: str):
    """
    Convertit une chaîne de taille au format
    'largeur*hauteur' en tuple d'entiers.

    Args:
        size_str (str): Taille sous forme de chaîne, par exemple "224*224".

    Returns:
        tuple[int, int] | None: Tuple (largeur, hauteur) si conversion réussie,
        sinon None.
    """
    try:
        width, height = map(int, str(size_str).split("*"))
        return width, height
    except (ValueError, AttributeError):
        return None


def load_metadata(xlsx_path: str):
    """
    Charge le fichier Excel contenant les métadonnées et prépare les données.

    Args:
        xlsx_path (str): Chemin vers le fichier Excel
        contenant les métadonnées.

    Returns:
        tuple[set[str], dict[str, tuple[int, int] | None]]:
            - Un ensemble contenant les noms de fichiers.
            - Un dictionnaire associant chaque
            nom de fichier à sa taille (w, h).
    """
    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()
    df["parsed_size"] = df["SIZE"].apply(parse_size)

    return (
        set(df["FILE NAME"].astype(str)),
        dict(zip(df["FILE NAME"], df["parsed_size"]))
    )
