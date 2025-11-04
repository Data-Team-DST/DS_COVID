"""
Module d'analyse des images.

Ce module collecte des informations statistiques sur les images
présentes dans un dossier donné : formats, dimensions, modes de couleur,
poids total, etc.
"""

import os
from PIL import Image  # type: ignore


def collect_image_stats(image_folder: str):
    """
    Analyse un dossier d'images et collecte des statistiques globales.

    Args:
        image_folder (str): Chemin vers le dossier contenant les images.

    Returns:
        dict: Dictionnaire contenant :
            - 'names' (set[str]): Ensemble des noms d’images sans extension.
            - 'dimensions' (dict[str, tuple[int, int]]):
            Dimensions de chaque image.
            - 'formats' (set[str]):
            Extensions de fichiers rencontrées (.png, .jpg...).
            - 'color_modes' (set[str]): Modes couleur (ex. "RGB", "L").
            - 'shapes' (set[tuple[int, int]]): Tailles uniques rencontrées.
            - 'total_size' (int): Poids total des fichiers (en octets).
            - 'logs' (list[str]): Liste des erreurs rencontrées.
    """
    total_size = 0
    formats = set()
    color_modes = set()
    shapes = set()
    image_dimensions = {}
    image_names_set = set()
    logs = []

    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            name_only = os.path.splitext(img_name)[0]
            img_path = os.path.join(image_folder, img_name)
            with Image.open(img_path) as im:
                image_names_set.add(name_only)
                image_dimensions[name_only] = im.size
                total_size += os.path.getsize(img_path)
                formats.add(os.path.splitext(img_name)[1].lower())
                color_modes.add(im.mode)
                shapes.add(im.size)

        except (OSError, ValueError) as err:
            logs.append(f"Erreur sur {img_name}: {err}")

    return {
        "names": image_names_set,
        "dimensions": image_dimensions,
        "formats": formats,
        "color_modes": color_modes,
        "shapes": shapes,
        "total_size": total_size,
        "logs": logs,
    }
