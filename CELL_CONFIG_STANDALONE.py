"""
╔════════════════════════════════════════════════════════════════════════════╗
║  🎯 CELLULE DE CONFIGURATION STANDALONE - COPIER-COLLER DANS VOS NOTEBOOKS ║
╚════════════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
-------------
1. Copiez TOUT le contenu de cette cellule
2. Collez-le comme PREMIÈRE CELLULE de votre notebook
3. Exécutez la cellule
4. Les variables sont prêtes à l'emploi !

Cette cellule est 100% autonome et fonctionne partout :
✅ Google Colab (clone + installe automatiquement)
✅ WSL / Linux Local
✅ Tout environnement Jupyter

APRÈS EXÉCUTION, VOUS POUVEZ UTILISER:
- config: Objet de configuration (config.batch_size, config.data_dir, etc.)
- ENV: Environnement détecté ('colab', 'wsl', 'local')
- Tous les imports des transformers

"""

# =============================================================================
# IMPORTS STANDARDS
# =============================================================================

import os
import sys
import subprocess
from pathlib import Path


# =============================================================================
# DÉTECTION AUTOMATIQUE DE L'ENVIRONNEMENT
# =============================================================================

def detect_environment():
    """Détecte l'environnement (colab, wsl, local)"""
    try:
        import google.colab
        return "colab"
    except ImportError:
        is_wsl = os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
        return "wsl" if is_wsl else "local"

ENV = detect_environment()
print(f"🌍 Environnement: {ENV.upper()}")


# =============================================================================
# BOOTSTRAP COLAB (Clone + Install si nécessaire)
# =============================================================================

if ENV == "colab":
    print("\n🚀 Bootstrap Colab...")
    
    os.chdir('/content')
    if not os.path.exists('/content/DS_COVID_ORGA'):
        print("📥 Clonage du repository...")
        subprocess.run(['git', 'clone', 'https://github.com/Data-Team-DST/DS_COVID.git', 'DS_COVID_ORGA'], check=True)
    
    os.chdir('/content/DS_COVID_ORGA')
    
    # Checkout de la branche rafael2
    result = subprocess.run(
        ['git', 'checkout', '-b', 'rafael2', 'origin/rafael2'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        # Si la branche locale existe déjà, juste switcher
        subprocess.run(['git', 'checkout', 'rafael2'], capture_output=True)
    
    # ✅ Colab a déjà tous les packages nécessaires
    print("✅ Utilisation des packages Colab natifs:")
    print("   • NumPy, Pandas, Matplotlib")
    print("   • scikit-learn, scipy")
    print("   • Pillow, tqdm")

    
    # Optionnel : Montage Google Drive pour le dataset
    try:
        print("💾 Montage Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Vérifier si le dataset est disponible sur Drive
        drive_dataset = Path('/content/drive/MyDrive/DS_COVID/archive_covid.zip')
        if drive_dataset.exists():
            print("📦 Extraction dataset depuis Drive...")
            os.makedirs('./data/raw/', exist_ok=True)
            subprocess.run(['unzip', '-o', '-q', str(drive_dataset), '-d', './data/raw/'])
            print("✅ Dataset extrait")
        else:
            print(f"⚠️ Dataset non trouvé sur Drive: {drive_dataset}")
            print("   Vous pouvez télécharger le dataset manuellement")
    except Exception as e:
        print(f"⚠️ Drive non monté: {e}")
    
    print("✅ Bootstrap Colab terminé")


# =============================================================================
# CONFIGURATION DES CHEMINS
# =============================================================================

# Déterminer project_root selon l'environnement
if ENV == "colab":
    project_root = Path('/content/DS_COVID_ORGA')
elif ENV == "wsl":
    project_root = Path('/home/cepa/DST/projet_DS/DS_COVID_ORGA')
else:  # local
    # Depuis un notebook dans notebooks/
    project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()

# Ajouter le projet au sys.path pour les imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✅ Chemin projet ajouté: {project_root}")

# Configuration manuelle (pas de fichier config.py dans ce projet)
data_dir = project_root / 'data' / 'raw' / 'COVID-19_Radiography_Dataset' / 'COVID-19_Radiography_Dataset'
categories = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
img_size = (299, 299)
batch_size = 32
epochs = 10

print(f"📂 Dataset configuré: {data_dir}")
print(f"🏷️ Classes: {', '.join(categories)}")


# =============================================================================
# IMPORTS DES TRANSFORMERS
# =============================================================================

try:
    from src.features.Pipelines.transformateurs.image_loaders import ImageLoader
    from src.features.Pipelines.transformateurs.image_preprocessing import (
        ImageResizer, ImageNormalizer, ImageFlattener, ImageMasker, ImageBinarizer
    )
    from src.features.Pipelines.transformateurs.image_augmentation import (
        ImageAugmenter, ImageRandomCropper
    )
    from src.features.Pipelines.transformateurs.image_features import (
        ImageHistogram, ImagePCA, ImageStandardScaler
    )
    from src.features.Pipelines.transformateurs.utilities import (
        VisualizeTransformer, SaveTransformer
    )
    print("✅ Tous les transformateurs importés")
except ImportError as e:
    print(f"⚠️ Erreur import transformateurs: {e}")
    print(f"   Vérifiez que le projet est bien dans: {project_root}")


# =============================================================================
# IMPORTS ML/DL
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# =============================================================================
# CONFIGURATION MATPLOTLIB
# =============================================================================

plt.rcParams['figure.figsize'] = (15, 10)
sns.set_style('whitegrid')

# =============================================================================
# AFFICHAGE DU RÉSUMÉ
# =============================================================================

print("\n" + "=" * 70)
print("✅ CONFIGURATION PRÊTE - DS_COVID Project")
print("=" * 70)
print(f"📂 Projet: {project_root}")
print(f"📊 Dataset: {data_dir}")
print(f"🏷️ Classes: {', '.join(categories)}")
print(f"🎛️ Images: {img_size}")
print(f"🔧 Batch: {batch_size} | Époques: {epochs}")
print(f"📐 Dataset accessible: {'✅' if data_dir.exists() else '❌'}")
if not data_dir.exists():
    print(f"   ⚠️ Le dataset doit être placé dans: {data_dir}")
    if ENV == "colab":
        print(f"   💡 Uploadez archive_covid.zip sur Google Drive ou téléchargez directement")
print("=" * 70)
print("\n💡 Variables disponibles:")
print("   • project_root: Racine du projet (Path)")
print("   • data_dir: Dossier des données (Path)")
print("   • categories: Liste des 4 classes")
print("   • img_size: Taille des images (tuple)")
print("   • batch_size, epochs: Hyperparamètres")
print("   • ENV: Environnement actuel")
print("\n🎯 Transformateurs disponibles:")
print("   • Loaders: ImageLoader")
print("   • Preprocessing: ImageResizer, ImageNormalizer, ImageFlattener, ImageMasker, ImageBinarizer")
print("   • Augmentation: ImageAugmenter, ImageRandomCropper")
print("   • Features: ImageHistogram, ImagePCA, ImageStandardScaler")
print("   • Utilities: VisualizeTransformer, SaveTransformer")
print("=" * 70)
