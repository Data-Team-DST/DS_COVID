# 🦠 DS_COVID - Détection COVID-19

[![CI/CD](https://github.com/Data-Team-DST/DS_COVID/actions/workflows/pipelinecici.yml/badge.svg)](https://github.com/Data-Team-DST/DS_COVID/actions/workflows/pipelinecici.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://tensorflow.org)

## 📋 Description

Application de **détection COVID-19 à partir d'images radiographiques** utilisant des techniques de Deep Learning et Machine Learning. Ce projet propose une solution complète pour la classification d'images pulmonaires en 4 catégories :

- 🦠 **COVID** - Cas positifs de COVID-19
- ✅ **Normal** - Radiographies pulmonaires saines
- 💨 **Lung_Opacity** - Opacités pulmonaires
- 🔬 **Viral Pneumonia** - Pneumonie virale

## ✨ Fonctionnalités

### 🧠 Modèles de Deep Learning
- **Custom CNN** : Architecture CNN personnalisée optimisée pour l'imagerie médicale (5 blocs convolutionnels)
- **Transfer Learning** : Support des modèles pré-entraînés (VGG16, ResNet50, EfficientNetB0, InceptionV3)
- **Fine-tuning** : Possibilité de débloquer les couches supérieures pour un apprentissage plus fin

### 🔄 Pipeline de Traitement d'Images
- **ImageLoader** : Chargement d'images avec validation des chemins
- **ImageResizer** : Redimensionnement avec préservation optionnelle du ratio d'aspect
- **ImageNormalizer** : Normalisation (min-max, standard, custom)
- **ImageAugmenter** : Augmentation de données (rotation, flip, zoom, bruit, luminosité)
- **ImageMasker** : Application de masques binaires pour segmentation
- **ImageFlattener** : Aplatissement pour modèles ML classiques

### 🔍 Interprétabilité des Modèles
- **Grad-CAM** : Visualisation des zones d'attention du modèle
- **LIME** : Explications par segmentation d'image (super-pixels)
- **SHAP** : Valeurs de Shapley pour explications au niveau pixel

### 📊 Visualisation & Évaluation
- Courbes d'entraînement (loss, accuracy, precision, recall)
- Matrice de confusion
- Rapports de classification détaillés

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip ou conda

### Installation Standard

```bash
# Cloner le repository
git clone https://github.com/Data-Team-DST/DS_COVID.git
cd DS_COVID

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

### Installation avec dépendances de développement

```bash
pip install -e ".[dev]"
```

### Installation sur Google Colab

Le projet supporte nativement Google Colab. Utilisez simplement la cellule de configuration standalone :

```python
# Copiez le contenu de CELL_CONFIG_STANDALONE.py comme première cellule
# La configuration est automatiquement détectée et adaptée pour Colab
```

## 📁 Structure du Projet

```
DS_COVID/
├── config/                          # Fichiers de configuration JSON
│   ├── default_config.json          # Configuration par défaut
│   └── colab_config.json            # Configuration spécifique Colab
│
├── data/                            # Données (non incluses dans Git)
│   ├── raw/                         # Dataset original
│   │   └── COVID-19_Radiography_Dataset/
│   └── processed/                   # Données transformées
│
├── models/                          # Modèles entraînés (.keras, .h5)
│
├── notebooks/                       # Notebooks Jupyter
│   ├── Main_Revamp.ipynb            # Notebook principal
│   ├── Main_Revamp_colab.ipynb      # Version Colab
│   └── test_*.ipynb                 # Notebooks de test
│
├── src/                             # Code source
│   ├── features/                    # Extraction de features
│   │   └── Pipelines/               # Pipelines sklearn
│   │       └── transformateurs/     # Transformateurs personnalisés
│   │           ├── image_augmentation.py
│   │           ├── image_features.py
│   │           ├── image_loaders.py
│   │           ├── image_preprocessing.py
│   │           └── utilities.py
│   │
│   ├── interpretability/            # Modules d'interprétabilité
│   │   ├── gradcam.py               # Grad-CAM
│   │   ├── lime_explainer.py        # LIME
│   │   ├── shap_explainer.py        # SHAP
│   │   └── utils.py                 # Utilitaires communs
│   │
│   ├── utils/                       # Utilitaires généraux
│   │   ├── config.py                # Gestion de configuration
│   │   ├── data_utils.py            # Chargement de données
│   │   ├── model_builders.py        # Construction de modèles
│   │   ├── training_utils.py        # Entraînement
│   │   └── visualization_utils.py   # Visualisation
│   │
│   └── test/                        # Tests unitaires
│
├── .streamlit/                      # Configuration Streamlit
├── pyproject.toml                   # Configuration du projet (PEP 517)
├── requirements.txt                 # Dépendances Python
└── README.md
```

## 💻 Utilisation

### Configuration

La configuration est centralisée dans des fichiers JSON (`config/default_config.json`). Les principaux paramètres :

```json
{
  "images": { "width": 256, "height": 256, "channels": 1 },
  "training": { "batch_size": 32, "epochs": 50, "learning_rate": 0.001 },
  "dataset": { "classes": ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"] }
}
```

### Exemple d'utilisation basique

```python
from src.utils.config import build_config
from src.utils.data_utils import load_dataset, create_preprocessing_pipeline
from src.utils.model_builders import build_custom_cnn, compile_model

# Charger la configuration
config = build_config(project_root, environment="local")

# Charger les données
image_paths, _, labels, labels_int = load_dataset(
    data_dir=config.data_dir,
    categories=config.classes,
    n_images_per_class=500
)

# Créer le pipeline de prétraitement
pipeline = create_preprocessing_pipeline(
    img_size=(128, 128),
    color_mode="RGB"
)

# Transformer les images
images = pipeline.fit_transform(image_paths)

# Construire et compiler le modèle
model = build_custom_cnn(
    input_shape=(128, 128, 3),
    num_classes=4
)
model = compile_model(model, learning_rate=0.001)
```

### Transfer Learning avec Fine-tuning

```python
from src.utils.model_builders import (
    build_transfer_learning_model,
    unfreeze_top_layers
)

# Phase 1 : Feature Extraction
model, base_model = build_transfer_learning_model(
    base_model_name="InceptionV3",
    input_shape=(224, 224, 3),
    num_classes=4,
    freeze_base=True
)

# Phase 2 : Fine-tuning
model = unfreeze_top_layers(
    base_model=base_model,
    model=model,
    n_layers=10,
    learning_rate=5e-5
)
```

### Interprétabilité avec Grad-CAM

```python
from src.interpretability import GradCAM, visualize_gradcam

# Créer l'explainer
gradcam = GradCAM(model)

# Calculer la heatmap pour une image
heatmap = gradcam.compute_heatmap(image, class_idx=0)

# Visualiser
visualize_gradcam(image, heatmap, class_name='COVID', confidence=0.95)
```

## 📊 Dataset

Ce projet utilise le **COVID-19 Radiography Database** disponible sur Kaggle :

- **Source** : [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
- **Classes** : COVID (3,616), Normal (10,192), Lung Opacity (6,012), Viral Pneumonia (1,345)
- **Format** : Images PNG en niveaux de gris
- **Résolution** : 299x299 pixels

### Téléchargement

```bash
# Via Kaggle CLI
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip -d data/raw/COVID-19_Radiography_Dataset/
```

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest

# Tests avec couverture
pytest --cov=src/ --cov-report=html

# Tests spécifiques
pytest src/test/test_pipelines_imports.py
```

## 🔧 CI/CD

Le projet utilise GitHub Actions pour l'intégration continue :

1. **Lint** : Vérification du code avec pylint (score minimum: 8/10)
2. **Unit Tests** : Exécution des tests pytest avec couverture
3. **SonarCloud** : Analyse de qualité du code

## 👥 Auteurs

- **Rafael Cepa** - *Développeur principal*
- **Cirine** - *Contributrice*
- **Steven Moire** - *Contributeur*

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- **DataScientest** pour l'encadrement du projet
- L'équipe du **COVID-19 Radiography Database** pour le dataset
- Les auteurs des méthodes d'interprétabilité (Grad-CAM, LIME, SHAP)

---

## 📚 Références

- Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017)
- Ribeiro et al. "Why Should I Trust You?: Explaining the Predictions of Any Classifier" (2016)
- Lundberg & Lee "A Unified Approach to Interpreting Model Predictions" (2017)

---

## 📖 Documentation Complémentaire

Pour plus de détails sur les modules d'interprétabilité, consultez :
- [src/interpretability/README.md](src/interpretability/README.md)
