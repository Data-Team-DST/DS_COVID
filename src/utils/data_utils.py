"""
Data loading and preprocessing utilities for Jupyter notebooks.

This module provides functions for:
- Loading datasets
- Creating preprocessing pipelines
- Train/val/test splitting
- Class weight computation
- Data augmentation generators

Author: Data Pipeline Team
Date: November 2025
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
from keras.applications.efficientnet import preprocess_input as effnet_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# ImageDataGenerator is deprecated in Keras 3, use tf.keras version
# pylint: disable=import-error,no-name-in-module
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from features.Pipelines.transformateurs.image_loaders import ImageLoader
from features.Pipelines.transformateurs.image_preprocessing import (
    ImageMasker,
    ImageResizer,
)

# Configure logger
logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def load_dataset_paths_and_labels(   # reviewed
    dataset_root_dir: Path,
    class_names: List[str],
    n_images_per_class: Optional[int] = None,
    load_masks: bool = False,
    random_sampling: bool = False,
    random_seed: int = 42,
    shuffle: bool = False,
    verbose: bool = True,
) -> Tuple[List[Path], List[Path], List[str], np.ndarray]:
    """
    Load image paths and labels from dataset directory.

    Args:
        dataset_root_dir: Root directory of the dataset
        class_names: List of class names
        n_images_per_class: Number of images per class (None = all images)
        load_masks: If True, also load mask paths
        random_sampling: If True, randomly sample images (with seed for reproducibility)
                        If False, take first N images (sorted order, deterministic)
        random_seed: Random seed for sampling reproducibility (only used if random_sampling=True or shuffle=True)
        shuffle: If True, shuffle all data after loading (maintains image/mask/label correspondence)
        verbose: Print loading information

    Returns:
        Tuple of (image_paths, mask_paths, labels, labels_int)
        If load_masks=False, mask_paths will be empty list
        
    Examples:
        >>> # Déterministe : 100 premières images par classe
        >>> classes = ['COVID', 'Normal', 'Viral Pneumonia']
        >>> imgs, masks, lbls, lbls_int = load_dataset_paths_and_labels(
        ...     data_dir, classes, n_images_per_class=100
        ... )
        
        >>> # Aléatoire équilibré : 100 images random par classe
        >>> imgs, masks, lbls, lbls_int = load_dataset_paths_and_labels(
        ...     data_dir, classes, n_images_per_class=100, random_sampling=True, random_seed=42
        ... )
        
        >>> # Tout le dataset mélangé
        >>> imgs, masks, lbls, lbls_int = load_dataset_paths_and_labels(
        ...     data_dir, classes, n_images_per_class=None, shuffle=True, random_seed=42
        ... )
    """
    if verbose:
        print("=" * 70)
        print("CHARGEMENT DES DONNÉES")
        print("=" * 70)

    # Set random seed if needed
    if random_sampling or shuffle:
        random.seed(random_seed)

    image_paths = []
    mask_paths = []
    labels = []
    labels_int = []

    for idx, cls in enumerate(class_names):
        cls_path = dataset_root_dir / cls / "images"

        if not cls_path.exists():
            if verbose:
                print(f"  ⚠️ Chemin de classe introuvable: {cls_path}")
            continue

        # Get all images for this class
        imgs = sorted(list(cls_path.glob("*.png"))) 
        
        # Sample images if requested
        if n_images_per_class is not None:
            if random_sampling:
                # Random sampling with seed for reproducibility
                num_to_sample = min(n_images_per_class, len(imgs))
                imgs = random.sample(imgs, num_to_sample)
                if num_to_sample < n_images_per_class and verbose:
                    print(f"  ⚠️ {cls}: seulement {num_to_sample}/{n_images_per_class} images disponibles")
            else:
                # Deterministic: take first N
                imgs = imgs[:n_images_per_class]

        if load_masks:
            # Load corresponding masks
            mask_cls_path = dataset_root_dir / cls / "masks"
            for img_path in imgs:
                mask_path = mask_cls_path / img_path.name # Masks have same filename
                if mask_path.exists():
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    labels.append(cls)
                    labels_int.append(idx)
        else:
            # Load only images
            image_paths.extend(imgs)
            labels.extend([cls] * len(imgs))
            labels_int.extend([idx] * len(imgs))

        if verbose:
            suffix = " (avec masques)" if load_masks else ""
            mode = " [aléatoire]" if random_sampling else " [séquentiel]"
            print(f"  {cls:20s}: {len(imgs):4d} images{suffix}{mode}")

    # Validate that dataset is not empty
    if not image_paths:
        raise ValueError(
            f"Aucune image trouvée dans {dataset_root_dir}. "
            f"Vérifiez que les classes {class_names} existent et contiennent des images."
        )

    # Shuffle all data together (maintains correspondence)
    if shuffle:
        # Optimized shuffle using indices (avoids zip/unzip overhead)
        indices = list(range(len(image_paths)))
        random.shuffle(indices)
        
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        labels_int = [labels_int[i] for i in indices]
        
        if load_masks:
            mask_paths = [mask_paths[i] for i in indices]
        
        if verbose:
            print(f"\n  🔀 Données mélangées (seed={random_seed})")

    labels_int = np.array(labels_int)

    if verbose:
        print(f"\n  Total: {len(image_paths)} images")
        print(f"  Classes: {len(class_names)}")
        print(f"  Distribution: {np.bincount(labels_int)}")
        if load_masks:
            print(f"\n✅ Chemins des masques récupérés: {len(mask_paths)}")
        if random_sampling:
            print(f"  🎲 Échantillonnage aléatoire activé (seed={random_seed})")
        if shuffle:
            print(f"  🔀 Mélange activé (seed={random_seed})")

    return image_paths, mask_paths, labels, labels_int


def create_preprocessing_pipeline(
    img_size: Tuple[int, int] = (128, 128),
    color_mode: str = "RGB",
    mask_paths: Optional[List[Path]] = None,
    verbose: bool = True,
) -> Pipeline:
    """
    Create a preprocessing pipeline for images.

    Args:
        img_size: Target image size (width, height)
        color_mode: 'RGB' or 'L' (grayscale)
        mask_paths: Optional mask paths for ImageMasker
        verbose: Print pipeline information

    Returns:
        sklearn Pipeline
    """
    if verbose:
        print("=" * 70)
        print("PREPROCESSING PIPELINE")
        print("=" * 70)

    steps = [
        ("load", ImageLoader(color_mode=color_mode, verbose=verbose)),
        ("resize", ImageResizer(img_size=img_size, verbose=verbose)),
    ]

    # Add masker if mask paths provided
    if mask_paths is not None and len(mask_paths) > 0:
        steps.append(("mask", ImageMasker(mask_paths=mask_paths, verbose=verbose)))

    pipeline = Pipeline(steps)

    if verbose:
        print(f"\n✅ Pipeline créée avec {len(steps)} étapes")

    return pipeline


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def prepare_train_val_test_split(
    images: np.ndarray,
    labels_int: np.ndarray,
    num_classes: Optional[int] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Split data into train/validation/test sets with one-hot encoding.

    Args:
        images: Array of images
        labels_int: Integer labels (0-indexed)
        num_classes: Number of classes (auto-computed if None)
        test_size: Proportion of test set
        val_size: Proportion of validation set (from total)
        random_seed: Random seed for reproducibility
        verbose: Print split information

    Returns:
        Tuple of (x_train, x_val, x_test, 
                  y_train_cat, y_val_cat, y_test_cat,
                  y_train, y_val, y_test)
        
    Examples:
        >>> # Split automatique avec num_classes calculé
        >>> x_tr, x_v, x_te, y_tr_cat, y_v_cat, y_te_cat, y_tr, y_v, y_te = prepare_train_val_test_split(
        ...     images, labels_int, verbose=True
        ... )
        >>> # Utiliser y_tr pour compute_class_weights
        >>> weights = compute_class_weights(y_tr, class_names)
    """
    if verbose:
        print("=" * 70)
        print("TRAIN/VALIDATION/TEST SPLIT")
        print("=" * 70)

    # Validate input consistency
    if len(images) != len(labels_int):
        raise ValueError(
            f"Incohérence: {len(images)} images mais {len(labels_int)} labels. "
            f"Différence: {abs(len(images) - len(labels_int))} échantillon(s)."
        )
    
    # Auto-compute num_classes if not provided
    if num_classes is None:
        num_classes = int(labels_int.max()) + 1
        if verbose:
            print(f"  Nombre de classes détecté automatiquement: {num_classes}")
    
    # Validate num_classes consistency
    actual_num_classes = int(labels_int.max()) + 1
    if num_classes < actual_num_classes:
        raise ValueError(
            f"num_classes={num_classes} mais labels contiennent des valeurs jusqu'à {labels_int.max()}. "
            f"Minimum requis: {actual_num_classes}"
        )
    
    if verbose and num_classes > actual_num_classes:
        print(f"  ⚠️  num_classes={num_classes} mais seulement {actual_num_classes} classes présentes dans les données")

    # First split: train+val vs test
    try:
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            images,
            labels_int,
            test_size=test_size,
            random_state=random_seed,
            stratify=labels_int,
        )
    except ValueError as e:
        raise ValueError(
            f"Échec du split stratifié (test). Vérifiez que chaque classe a au moins 2 échantillons. "
            f"Erreur: {e}"
        ) from e

    # Train+val now contains (1 - test_size) of data

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=y_train_val,
        )
    except ValueError as e:
        raise ValueError(
            f"Échec du split stratifié (val). Essayez de réduire val_size ou augmentez le dataset. "
            f"Erreur: {e}"
        ) from e

    # One-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=num_classes)

    if verbose:
        print(f"\nTrain set: {x_train.shape[0]} images ({x_train.shape[0]/len(images)*100:.1f}%)")
        print(f"  Distribution: {np.bincount(y_train)}")
        print(f"\nValidation set: {x_val.shape[0]} images ({x_val.shape[0]/len(images)*100:.1f}%)")
        print(f"  Distribution: {np.bincount(y_val)}")
        print(f"\nTest set: {x_test.shape[0]} images ({x_test.shape[0]/len(images)*100:.1f}%)")
        print(f"  Distribution: {np.bincount(y_test)}")
        print(f"\n✅ Split effectué avec succès (seed={random_seed})")

    return x_train, x_val, x_test, y_train_cat, y_val_cat, y_test_cat, y_train, y_val, y_test


def compute_class_weights(
    y_train: np.ndarray, classes: List[str], verbose: bool = True
) -> Dict[int, float]:
    """
    Compute balanced class weights.

    Args:
        y_train: Training labels (integer)
        classes: List of class names
        verbose: Print weight information

    Returns:
        Dictionary mapping class index to weight
    """
    if verbose:
        print("=" * 70)
        print("CLASS WEIGHTING")
        print("=" * 70)

    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )

    class_weights = dict(enumerate(class_weights_array))

    if verbose:
        print("\nPoids de classe:")
        for i, cls in enumerate(classes):
            print(f"  {cls:20s}: {class_weights[i]:.3f}")
        print("\n✅ Class weighting activé pour un apprentissage équilibré")

    return class_weights


# pylint: disable=too-many-arguments,too-many-positional-arguments
def create_data_generators(
    x_train: np.ndarray,
    y_train_cat: np.ndarray,
    x_val: np.ndarray,
    y_val_cat: np.ndarray,
    x_test: Optional[np.ndarray] = None,
    y_test_cat: Optional[np.ndarray] = None,
    batch_size: int = 32,
    augment_train: bool = True,
    verbose: bool = True,
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Create Keras data generators with optional augmentation.

    Args:
        x_train: Training images
        y_train_cat: Training labels (one-hot)
        x_val: Validation images
        y_val_cat: Validation labels (one-hot)
        x_test: Test images (optional)
        y_test_cat: Test labels (one-hot, optional)
        batch_size: Batch size
        augment_train: Apply augmentation to training data
        verbose: Print generator information

    Returns:
        Tuple of (train_generator, val_generator, test_generator)
        test_generator is None if x_test not provided
    """
    if verbose:
        print("=" * 70)
        print("DATA AUGMENTATION")
        print("=" * 70)

    # Training generator with augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode="nearest",
        )
        if verbose:
            print("\n✅ Data augmentation configurée:")
            print("  • Rotation: ±10°")
            print("  • Shift: ±10%")
            print("  • Zoom: ±10%")
            print("  • Horizontal flip")
    else:
        train_datagen = ImageDataGenerator()
        if verbose:
            print("\n⚠️ Pas d'augmentation sur le training set")

    # Validation and test generators (no augmentation)
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    if verbose:
        print("\n📊 Création des générateurs...")

    train_generator = train_datagen.flow(
        x_train, y_train_cat, batch_size=batch_size, shuffle=True
    )

    val_generator = val_datagen.flow(
        x_val, y_val_cat, batch_size=batch_size, shuffle=False
    )

    test_generator = None
    if x_test is not None and y_test_cat is not None:
        test_generator = test_datagen.flow(
            x_test, y_test_cat, batch_size=batch_size, shuffle=False
        )

    if verbose:
        print(f"  Train: {len(train_generator)} batches de {batch_size}")
        print(f"  Val:   {len(val_generator)} batches de {batch_size}")
        if test_generator:
            print(f"  Test:  {len(test_generator)} batches de {batch_size}")

    return train_generator, val_generator, test_generator


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def create_transfer_learning_generators(
    x_train: np.ndarray,
    y_train_cat: np.ndarray,
    x_val: np.ndarray,
    y_val_cat: np.ndarray,
    x_test: Optional[np.ndarray] = None,
    y_test_cat: Optional[np.ndarray] = None,
    base_model_name: str = "InceptionV3",
    batch_size: int = 32,
    augment_train: bool = True,
    verbose: bool = True,
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Create data generators with model-specific preprocessing.

    Each pretrained model requires its own preprocessing:
    - VGG16/ResNet50: Subtract ImageNet mean
    - InceptionV3: Normalize to [-1, 1]
    - EfficientNetB0: Normalize to [0, 1]

    Args:
        x_train: Training images
        y_train_cat: Training labels (one-hot)
        x_val: Validation images
        y_val_cat: Validation labels (one-hot)
        x_test: Test images (optional)
        y_test_cat: Test labels (one-hot, optional)
        base_model_name: Name of the pretrained model
        batch_size: Batch size
        augment_train: Apply augmentation to training data
        verbose: Print generator information

    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    if verbose:
        print("=" * 70)
        print(f"DATA GENERATORS - {base_model_name.upper()} PREPROCESSING")
        print("=" * 70)

    # Select preprocessing function
    preprocess_funcs = {
        "VGG16": vgg16_preprocess,
        "ResNet50": resnet_preprocess,
        "EfficientNetB0": effnet_preprocess,
        "InceptionV3": inception_preprocess,
    }

    if base_model_name not in preprocess_funcs:
        raise ValueError(f"Preprocessing inconnu pour: {base_model_name}")

    preprocess_func = preprocess_funcs[base_model_name]

    # Training generator with augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_func,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,  # Medical images - no horizontal flip
            zoom_range=0.05,
            fill_mode="nearest",
        )
        if verbose:
            print("\n✅ Data augmentation configurée:")
            print("  • Rotation: ±10°")
            print("  • Shift: ±5%")
            print("  • Zoom: ±5%")
            print("  • Horizontal flip: NON (images médicales)")
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
        if verbose:
            print("\n⚠️ Pas d'augmentation sur le training set")

    # Validation and test generators (no augmentation)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

    if verbose:
        print(f"  • Preprocessing: {base_model_name}")
        print("\n📊 Création des générateurs...")

    train_generator = train_datagen.flow(
        x_train, y_train_cat, batch_size=batch_size, shuffle=True
    )

    val_generator = val_datagen.flow(
        x_val, y_val_cat, batch_size=batch_size, shuffle=False
    )

    test_generator = None
    if x_test is not None and y_test_cat is not None:
        test_generator = test_datagen.flow(
            x_test, y_test_cat, batch_size=batch_size, shuffle=False
        )

    if verbose:
        print(f"  Train: {len(train_generator)} batches de {batch_size}")
        print(f"  Val:   {len(val_generator)} batches de {batch_size}")
        if test_generator:
            print(f"  Test:  {len(test_generator)} batches de {batch_size}")

    return train_generator, val_generator, test_generator
