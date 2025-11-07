"""Image loading transformers for the data pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from tqdm import tqdm  # type: ignore

# Configuration du logger
logger = logging.getLogger(__name__)


class ImageLoader(BaseEstimator, TransformerMixin):
    """Transformer to load images from file paths.

    This transformer loads images from disk and converts them to grayscale.
    It validates file paths and provides error handling for corrupted images.

    Parameters
    ----------
    img_size : tuple, default=(128, 128)
        Target size for images (not used in loading, kept for compatibility)
    color_mode : str, default='L'
        Color mode for PIL images ('L' for grayscale, 'RGB' for color)
    validate_paths : bool, default=True
        Whether to validate file paths before loading
    fail_on_error : bool, default=False
        If True, raises exception on loading errors.
        If False, skips invalid images
    verbose : bool, default=True
        Whether to display progress bar and status messages

    Attributes
    ----------
    n_images_loaded_ : int
        Number of successfully loaded images
    failed_images_ : list
        List of file paths that failed to load
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments # noqa
        self,
        img_size: tuple = (128, 128),
        color_mode: str = "L",
        validate_paths: bool = True,
        fail_on_error: bool = False,
        verbose: bool = True,
    ):
        """Initialize the ImageLoader."""
        self.img_size = img_size
        self.color_mode = color_mode
        self.validate_paths = validate_paths
        self.fail_on_error = fail_on_error
        self.verbose = verbose
        self.n_images_loaded_: int = 0
        self.failed_images_: list[str] = []

    def fit(self):
        """Fit the transformer (no-op for image loading)."""
        return self

    def _validate_path(self, path: Union[str, Path]) -> bool:
        """Validate if path exists and is a file."""
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning("Path does not exist: %s", path)
            return False
        if not path_obj.is_file():
            logger.warning("Path is not a file: %s", path)
            return False
        return True

    def _load_single_image(
        self,
        path: Union[str, Path],
    ) -> Optional[Image.Image]:
        """Load a single image from path."""
        try:
            if self.validate_paths and not self._validate_path(path):
                return None

            img = Image.open(path).convert(self.color_mode)
            return img

        except OSError as err:
            logger.error("Failed to load image %s: %s", path, err)
            if self.fail_on_error:
                raise
            return None

        except (ValueError, TypeError) as err:
            logger.error("Unexpected error loading %s: %s", path, err)
            if self.fail_on_error:
                raise
            return None

    def transform(self, data_x: List[Union[str, Path]]) -> List[Image.Image]:
        """Transform file paths into loaded PIL images."""
        if self.verbose:
            logger.info("Loading %d images...", len(data_x))

        images: list[Image.Image] = []
        self.failed_images_ = []

        iterator = tqdm(data_x, desc="Loading images") if self.verbose else data_x

        for path in iterator:
            img = self._load_single_image(path)
            if img is not None:
                images.append(img)
            else:
                self.failed_images_.append(str(path))

        self.n_images_loaded_ = len(images)

        if self.verbose:
            success_rate = (self.n_images_loaded_ / len(data_x)) * 100
            logger.info(
                "Loading completed: %d/%d images loaded successfully (%.1f%%)",
                self.n_images_loaded_,
                len(data_x),
                success_rate,
            )
            if self.failed_images_:
                logger.warning(
                    "Failed to load %d images",
                    len(self.failed_images_),
                )

        if not images:
            raise ValueError("No valid images could be loaded from the provided paths")

        return images
