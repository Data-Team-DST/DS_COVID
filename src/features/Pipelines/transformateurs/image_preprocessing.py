"""Image preprocessing transformers for the data pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from tqdm import tqdm  # type: ignore

# Logger
logger = logging.getLogger(__name__)

# Pillow resampling compatibility
try:
    resampling = Image.resampling  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    resampling = Image

RESAMPLE_LANCZOS = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
resample_nearest = getattr(resampling, "NEAREST", getattr(Image, "NEAREST", 0))


class ImageResizer(BaseEstimator, TransformerMixin):
    """Resize images to a target size."""

    def __init__(
        self,
        img_size: tuple[int, int] = (256, 256),
        resample: int = RESAMPLE_LANCZOS,
        preserve_aspect_ratio: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the ImageResizer.

        Args:
            img_size: Target size (width, height) for resized images.
            resample: Resampling filter to use.
            preserve_aspect_ratio: Whether to preserve the aspect ratio.
            verbose: Whether to log processing information.
        """
        self.img_size = img_size
        self.resample = resample
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.verbose = verbose
        self.original_shapes_: list[tuple[int, ...]] = []
        self.n_images_processed_: int = 0

    def fit(self, *_: Any) -> ImageResizer:
        """No-op fit to keep sklearn API compatible."""
        return self

    def _resize_with_aspect_ratio(self, img: Image.Image) -> Image.Image:
        """Resize an image while preserving its aspect ratio."""
        ratio = min(
            self.img_size[0] / img.size[0],
            self.img_size[1] / img.size[1],
        )
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img_resized = img.resize(new_size, self.resample)

        new_img = Image.new(img.mode, self.img_size, color=0)
        paste_x = (self.img_size[0] - new_size[0]) // 2
        paste_y = (self.img_size[1] - new_size[1]) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        return new_img

    def transform(self, data_x: list[Image.Image | np.ndarray]) -> np.ndarray:
        """
        Resize a list of images to the target size.

        Args:
            data_x: List of PIL Images or NumPy arrays.

        Returns:
            NumPy array of resized images.
        """
        if self.verbose:
            logger.info("Resizing %d images to %s", len(data_x), self.img_size)

        self.original_shapes_ = []
        resized: list[np.ndarray] = []
        iterator = tqdm(
            data_x,
            desc="Resizing images"
            ) if self.verbose else data_x

        for img in iterator:
            if isinstance(img, np.ndarray):
                self.original_shapes_.append(img.shape)
                img_obj = Image.fromarray(img.astype(np.uint8))
            else:
                self.original_shapes_.append(img.size)
                img_obj = img

            img_resized = (
                self._resize_with_aspect_ratio(img_obj)
                if self.preserve_aspect_ratio
                else img_obj.resize(self.img_size, self.resample)
            )
            resized.append(np.array(img_resized))

        self.n_images_processed_ = len(resized)
        if self.verbose:
            logger.info(
                "Resizing completed: %d images processed",
                self.n_images_processed_
                )
        return np.array(resized)


class ImageNormalizer(BaseEstimator, TransformerMixin):
    """Normalize image pixel values."""

    def __init__(
        self,
        method: str = "minmax",
        feature_range: tuple[float, float] = (0.0, 1.0),
        per_image: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the ImageNormalizer.

        Args:
            method: Normalization method ('minmax', 'standard', 'custom').
            feature_range: Range for 'minmax' or 'custom' normalization.
            per_image: Normalize each image individually if True.
            verbose: Whether to log processing information.
        """
        self.method = method.lower()
        self.feature_range = feature_range
        self.per_image = per_image
        self.verbose = verbose
        self._globals: dict[
            str,
            float | None
            ] = {"min": None, "max": None, "mean": None, "std": None}

    def fit(
            self,
            data_x: list[np.ndarray] | np.ndarray, *_: Any
            ) -> ImageNormalizer:
        """Fit normalizer on data (compute global stats if needed)."""
        if not self.per_image:
            arr = np.array(data_x, dtype=np.float32)
            if self.method in ("minmax", "custom"):
                self._globals["min"] = float(arr.min())
                self._globals["max"] = float(arr.max())
            elif self.method == "standard":
                self._globals["mean"] = float(arr.mean())
                self._globals["std"] = float(arr.std())

            if self.verbose:
                logger.info("Fitted normalizer with method %s", self.method)
        return self

    def transform(
            self,
            data_x: list[np.ndarray] | np.ndarray, *_: Any
            ) -> np.ndarray:
        """
        Apply normalization to images.

        Args:
            data_x: List or array of images.

        Returns:
            Normalized images as NumPy array.
        """
        if self.verbose:
            logger.info(
                "Normalizing %d images using '%s' method",
                len(data_x), self.method
                )
        arr = np.array(data_x, dtype=np.float32)

        if self.method == "minmax":
            result = self._normalize_minmax(arr)
        elif self.method == "standard":
            result = self._normalize_standard(arr)
        elif self.method == "custom":
            result = self._normalize_custom(arr)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        if self.verbose:
            logger.info(
                "Normalization completed. Output range: [%.2f, %.2f]",
                float(result.min()),
                float(result.max())
                )
        return result

    def _normalize_minmax(self, arr: np.ndarray) -> np.ndarray:
        """Apply min-max normalization."""
        if self.per_image:
            return np.array([self._minmax_image(img) for img in arr])
        gmin, gmax = self._globals["min"], self._globals["max"]
        if gmin is None or gmax is None:
            raise RuntimeError("Normalizer not fitted for global min/max")
        return (arr - gmin) / (gmax - gmin)

    def _normalize_standard(self, arr: np.ndarray) -> np.ndarray:
        """Apply standard score normalization."""
        if self.per_image:
            return np.array([self._standardize_image(img) for img in arr])
        gmean, gstd = self._globals["mean"], self._globals["std"]
        if gmean is None or gstd is None:
            raise RuntimeError("Normalizer not fitted for global mean/std")
        return (arr - gmean) / (gstd + 1e-8)

    def _normalize_custom(self, arr: np.ndarray) -> np.ndarray:
        """Apply custom range normalization."""
        fr_min, fr_max = self.feature_range
        if self.per_image:
            return np.array(
                [self._custom_image(img, fr_min, fr_max) for img in arr]
                )
        gmin, gmax = self._globals["min"], self._globals["max"]
        if gmin is None or gmax is None:
            raise RuntimeError("Normalizer not fitted for global min/max")
        normalized = (arr - gmin) / (gmax - gmin)
        return normalized * (fr_max - fr_min) + fr_min

    @staticmethod
    def _minmax_image(img: np.ndarray) -> np.ndarray:
        """Normalize single image with min-max."""
        img_min, img_max = img.min(), img.max()
        return (
            img - img_min
            ) / (img_max - img_min) if img_max > img_min else img

    @staticmethod
    def _standardize_image(img: np.ndarray) -> np.ndarray:
        """Standardize single image to zero mean and unit variance."""
        img_mean, img_std = img.mean(), img.std()
        return (img - img_mean) / img_std if img_std > 0 else img - img_mean

    @staticmethod
    def _custom_image(
        img: np.ndarray,
        fr_min: float,
        fr_max: float
         ) -> np.ndarray:
        """Normalize single image to a custom range."""
        img_min, img_max = img.min(), img.max()
        if img_max <= img_min:
            return img
        normalized = (img - img_min) / (img_max - img_min)
        return normalized * (fr_max - fr_min) + fr_min


@dataclass
class MaskerConfig:
    """Configuration dataclass for ImageMasker."""
    mask_threshold: float = 0.5
    resize_masks: bool = True
    invert_mask: bool = False
    verbose: bool = True


class ImageMasker(BaseEstimator, TransformerMixin):
    """Apply binary masks to images."""

    def __init__(
            self, mask_paths: list[str],
            config: MaskerConfig = MaskerConfig()
            ):
        """
        Initialize the ImageMasker.

        Args:
            mask_paths: List of file paths to mask images.
            config: MaskerConfig dataclass with optional parameters:
                mask_threshold, resize_masks, invert_mask, verbose
        """
        self.mask_paths = mask_paths
        self.mask_threshold = config.mask_threshold
        self.resize_masks = config.resize_masks
        self.invert_mask = config.invert_mask
        self.verbose = config.verbose
        self.n_images_masked_: int = 0

    def fit(self, *_):
        """No-op fit to keep sklearn API compatible."""
        if not self.mask_paths:
            logger.warning("No mask paths provided")
        return self

    def transform(self, data_x: np.ndarray) -> np.ndarray:
        """
        Apply binary masks to a list of images.

        Args:
            data_x: Array of images to mask.

        Returns:
            Masked images as NumPy array.
        """
        if len(self.mask_paths) != len(data_x):
            raise ValueError(
                f"Number of masks ({len(self.mask_paths)})"
                f" must match number of images ({len(data_x)})"
            )
        if self.verbose:
            logger.info("Applying masks to %d images", len(data_x))

        masked = []
        iterator = zip(data_x, self.mask_paths)
        if self.verbose:
            iterator = tqdm(list(iterator), desc="Applying masks")

        for img, mask_path in iterator:
            try:
                mask = Image.open(mask_path).convert("L")
                if self.resize_masks:
                    target_size = (img.shape[1], img.shape[0])
                    mask = mask.resize(target_size, resample_nearest)

                mask_arr = (np.array(mask) / 255.0) > self.mask_threshold
                if self.invert_mask:
                    mask_arr = ~mask_arr
                if img.ndim == 3:
                    mask_arr = mask_arr[:, :, np.newaxis]
                masked.append(img * mask_arr)
            except (OSError, ValueError) as err:
                logger.error("Failed to apply mask %s: %s", mask_path, err)
                masked.append(img)

        self.n_images_masked_ = len(masked)
        if self.verbose:
            logger.info(
                "Masking completed: %d images processed",
                self.n_images_masked_
                )
        return np.array(masked)


class ImageFlattener(BaseEstimator, TransformerMixin):
    """Flatten images for ML models."""

    def __init__(self, order: str = "C", verbose: bool = True):
        """
        Initialize the ImageFlattener.

        Args:
            order: Memory order for flattening ('C' or 'F').
            verbose: Whether to log processing information.
        """
        self.order = order
        self.verbose = verbose
        self.original_shape_: tuple[int, ...] = ()
        self.n_features_: int = 0

    def fit(self, data_x, *_):
        """
        Fit flattener by storing original shape.

        Args:
            data_x: Array of images.

        Returns:
            Self.
        """
        arr = np.array(data_x)
        self.original_shape_ = arr.shape[1:]
        self.n_features_ = int(np.prod(self.original_shape_))
        if self.verbose:
            logger.info(
                "Fitted flattener: %s -> %d features",
                self.original_shape_,
                self.n_features_
                )
        return self

    def transform(self, data_x: np.ndarray) -> np.ndarray:
        """
        Flatten images into 2D array.

        Args:
            data_x: Array of images.

        Returns:
            Flattened images.
        """
        if self.verbose:
            logger.info("Flattening %d images...", len(data_x))
        arr = np.array(data_x)
        n_samples = arr.shape[0]
        data_flat = arr.reshape((n_samples, -1), order=self.order)
        if self.verbose:
            logger.info(
                "Flattening completed: %s -> %s", arr.shape,
                data_flat.shape
                )
        return data_flat

    def inverse_transform(self, data_x: np.ndarray) -> np.ndarray:
        """
        Restore flattened images to original shape.

        Args:
            data_x: Flattened images.

        Returns:
            Images reshaped to original dimensions.
        """
        if not self.original_shape_:
            raise RuntimeError(
                "Transformer must be fitted before inverse_transform"
                )
        n_samples = data_x.shape[0]
        target_shape = (n_samples,) + self.original_shape_
        return data_x.reshape(target_shape, order=self.order)


class ImageBinarizer(BaseEstimator, TransformerMixin):
    """Binarize images using threshold."""

    def __init__(
        self,
        threshold: float | str = 0.5,
        invert: bool = False,
        output_dtype=np.float32,
        verbose: bool = True,
    ):
        """
        Initialize the ImageBinarizer.

        Args:
            threshold: Threshold value or method ('mean', 'median', 'otsu').
            invert: Invert the binarization.
            output_dtype: Output data type of binarized images.
            verbose: Whether to log processing information.
        """
        self.threshold = threshold
        self.invert = invert
        self.output_dtype = output_dtype
        self.verbose = verbose
        self.threshold_value_: float | None = None

    def fit(self, data_x, *_):
        """
        Fit the binarizer (compute threshold if needed).

        Args:
            data_x: Array of images.

        Returns:
            Self.
        """
        arr = np.array(data_x)
        if isinstance(self.threshold, str):
            self.threshold_value_ = self._compute_threshold(arr)
        else:
            self.threshold_value_ = float(self.threshold)
        if self.verbose:
            logger.info(
                "Fitted binarizer with threshold: %.4f",
                self.threshold_value_
                )
        return self

    def _compute_threshold(self, arr: np.ndarray) -> float:
        """Compute adaptive threshold based on method."""
        method = str(self.threshold).lower()
        if method == "mean":
            return float(arr.mean())
        if method == "median":
            return float(np.median(arr))
        if method == "otsu":
            return self._compute_otsu_threshold(arr)
        raise ValueError(f"Unknown threshold method: {self.threshold}")

    @staticmethod
    def _compute_otsu_threshold(arr: np.ndarray) -> float:
        """Compute Otsu's threshold for an image."""
        hist, bins = np.histogram(arr.flatten(), bins=256)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        max_var = 0.0
        threshold = float(bin_centers[0])
        for t_idx in range(1, len(hist) - 1):
            w0 = hist[:t_idx].sum()
            w1 = hist[t_idx:].sum()
            if w0 == 0 or w1 == 0:
                continue
            mu0 = (hist[:t_idx] * bin_centers[:t_idx]).sum() / w0
            mu1 = (hist[t_idx:] * bin_centers[t_idx:]).sum() / w1
            var = w0 * w1 * (mu0 - mu1) ** 2
            if var > max_var:
                max_var = var
                threshold = float(bin_centers[t_idx])
        return threshold

    def transform(self, data_x, *_):
        """
        Apply binarization to images.

        Args:
            data_x: Array of images.

        Returns:
            Binarized images as NumPy array.
        """
        if self.threshold_value_ is None:
            if isinstance(self.threshold, str):
                raise RuntimeError(
                    "Transformer must be fitted for adaptive thresholding"
                    )
            self.threshold_value_ = float(self.threshold)

        if self.verbose:
            logger.info(
                "Binarizing %d images with threshold %.4f",
                len(data_x),
                self.threshold_value_
                )

        arr = np.array(data_x)
        binarized = (
            (arr <= self.threshold_value_).astype(self.output_dtype)
            if self.invert
            else (arr > self.threshold_value_).astype(self.output_dtype)
        )

        if self.verbose:
            positive_ratio = float((binarized == 1).mean() * 100)
            logger.info(
                "Binarization completed. Positive pixels: %.1f%%",
                positive_ratio
                )

        return binarized
