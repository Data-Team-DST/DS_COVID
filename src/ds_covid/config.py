"""
Module de configuration principal du projet ds-covid.
Gère les paramètres d'entraînement, les chemins, la visualisation, etc.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


# =====================================================================
# --- CONFIGS DE BASE -------------------------------------------------
# =====================================================================

@dataclass
class TrainingSettings:
    """Paramètres d'entraînement du modèle."""
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    validation_split: float = 0.2
    random_seed: int = 42
    shuffle: bool = True
    use_gpu: bool = False


@dataclass
class DataSettings:
    """Paramètres liés aux données et images."""
    image_height: int = 224
    image_width: int = 224
    channels: int = 3
    class_names: list[str] = field(
        default_factory=lambda: ["NORMAL", "COVID", "VIRAL"]
    )
    test_split: float = 0.2
    augment_data: bool = True


@dataclass
class MLSettings:
    """Paramètres pour les modèles de Machine Learning."""
    model_name: str = "xgboost_covid"
    test_size: float = 0.2
    random_state: int = 42
    scale_data: bool = True


@dataclass
class DeepLearningSettings:
    """Paramètres pour le Deep Learning."""
    base_model: str = "MobileNetV2"
    dropout_rate: float = 0.3
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])


@dataclass
class VisualizationSettings:
    """Paramètres de visualisation."""
    style: str = "whitegrid"
    figsize: tuple[int, int] = (10, 6)
    palette: str = "deep"


# =====================================================================
# --- CONFIG DES CHEMINS ----------------------------------------------
# =====================================================================

@dataclass
class PathsSettings:
    """Configuration des chemins du projet."""
    _data_dir: Optional[Path] = None
    _models_dir: Optional[Path] = None
    _results_dir: Optional[Path] = None

    @property
    def project_root(self) -> Path:
        """Retourne la racine du projet."""
        project_root = os.getenv('PROJECT_ROOT', '.')
        if project_root == '.':
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                if (current_dir / 'pyproject.toml').exists():
                    return current_dir
                current_dir = current_dir.parent
            return Path.cwd()
        return Path(project_root).expanduser().resolve()

    def _resolve_env_path(self, env_var: str, default_dir: str) -> Path:
        """Résout un chemin à partir d'une variable d'environnement."""
        env_path = os.getenv(env_var)
        if env_path:
            if env_path.startswith('.'):
                return (self.project_root / env_path).resolve()
            return Path(env_path).expanduser().resolve()
        return self.project_root / default_dir

    @property
    def data_dir(self) -> Path:
        """Retourne le chemin du dossier data."""
        if self._data_dir is None:
            self._data_dir = self._resolve_env_path('DATA_DIR', 'data')
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value: Union[str, Path]):
        """Définit le chemin du dossier data."""
        self._data_dir = Path(value).expanduser().resolve()

    @property
    def models_dir(self) -> Path:
        """Retourne le chemin du dossier models."""
        if self._models_dir is None:
            self._models_dir = self._resolve_env_path('MODELS_DIR', 'models')
        return self._models_dir

    @models_dir.setter
    def models_dir(self, value: Union[str, Path]):
        """Définit le chemin du dossier models."""
        self._models_dir = Path(value).expanduser().resolve()

    @property
    def results_dir(self) -> Path:
        """Retourne le chemin du dossier results."""
        if self._results_dir is None:
            self._results_dir = self._resolve_env_path(
                'RESULTS_DIR',
                'results'
                )
        return self._results_dir

    @results_dir.setter
    def results_dir(self, value: Union[str, Path]):
        """Définit le chemin du dossier results."""
        self._results_dir = Path(value).expanduser().resolve()


# =====================================================================
# --- CONFIG CŒUR -----------------------------------------------------
# =====================================================================

@dataclass
class CoreSettings:
    """Configuration centrale (hors chemins)."""
    training: TrainingSettings = field(default_factory=TrainingSettings)
    data: DataSettings = field(default_factory=DataSettings)
    ml: MLSettings = field(default_factory=MLSettings)
    deep_learning: DeepLearningSettings = field(
        default_factory=DeepLearningSettings
        )
    visualization: VisualizationSettings = field(
        default_factory=VisualizationSettings
        )
    verbose: int = 1
    log_level: str = 'INFO'

    def _load_dotenv(self):
        """Charge le fichier .env si présent."""
        def parse_line(line: str):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                return None, None
            key, value = map(str.strip, line.split('=', 1))
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return key, value

        candidates = [
            Path.cwd() / '.env',
            Path.cwd().parent / '.env',
            Path(__file__).resolve().parent.parent.parent / '.env'
        ]
        env_path = next((p for p in candidates if p.exists()), None)

        if not env_path:
            if self.verbose:
                print("⚠️ Aucun fichier .env trouvé")
            return

        try:
            with open(env_path, 'r', encoding='utf-8') as file:
                for line in file:
                    key, value = parse_line(line)
                    if key:
                        os.environ[key] = value
            if self.verbose:
                print(f"📄 Fichier .env chargé: {env_path}")
        except OSError as err:
            print(f"⚠️ Erreur lors du chargement .env: {err}")

    def load_from_env(self):
        """Met à jour la configuration depuis les variables d'environnement."""
        self._load_dotenv()

        def getenv_cast(var_name, cast_type, default):
            value = os.getenv(var_name, str(default))
            try:
                return cast_type(value)
            except ValueError:
                return default

        self.training.batch_size = getenv_cast(
            'BATCH_SIZE', int, self.training.batch_size
            )
        self.training.epochs = getenv_cast('EPOCHS', int, self.training.epochs)
        self.training.learning_rate = getenv_cast(
            'LEARNING_RATE', float, self.training.learning_rate
            )
        self.verbose = getenv_cast('VERBOSE', int, self.verbose)

        class_names = os.getenv('CLASS_NAMES')
        if class_names:
            self.data.class_names = [
                n.strip() for n in class_names.split(',') if n.strip()
                ]


# =====================================================================
# --- CONFIG PRINCIPALE -----------------------------------------------
# =====================================================================

@dataclass
class Settings:
    """Configuration principale du projet ds-covid."""
    core: CoreSettings = field(default_factory=CoreSettings)
    paths: PathsSettings = field(default_factory=PathsSettings)

    def setup_environment(self):
        """Prépare TensorFlow, NumPy, Matplotlib et Seaborn."""
        import warnings  # pylint: disable=import-outside-toplevel
        import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel # noqa
        import tensorflow as tf  # type: ignore # pylint: disable=import-outside-toplevel # noqa
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource] # pylint: disable=import-outside-toplevel # noqa
        import seaborn as sns  # type: ignore # pylint: disable=import-outside-toplevel # noqa

        warnings.filterwarnings('ignore')
        np.random.seed(self.core.training.random_seed)
        random.seed(self.core.training.random_seed)
        tf.random.set_seed(self.core.training.random_seed)

        sns.set_style(self.core.visualization.style)
        plt.rcParams['figure.figsize'] = self.core.visualization.figsize

        print(
            "🎯 Environnement initialisé (seed="
            f" {self.core.training.random_seed})"
            )
        print(f"📦 TensorFlow v{tf.__version__} | NumPy v{np.__version__}")
        print(f"🧠 GPU utilisé : {tf.config.list_physical_devices('GPU')}")

    def __repr__(self) -> str:
        """Affichage simplifié de la configuration."""
        return (
            f"Settings("
            f"batch_size={self.core.training.batch_size}, "
            f"epochs={self.core.training.epochs}, "
            f"learning_rate={self.core.training.learning_rate}, "
            f"data_dir='{self.paths.data_dir}', "
            f"models_dir='{self.paths.models_dir}', "
            f"results_dir='{self.paths.results_dir}')"
        )


# =====================================================================
# --- INSTANCE GLOBALE ------------------------------------------------
# =====================================================================

settings = Settings()
settings.core.load_from_env()
