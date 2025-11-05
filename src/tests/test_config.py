import os

import pytest  # type: ignore

from ds_covid.config import Settings


@pytest.fixture
def settings_instance():
    return Settings()


def test_default_training_settings(settings_instance):
    """Vérifie les valeurs par défaut du training"""
    training = settings_instance.core.training
    assert training.batch_size == 32
    assert training.epochs == 10
    assert training.learning_rate == 0.001
    assert training.validation_split == 0.2
    assert training.random_seed == 42


def test_paths_resolution(settings_instance, tmp_path):
    """Vérifie que les chemins se résolvent correctement"""
    # On override temporairement le project root
    os.environ["PROJECT_ROOT"] = str(tmp_path)
    paths = settings_instance.paths
    assert paths.project_root == tmp_path.resolve()

    # Test des dossiers
    assert paths.data_dir.exists() or isinstance(paths.data_dir, type(
        tmp_path
        ))
    assert paths.models_dir.exists() or isinstance(paths.models_dir, type(
        tmp_path
        ))
    assert paths.results_dir.exists() or isinstance(paths.results_dir, type(
        tmp_path
        ))


def test_load_from_env(settings_instance, monkeypatch):
    """Vérifie le chargement depuis les variables d'environnement"""
    monkeypatch.setenv("BATCH_SIZE", "64")
    monkeypatch.setenv("EPOCHS", "5")
    monkeypatch.setenv("CLASS_NAMES", "A,B,C")
    settings_instance.core.load_from_env()

    assert settings_instance.core.training.batch_size == 64
    assert settings_instance.core.training.epochs == 5
    assert settings_instance.core.data.class_names == ["A", "B", "C"]
