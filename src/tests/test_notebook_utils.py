import sys
from pathlib import Path
from ds_covid import notebook_utils as nb


def test_setup_notebook_imports(tmp_path, monkeypatch):
    """Vérifie l'ajout de src/ au path, ou la gestion propre si non trouvé."""
    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    monkeypatch.chdir(notebooks_dir)

    _, src_path = nb.setup_notebook_imports()

    # Le src peut ne pas exister (comme en CI)
    if src_path.exists():
        assert str(src_path) in sys.path
    else:
        # On vérifie que la fonction retourne bien un Path et gère proprement
        assert isinstance(src_path, Path)
        assert str(src_path) not in sys.path


def test_simple_notebook_setup(tmp_path, monkeypatch):
    """Teste le setup simple"""
    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    monkeypatch.chdir(notebooks_dir)

    project_root = nb.simple_notebook_setup(verbose=False)
    assert project_root.exists()


def test_get_project_paths(tmp_path, monkeypatch):
    """Vérifie la génération des chemins"""
    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    monkeypatch.chdir(notebooks_dir)

    paths = nb.get_project_paths()
    required_keys = [
        "project_root", "data_dir", "raw_data", "covid_data",
        "models_dir", "notebooks_dir", "results_dir", "reports_dir",
    ]
    for key in required_keys:
        assert key in paths
