import sys
from ds_covid import notebook_utils as nb


def test_setup_notebook_imports(tmp_path, monkeypatch):
    """Vérifie l'ajout de src/ au path"""
    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    monkeypatch.chdir(notebooks_dir)

    project_root, src_path = nb.setup_notebook_imports()
    assert str(src_path) in sys.path
    assert project_root.exists()


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
        "models_dir", "notebooks_dir", "results_dir", "reports_dir"
    ]
    for key in required_keys:
        assert key in paths
