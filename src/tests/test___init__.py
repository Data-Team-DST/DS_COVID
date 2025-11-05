from ds_covid import Settings, settings


def test_settings_instance():
    """Vérifie que settings est une instance de Settings"""
    assert isinstance(settings, Settings)
    assert settings.paths is not None
    assert settings.core is not None


def test_version_author_email():
    """Vérifie les métadonnées du package"""
    # pas défini dans settings
    assert hasattr(settings, "__version__") is False
    # Si tu veux tester les valeurs du package,
    # tu peux les importer depuis __init__.py global
