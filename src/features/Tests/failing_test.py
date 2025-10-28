import pytest

@pytest.mark.skip(reason="Test volontairement désactivé pendant la phase d'intégration CI/CD")
def test_failing():
    assert 1 + 1 == 3
