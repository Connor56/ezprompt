import pytest
from ezprompt.models import _fetch_openai_models


def test_get_openai_models():
    """Test the get_openai_models function."""
    models = _fetch_openai_models()
    assert models is not None
    assert "id" in models
    assert "object" in models
