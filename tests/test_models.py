import pytest
from ez_prompt.models import get_model_info


def test_get_mdeol():
    """Test the get_openai_models function."""
    model = get_model_info("gpt-4o")
    assert model is not None
    assert model.id == "gpt-4o"
    assert model.context_length == 128000
    assert model.pricing_in == 2.50
    assert model.pricing_out == 10.00
    assert model.max_output_tokens == 16384
