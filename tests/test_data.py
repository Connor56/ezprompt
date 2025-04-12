import pytest
from ezprompt.data import PromptOutcome, save_outcome, CACHE_DIR
import os
import json


@pytest.fixture
def test_outcome():
    return PromptOutcome(
        input_cost=0.0,
        reasoning_cost=0.0,
        output_cost=0.0,
        tool_cost=0.0,
        total_cost=0.0,
        input_tokens=13,
        reasoning_tokens=234,
        output_tokens=22,
        input="Is the sky always blue?",
        response="No, the sky is not always blue. It depends on what angle the sun is hitting it.",
    )


def test_save_outcome(test_outcome):
    """Test that data is correctly saved in the cache directory"""
    save_outcome(test_outcome, "test_prompt", "test_template_hash")

    expected_file = f"{CACHE_DIR}/test_prompt_test_template_hash.json"

    assert os.path.exists(f"{CACHE_DIR}/test_prompt_test_template_hash.json")

    with open(expected_file, "r") as f:
        outcome = json.load(f)

    # Delete the file after reading it
    os.remove(expected_file)

    expected_outcome = [
        {
            "input": "Is the sky always blue?",
            "response": "No, the sky is not always blue. It depends on what angle the sun is hitting it.",
            "input_cost": 0.0,
            "reasoning_cost": 0.0,
            "output_cost": 0.0,
            "tool_cost": 0.0,
            "total_cost": 0.0,
            "input_tokens": 13,
            "reasoning_tokens": 234,
            "output_tokens": 22,
        }
    ]

    assert outcome == expected_outcome
