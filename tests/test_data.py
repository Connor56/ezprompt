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


def test_save_outcome_multiple_times(test_outcome):
    """Test that saving the same prompt outcome twice works correctly"""
    # Save the outcome twice with the same name and template hash
    save_outcome(test_outcome, "test_prompt_multiple", "test_hash")
    save_outcome(test_outcome, "test_prompt_multiple", "test_hash")

    expected_file = f"{CACHE_DIR}/test_prompt_multiple_test_hash.json"

    assert os.path.exists(expected_file)

    with open(expected_file, "r") as f:
        outcomes = json.load(f)

    # Delete the file after reading it
    os.remove(expected_file)

    # Check that we have two identical outcomes in the list
    assert len(outcomes) == 2

    expected_outcome = {
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

    assert outcomes[0] == expected_outcome
    assert outcomes[1] == expected_outcome
