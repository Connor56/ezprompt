import pytest
from ezprompt.data import (
    PromptOutcome,
    save_outcome,
    CACHE_DIR,
    process_completion,
)
from ezprompt.models import ModelInfo, get_model_info
import os
import json
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import (
    CompletionUsage,
    CompletionTokensDetails,
    PromptTokensDetails,
)


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


@pytest.fixture
def test_model_info():
    return get_model_info("test_model")


@pytest.fixture
def test_completion():
    return ChatCompletion(
        id="chatcmpl-BLarZrpR7V3PRzRExUyHLl5b1QoGU",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="The Battle of Agincourt took place on October 25, 1415, and the weather is often noted for having a significant impact on the outcome. Historical accounts suggest that it had been raining heavily in the days leading up to the battle, making the ground muddy and difficult to traverse. This muddy terrain played a crucial role, as it hindered the heavily armored French knights, putting them at a disadvantage against the English forces led by Henry V, who utilized longbowmen to great effect.",
                    refusal=None,
                    role="assistant",
                    annotations=[],
                    audio=None,
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
        created=1744486289,
        model="gpt-4o-2024-08-06",
        object="chat.completion",
        service_tier="default",
        system_fingerprint="fp_432e014d75",
        usage=CompletionUsage(
            completion_tokens=101,
            prompt_tokens=25,
            total_tokens=126,
            completion_tokens_details=CompletionTokensDetails(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=0,
                rejected_prediction_tokens=0,
            ),
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=0, cached_tokens=0
            ),
        ),
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
