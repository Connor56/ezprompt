"""
Processes and saves data from the prompt information page
"""

from platformdirs import PlatformDirs
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional
from openai.types.chat import ChatCompletion
from ezprompt.models import ModelInfo

# Set up the cache
CACHE_DIR = PlatformDirs("ezprompt", "").user_data_dir


@dataclass
class PromptOutcome:
    input_cost: float
    reasoning_cost: float
    output_cost: float
    tool_cost: float
    total_cost: float
    input_tokens: int
    reasoning_tokens: int
    output_tokens: int
    model: str
    # The text that was input and the response that was generated
    # Optional, as potentially useful for analysis
    input: Optional[str] = None
    response: Optional[str] = None


def build_cache():
    if not os.path.exists(CACHE_DIR):
        print("hi")
        print(f"Create ezprompt cache directory: {CACHE_DIR}")
        os.makedirs(CACHE_DIR, exist_ok=True)


def save_outcome(outcome: PromptOutcome, prompt_name: str, template_hash: str):
    """Saves the prompt outcome to a json file in the cache directory"""
    build_cache()
    # Check file exists
    file_path = f"{CACHE_DIR}/{prompt_name}_{template_hash}.json"
    exists = os.path.exists(file_path)

    if exists:
        # Load the existing data
        with open(file_path, "r") as f:
            data = json.load(f)

        # Append the new outcome
        data.append(asdict(outcome))

        # Save the updated data
        with open(file_path, "w") as f:
            json.dump(data, f)
    else:
        # Create a new file
        with open(file_path, "w") as f:
            json.dump([asdict(outcome)], f)


def process_response(
    response: ChatCompletion,
    model_info: ModelInfo,
) -> PromptOutcome:
    """Processes the openai chat completion and returns an outcome
    object."""
    # get the tokens
    input_tokens = response.usage.prompt_tokens
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    output_tokens = response.usage.completion_tokens

    # get per token pricing
    pricing_in = model_info.pricing_in / 1_000_000
    pricing_out = model_info.pricing_out / 1_000_000

    # calculate the costs
    input_cost = input_tokens * pricing_in
    reasoning_cost = reasoning_tokens * pricing_out
    output_cost = output_tokens * pricing_out
    total_cost = (
        input_cost + reasoning_cost + output_cost + model_info.call_cost
    )

    # return the outcome
    return PromptOutcome(
        input_cost=input_cost,
        reasoning_cost=reasoning_cost,
        output_cost=output_cost,
        tool_cost=model_info.call_cost,
        total_cost=total_cost,
        input_tokens=input_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        model=model_info.id,
    )
