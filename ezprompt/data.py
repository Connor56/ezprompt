"""
Processes and saves data from the prompt information page
"""

from platformdirs import PlatformDirs
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from openai.types.chat import ChatCompletion
from ezprompt.models import ModelInfo
import numpy as np

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
    time_taken: float
    # The text that was input and the response that was generated
    # Optional, as potentially useful for analysis
    prompt: Optional[str] = None
    response: Optional[str] = None


@dataclass
class Centile:
    centile: float
    cost: float


@dataclass
class PromptStatistics:
    total_outcomes: int
    mean_cost: float
    centiles: List[Centile]
    min_cost: float
    max_cost: float
    input_tokens_vs_cost: List[Tuple[int, float]]


def build_cache():
    if not os.path.exists(CACHE_DIR):
        print("hi")
        print(f"Create ezprompt cache directory: {CACHE_DIR}")
        os.makedirs(CACHE_DIR, exist_ok=True)


def save_outcome(
    outcome: PromptOutcome,
    prompt_name: str,
    template_hash: str,
    model_id: str,
):
    """Saves the prompt outcome to a json file in the cache directory"""
    build_cache()
    # Check file exists
    file_path = f"{CACHE_DIR}/{prompt_name}_{model_id}_{template_hash}.json"
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
    time_taken: float,
) -> PromptOutcome:
    """Processes the openai chat completion and returns an outcome
    object."""
    # get the tokens
    input_tokens = response.usage.prompt_tokens
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    output_tokens = response.usage.completion_tokens - reasoning_tokens

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
        time_taken=time_taken,
    )


def load_cache(
    prompt_name: str,
    template_hash: str,
    model_id: str,
) -> List[PromptOutcome]:
    """Loads the prompt outcomes from the cache"""
    file_path = f"{CACHE_DIR}/{prompt_name}_{model_id}_{template_hash}.json"
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r") as f:
        data = json.load(f)

    # TODO: this likely needs better error handling
    cached_data = []
    for item in data:
        cached_data.append(PromptOutcome(**item))

    return cached_data


def get_statistics(
    outcomes: List[PromptOutcome],
) -> PromptStatistics:
    """Gets statistical measures of the cost of a prompt"""
    costs = [outcome.total_cost for outcome in outcomes]

    mean_cost = float(np.mean(costs))

    # Get the centiles
    boundaries = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    centile_values = np.percentile(costs, boundaries).tolist()
    centiles = [
        Centile(boundary, value)
        for boundary, value in zip(boundaries, centile_values)
    ]

    min_cost = min(costs)
    max_cost = max(costs)

    input_tokens_vs_cost = [
        (outcome.input_tokens, outcome.total_cost) for outcome in outcomes
    ]

    return PromptStatistics(
        total_outcomes=len(outcomes),
        mean_cost=mean_cost,
        centiles=centiles,
        min_cost=min_cost,
        max_cost=max_cost,
        input_tokens_vs_cost=input_tokens_vs_cost,
    )
