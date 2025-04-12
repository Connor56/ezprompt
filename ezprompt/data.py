"""
Processes and saves data from the prompt information page
"""

from platformdirs import PlatformDirs
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional
from openai.types.chat import ChatCompletion

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


def process_completion(completion: ChatCompletion) -> PromptOutcome:
    """Processes the openai chat completion and returns an outcome
    object."""
    pass
