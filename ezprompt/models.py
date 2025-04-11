"""
Functions for retrieving and managing LLM model information.
"""

import os
from datetime import datetime
from ezprompt.exceptions import EZPromptError
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# Load the dotenv file
load_dotenv(f"{DIR_PATH}/../.env")


@dataclass
class ModelInfo:
    """Data class containing standardized model information."""

    id: str
    context_length: int
    pricing_in: Optional[float] = None
    pricing_out: Optional[float] = None
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    provider: Optional[str] = None
    version: Optional[str] = None
    is_deprecated: bool = False
    max_output_tokens: Optional[int] = None
    call_cost: Optional[float] = None


# WARNING: This list is not up to date.
# Up to date as of: 2025-04-11
LAST_UPDATED = datetime(2025, 4, 11)
MODEL_INFO = [
    ModelInfo(
        id="o1",
        context_length=200_000,
        pricing_in=15.0,
        pricing_out=60.0,
        max_output_tokens=100_000,
        call_cost=0.0,
        description="High-intelligence reasoning model designed for complex problem-solving tasks.",
        capabilities=[
            "Advanced reasoning",
            "Mathematics",
            "Coding",
            "Scientific analysis",
        ],
        version="o1",
    ),
    ModelInfo(
        id="o1-mini",
        context_length=128000,
        pricing_in=1.1,
        pricing_out=4.4,
        max_output_tokens=65_536,
        call_cost=0.0,
        description="Faster, more affordable reasoning model than o1, optimized for efficiency.",
        capabilities=["Reasoning", "Mathematics", "Coding"],
        version="o1",
    ),
    ModelInfo(
        id="o3-mini",
        context_length=200_000,
        pricing_in=1.1,
        pricing_out=4.4,
        max_output_tokens=100_000,
        call_cost=0.0,
        description="Cost-efficient reasoning model in the o-series, delivering strong performance in STEM tasks.",
        capabilities=["Reasoning", "Mathematics", "Coding"],
        version="o3",
    ),
    ModelInfo(
        id="gpt-3.5-turbo",
        context_length=16384,
        pricing_in=0.5,
        pricing_out=1.5,
        max_output_tokens=4096,
        call_cost=0.0,
        description="Optimized for chat applications; cost-effective and widely used.",
        capabilities=["Text generation", "Conversational AI"],
        version="3.5",
    ),
    ModelInfo(
        id="gpt-4o",
        context_length=128000,
        pricing_in=2.50,
        pricing_out=10.00,
        max_output_tokens=16384,
        call_cost=0.0,
        description="High-intelligence model for complex tasks with extensive context handling.",
        capabilities=[
            "Text generation",
            "Conversational AI",
            "Multimodal inputs",
        ],
        version="4o",
    ),
    ModelInfo(
        id="gpt-4o-mini",
        context_length=128_000,
        pricing_in=0.15,
        pricing_out=0.60,
        max_output_tokens=16_384,
        call_cost=0.0,
        description="Affordable small model for fast, everyday tasks.",
        capabilities=["Text generation", "Conversational AI"],
        version="4o",
    ),
    ModelInfo(
        id="gpt-4.5",
        context_length=128000,
        pricing_in=75.00,
        pricing_out=150.00,
        max_output_tokens=16384,
        call_cost=0.0,
        description="Largest GPT model designed for creative tasks and agentic planning.",
        capabilities=[
            "Text generation",
            "Conversational AI",
            "Creative writing",
        ],
        version="4.5",
    ),
    ModelInfo(
        id="gpt-4o-search-preview",
        context_length=128000,
        max_output_tokens=16384,
        pricing_in=2.50,
        pricing_out=10.00,
        call_cost=0.03,
        description="GPT model that relies on search to answer questions.",
        capabilities=["Understand and execute web queries"],
        version="4.5",
    ),
]


async def get_model_info(model_id: str) -> ModelInfo:
    """
    Gets information about a specific model, using cached data if
    available.

    Parameters
    ----------
    model_id : str
        ID of the model to look up

    Returns
    -------
    ModelInfo
        Model information

    Raises
    ------
    EZPromptError
        If the model is not found
    """
    for model in MODEL_INFO:
        if model.id == model_id:
            return model
    raise EZPromptError(f"Model '{model_id}' not found")
