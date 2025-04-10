"""Functions for retrieving and managing LLM model information."""

import requests
import json
from typing import Dict, Any, Optional
import os
from datetime import datetime, timedelta

# Cache model info for 24 hours
_model_cache: Dict[str, Any] = {}
_last_update: Optional[datetime] = None
_CACHE_DURATION = timedelta(hours=24)

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelInfo:
    """Data class containing standardized model information."""

    id: str
    name: str
    context_length: int
    pricing: Optional[float] = None
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    provider: Optional[str] = None
    version: Optional[str] = None
    is_deprecated: bool = False


def _fetch_openai_models() -> List[ModelInfo]:
    """
    Fetches the latest model information from OpenAI's API.

    Returns
    -------
    List[ModelInfo]
        List of ModelInfo objects containing model information from
        OpenAI

    Raises
    ------
    Exception
        If there's an error fetching the model data
    """
    print("Fetching OpenAI models...")
    response = requests.get("https://api.openai.com/v1/models")
    print(response.json())


async def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Gets information about a specific model, using cached data if
    available.

    Parameters
    ----------
    model_name : str
        Name of the model to look up

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model information

    Raises
    ------
    ValueError
        If the model is not found
    """
    # TODO: Implement code that picks up model info objects from file.
    pass
    # global _model_cache, _last_update

    # # Check if we need to refresh the cache
    # if (
    #     not _last_update
    #     or datetime.now() - _last_update > _CACHE_DURATION
    #     or not _model_cache
    # ):
    #     try:
    #         _model_cache = await _fetch_openai_models()
    #         _last_update = datetime.now()
    #     except Exception as e:
    #         # If fetch fails and we have cached data, use it
    #         if _model_cache:
    #             pass
    #         else:
    #             raise e

    # if model_name not in _model_cache:
    #     raise ValueError(f"Model '{model_name}' not found")

    # return _model_cache[model_name]
