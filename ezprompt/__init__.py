"""
ezprompt: An easy-to-use library for creating and sending prompts to LLMs.

Created by:
    - Connor Skelland
    - https://github.com/connor56
    - 2025-04-10
"""

# Import key components to make them available at the package level
from .prompt import Prompt
from .exceptions import (  # noqa: F401
    EZPromptError,
    TemplateError,
    ValidationError,
    ModelError,
    ContextLengthError,
)
from .warnings import (  # noqa: F401
    EZPromptWarning,
    UnusedInputWarning,
)

__all__ = [
    "Prompt",
    "EZPromptError",
    "TemplateError",
    "ValidationError",
    "ModelError",
    "ContextLengthError",
    "EZPromptWarning",
    "UnusedInputWarning",
    "all_models",
    "get_model_info",
]
