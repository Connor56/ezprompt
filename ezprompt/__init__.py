"""
ezprompt: An easy-to-use library for creating and sending prompts to LLMs.

Created by:
    - Connor Skelland
    - https://github.com/connor56
    - 2025-04-10
"""

# Import key components to make them available at the package level
from .prompt import Prompt
from .exceptions import ( # noqa: F401
    EzpromptError,
    TemplateError,
    ValidationError,
    ModelError,
    ContextLengthError,
)

__all__ = [
    "Prompt",
    "EzpromptError",
    "TemplateError",
    "ValidationError",
    "ModelError",
    "ContextLengthError",
] 