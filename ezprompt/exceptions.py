# ezprompt/exceptions.py

"""Custom exception types for the ezprompt library."""


class EzpromptError(Exception):
    """Base class for all ezprompt exceptions."""
    pass


class TemplateError(EzpromptError):
    """Raised when there is an issue processing the prompt template."""
    pass


class ValidationError(EzpromptError):
    """Raised when input validation fails against the template."""
    pass


class ModelError(EzpromptError):
    """Raised for issues related to the specified model (e.g., not found, API error)."""
    pass


class ContextLengthError(ModelError):
    """Raised when the rendered prompt exceeds the model's context length."""
    def __init__(self, message, prompt_length, max_length, suggested_models=None):
        super().__init__(message)
        self.prompt_length = prompt_length
        self.max_length = max_length
        self.suggested_models = suggested_models or [] 