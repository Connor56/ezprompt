# ezprompt/prompt.py

"""The main Prompt class for ezprompt."""

import jinja2
import asyncio
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from .models import get_model_info

# Custom exceptions and warnings
from .exceptions import (
    TemplateError,
    ValidationError,
    ModelError,
    ContextLengthError,
)
from .warnings import UnusedInputWarning


# Define the Abstract Base Class
class BasePrompt(ABC):
    """
    Abstract base class defining the interface for prompt objects.

    Use this when you're building your own custom prompts to stay
    in line with the EZPrompt API.
    """

    @abstractmethod
    def __init__(
        self,
        template: str,
        inputs: Dict[str, Any],
        model: str,
        api_key: str,
    ):
        """Initialize the prompt object."""
        pass

    @abstractmethod
    def _render_prompt(self) -> str:
        """Render the prompt template with inputs."""
        pass

    @abstractmethod
    def check(self) -> Tuple[Optional[List[str]], Optional[float]]:
        """Perform validation checks and estimate cost."""
        pass

    @abstractmethod
    async def send(self, **kwargs) -> Any:
        """Send the prompt to the LLM."""
        pass


# Make Prompt implement the BasePrompt interface
class Prompt(BasePrompt):
    """Represents a prompt using Jinja2 templating to be sent to an LLM."""

    ):
        """
        Initialize the Prompt object.

        Parameters
        ----------
        template : str
            The prompt template string (can use Jinja2 syntax).
        inputs : dict
            A dictionary of input variables for the template.
        model : str
            The name of the target LLM (e.g., 'gpt-3.5-turbo').
        """
        self.template = template
        self.inputs = inputs
        self.model = model
        self._rendered_prompt: Optional[str] = None
        self._input_tokens: Optional[int] = None
        self._model_info: Optional[Dict[str, Any]] = None

        # Initialize Jinja environment
        # StrictUndefined: Enforces perfect adherence to the template
        self._jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined)

        # Check template can be parsed
        try:
            self._parsed_template = self._jinja_env.parse(template)
        except jinja2.TemplateSyntaxError as e:
            raise TemplateError(f"Syntax error in template: {e}") from e

        self._validate_inputs()

    def _validate_inputs(self):
        """
        Checks if all the prompt template variables are present in
        the provided inputs.
        """
        required_vars = jinja2.meta.find_undeclared_variables(
            self._parsed_template
        )

        missing_vars = required_vars - set(self.inputs.keys())

        if missing_vars:
            raise ValidationError(
                f"Missing required inputs for template: {', '.join(missing_vars)}"
            )

        # Check for unused inputs and add warning
        unused_vars = set(self.inputs.keys()) - required_vars

        if unused_vars:
            warnings.warn(
                f"Some provided inputs are not used in template: {', '.join(unused_vars)}",
                UnusedInputWarning,
            )

    def _render_prompt(self) -> str:
        """
        Renders the Jinja template with the provided inputs. Returns
        the rendered prompt string.
        """
        if self._rendered_prompt is None:
            try:
                template = self._jinja_env.from_string(self.template)
                self._rendered_prompt = template.render(self.inputs)

            except jinja2.UndefinedError as e:
                # This might be redundant due to _validate_inputs, but catches render-time issues
                raise ValidationError(f"Error rendering template: {e}") from e

            except Exception as e:
                raise TemplateError(f"Failed to render template: {e}") from e

        return self._rendered_prompt

    def check(self) -> Tuple[Optional[List[str]], Optional[float]]:
        """
        Performs validation checks and estimates cost.

        Returns
        -------
        tuple
            A tuple containing:
                - list or None
                    List of issue strings if any problems found, otherwise None
                - float or None
                    Estimated cost if checks pass, otherwise None

        Notes
        -----
        Performs the following checks:
        1. Model existence and basic info retrieval
        2. Renders the prompt
        3. Calculates token count
        4. Checks context length against model limits
        5. Estimates cost
        """
        issues = []
        cost = None

        # 1. Check model info
        self._model_info = get_model_info(self.model)

    async def send(self, **kwargs) -> Any:
        """Sends the prompt to the specified LLM and returns the response.

        Performs checks automatically before sending. If checks fail,
        it raises the relevant exception (e.g., ContextLengthError).

        Args:
            **kwargs: Additional arguments to pass to litellm.completion
                      (e.g., temperature, max_tokens for output).

        Returns:
            The response from the LLM (typically a litellm ModelResponse object).

        Raises:
            EzpromptError: If checks fail before sending.
            ModelError: If there's an API error during the call to the LLM.
        """
        issues, _ = await self.check()
        if issues:
            # If context length error is the primary issue, raise it specifically
            context_error_msg = next(
                (issue for issue in issues if "Context Length Error" in issue),
                None,
            )
            if (
                context_error_msg
                and self._input_tokens
                and self._model_info
                and self._model_info.get("max_input_tokens")
            ):
                # Re-fetch suggested models if needed, or reuse from check if stored
                suggested = await find_models_for_prompt_length(
                    self._input_tokens, self.model
                )
                raise ContextLengthError(
                    context_error_msg,
                    prompt_length=self._input_tokens,
                    max_length=self._model_info["max_input_tokens"],
                    suggested_models=suggested,
                )
            # Otherwise, raise a general validation error summarizing issues
            raise ValidationError("Prompt checks failed:\n" + "\n".join(issues))

        if not self._rendered_prompt:
            # Should have been rendered by check(), but as a fallback:
            await self._render_prompt()

        messages = [{"role": "user", "content": self._rendered_prompt}]

        try:
            # Ensure litellm is called asynchronously if possible
            # litellm.completion itself is blocking, but acompletion is async
            response = await litellm.acompletion(
                model=self.model, messages=messages, **kwargs
            )
            return response
        except Exception as e:
            # Catch potential API errors, connection issues, etc. from litellm
            raise ModelError(
                f"Error sending prompt to model '{self.model}': {e}"
            ) from e

    # Potentially add a method to get the rendered prompt without sending
    async def get_rendered_prompt(self) -> str:
        """Returns the rendered prompt string."""
        return await self._render_prompt()
