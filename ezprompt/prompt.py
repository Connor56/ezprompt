# ezprompt/prompt.py

"""The main Prompt class for ezprompt."""

import jinja2
from jinja2 import meta
import openai
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from .models import get_model_info
from .data import save_outcome, process_response, PromptOutcome
    save_outcome,
    PromptOutcome,
import hashlib
import tiktoken

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
    def _render(self) -> str:
        """Render the prompt template with inputs."""
        pass

    @abstractmethod
    def _check(self) -> Tuple[Optional[List[str]], Optional[float]]:
        """Perform validation checks and estimate cost."""
        pass

    @abstractmethod
    async def send(self, **kwargs) -> Any:
        """Send the prompt to the LLM."""
        pass


# Make Prompt implement the BasePrompt interface
class Prompt(BasePrompt):
    """Represents a prompt using Jinja2 templating to be sent to an LLM."""

    def __init__(
        self,
        template: str,
        model: str,
        api_key: str,
        log: bool = False,
    ):
        """
        Initialize the Prompt instance with a template, model, and
        an api key for the LLM provider.

        Parameters
        ----------
        template : str
            The prompt template string (can use Jinja2 syntax).
        model : str
            The name of the target LLM (e.g., 'gpt-3.5-turbo').
        api_key : str
            The API key for the LLM provider.
        log : bool
            Whether to log output information for debugging.
        """
        self.template = template
        self.model = model
        self.api_key = api_key
        self.log = log

        # Hash the template for caching
        self._hash = hashlib.sha256(self.template.encode()).hexdigest()

        self._rendered_prompt: Optional[str] = None
        self._input_tokens: Optional[int] = None

        try:
            self._model_info = get_model_info(self.model)
        except ModelError as e:
            raise e
        except Exception as e:
            # Wrap other potential errors during model info retrieval
            raise ModelError(
                f"Failed to retrieve info for model '{self.model}': {e}"
            ) from e

        # StrictUndefined: Enforces perfect adherence to the template
        self._jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined)

        # Check template can be parsed
        try:
            self._parsed_template = self._jinja_env.parse(template)

        except jinja2.TemplateSyntaxError as e:
            raise TemplateError(f"Syntax error in template: {e}") from e

    def _validate_inputs(self):
        """
        Checks if all the prompt template variables are present in
        the provided inputs.
        """
        if self.inputs is None:
            raise ValidationError("Inputs must be provided before validation")

        required_vars = meta.find_undeclared_variables(self._parsed_template)

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

    def _render(self) -> str:
        """
        Renders the Jinja template with the provided inputs. Returns
        the rendered prompt string.
        """
        try:
            template = self._jinja_env.from_string(self.template)
            self._rendered_prompt = template.render(self.inputs)

            # Count the tokens
            encoding = tiktoken.encoding_for_model(self._model_info.id)
            self._input_tokens = len(encoding.encode(self._rendered_prompt))

        except jinja2.UndefinedError as e:
            # This should ideally be caught by _validate_inputs, but acts as a safeguard
            raise ValidationError(
                f"Error rendering template: Missing input {e}"
            ) from e

        except Exception as e:
            raise TemplateError(f"Failed to render template: {e}") from e

        return self._rendered_prompt

    def _check(self) -> float | None:
        """
        Performs validation checks and estimates cost.
        """
        if self._input_tokens > self._model_info.context_length:
            raise ContextLengthError(
                f"Input tokens ({self._input_tokens}) exceed the model's context length ({self._model_info.context_length})"
            )

        # TODO: add the ability to load cached outcomes and predict cost based on that

        min_cost = (
            self._model_info.call_cost  # Base cost
            + self._input_tokens
            * self._model_info.pricing_in
            / 1_000_000  # Input cost
        )

        return min_cost

    def format(self, inputs: Dict[str, Any]):
        """
        Format the prompt with the provided inputs.
        """
        self.inputs = inputs
        self._validate_inputs()
        self._render()
        self._check()

    async def send(self, **kwargs) -> PromptOutcome:
        """
        Sends the prompt to the specified LLM and returns the response.
        """
        cost = self._check()

        if cost is not None and self.log:
            print(f"Minimum prompt cost is: {cost}")

        rendered_prompt = self._render()  # Ensure it's rendered

        client = openai.AsyncOpenAI(api_key=self.api_key)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": rendered_prompt}],
            **kwargs,
        )

        outcome = process_response(response, self._model_info)

        # Save the outcome to the cache
        save_outcome(outcome, self.__class__.__name__, self._hash)

        # Add the text of the prompt and response to the outcome
        outcome.prompt = rendered_prompt
        outcome.response = response.choices[0].message.content

        return outcome

    async def format_and_send(self, inputs: Dict[str, Any], **kwargs):
        """
        Format the prompt with the provided inputs and send it to the LLM.
        """
        self.format(inputs)
        return await self.send(**kwargs)
