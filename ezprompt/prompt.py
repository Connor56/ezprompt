# ezprompt/prompt.py

"""The main Prompt class for ezprompt."""

import jinja2
from jinja2 import meta
import openai
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

    def __init__(
        self,
        template: str,
        inputs: Dict[str, Any],
        model: str,
        api_key: str,
        log: bool = False,
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
        api_key : str
            The API key for the LLM provider.
        log : bool
            Whether to log output information for debugging.
        """
        self.template = template
        self.inputs = inputs
        self.model = model
        self.api_key = api_key
        self.log = log

        self._rendered_prompt: Optional[str] = None
        self._input_tokens: Optional[int] = None

        # Retrieve model info immediately to fail fast if model invalid
        # And to have info available for checks/rendering logic
        try:
            self._model_info = get_model_info(self.model)
        except ModelError as e:
            # Re-raise ModelError if get_model_info specifically raises it
            raise e
        except Exception as e:
            # Wrap other potential errors during model info retrieval
            raise ModelError(
                f"Failed to retrieve info for model '{self.model}': {e}"
            ) from e

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
                # This should ideally be caught by _validate_inputs, but acts as a safeguard
                raise ValidationError(
                    f"Error rendering template: Missing input {e}"
                ) from e

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
                    Estimated input cost if checks pass, otherwise None

        Notes
        -----
        Performs the following checks:
        1. Renders the prompt (catches rendering errors).
        2. Calculates token count using the specific model's tokenizer.
        3. Checks context length against model limits.
        4. Estimates cost based on input tokens (output tokens unknown).
        """
        issues = []
        cost = None
        self._input_tokens = None  # Ensure it's None initially

        # 1. Render prompt (implicitly checks template syntax and inputs)
        try:
            rendered_prompt = self._render_prompt()

        except (ValidationError, TemplateError) as e:
            issues.append(f"Template Error: {e}")

            return issues, cost

        # 2. Calculate token count using the model's tokenizer
        try:
            self._input_tokens = self._model_info.count_tokens(rendered_prompt)

        except ValidationError as e:
            issues.append(f"Tokenization Error: {e}")

            return issues, cost

        except Exception as e:
            issues.append(f"Tokenization Error: Failed to count tokens - {e}")

            return issues, cost

        # 3. Check context length
        if self._input_tokens > self._model_info.context_length:
            msg = (
                f"Context Length Error: Calculated prompt length ({self._input_tokens} tokens) "
                f"exceeds model '{self.model}' maximum ({self._model_info.context_length} tokens)."
            )
            issues.append(msg)

        # 4. Estimate cost (Input only)
        if self._model_info.pricing_in is not None:
            # Standard cost per 1M tokens
            cost = (
                self._input_tokens / 1_000_000
            ) * self._model_info.pricing_in

            # Add per-call cost if applicable
            if self._model_info.call_cost is not None:
                cost += self._model_info.call_cost
        else:
            # Only add warning if no more serious errors occurred
            issues.append(
                f"Cost Estimation Skipped: Pricing info unavailable for model '{self.model}'."
            )

        return issues if issues else None, cost

    async def send(self, **kwargs) -> Any:
        """
        Sends the prompt to the specified LLM and returns the response.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to pass to the LLM API call
            (e.g., temperature, max_tokens for output).
            Specific arguments depend on the underlying API (e.g., OpenAI).

        Returns
        -------
        response : Any
            The response from the LLM (actual format depends on the provider/API).

        Raises
        ------
        ValidationError
            If prompt rendering or input validation fails during checks.
        ContextLengthError
            If the prompt exceeds the model's context limit.
        ModelError
            If there's an API error during the call to the LLM or model info issues.

        Notes
        -----
        Performs checks automatically before sending. If checks fail,
        it raises the relevant exception.
        """
        issues, input_cost = self.check()

        if issues:
            raise ValidationError("Prompt checks failed:\n" + "\n".join(issues))

        rendered_prompt = self._render_prompt()  # Ensure it's rendered

        client = openai.AsyncOpenAI(api_key=self.api_key)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": rendered_prompt}],
            **kwargs,
        )

        model_info = get_model_info(self.model)

        token_cost = (
            response.usage.prompt_tokens * model_info.pricing_in
            + response.usage.completion_tokens * model_info.pricing_out
        ) / 1_000_000

        if self.log:
            print(f"Prompt Cost: {token_cost}")

        return response.choices[0].message.content
