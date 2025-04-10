# ezprompt/prompt.py

"""The main Prompt class for ezprompt."""

import jinja2
import litellm
import asyncio
from typing import Dict, Any, Optional, Tuple, List

from .exceptions import (
    TemplateError,
    ValidationError,
    ModelError,
    ContextLengthError,
)
from .models import (
    get_model_info,
    get_token_count,
    estimate_cost,
    find_models_for_prompt_length,
)

class Prompt:
    """Represents a prompt to be sent to an LLM."""

    def __init__(self, template: str, inputs: Dict[str, Any], model: str):
        """Initializes the Prompt object.

        Args:
            template: The prompt template string (can use Jinja2 syntax).
            inputs: A dictionary of input variables for the template.
            model: The name of the target LLM (e.g., 'gpt-3.5-turbo').
        """
        self.template = template
        self.inputs = inputs
        self.model = model
        self._rendered_prompt: Optional[str] = None
        self._input_tokens: Optional[int] = None
        self._model_info: Optional[Dict[str, Any]] = None

        # Initialize Jinja environment
        self._jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        try:
            self._parsed_template = self._jinja_env.parse(template)
        except jinja2.TemplateSyntaxError as e:
            raise TemplateError(f"Syntax error in template: {e}") from e
        
        self._validate_inputs()

    def _validate_inputs(self):
        """Checks if all required template variables are present in inputs."""
        required_vars = jinja2.meta.find_undeclared_variables(self._parsed_template)
        missing_vars = required_vars - set(self.inputs.keys())
        if missing_vars:
            raise ValidationError(
                f"Missing required inputs for template: {', '.join(missing_vars)}"
            )
        # We could add checks for unused inputs as well, if desired.

    async def _render_prompt(self) -> str:
        """Renders the Jinja template with the provided inputs."""
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

    async def check(self) -> Tuple[Optional[List[str]], Optional[float]]:
        """Performs validation checks and estimates cost.

        Checks:
        1. Model existence and basic info retrieval.
        2. Renders the prompt.
        3. Calculates token count.
        4. Checks context length against model limits.
        5. Estimates cost.

        Returns:
            A tuple containing:
            - A list of issue strings if any problems are found, otherwise None.
            - The estimated cost (float) if checks pass, otherwise None.
        """
        issues = []
        cost = None
        
        # 1. Check model info
        self._model_info = await get_model_info(self.model)
        if not self._model_info or self._model_info.get("error"):
            err_msg = self._model_info.get("error", "Unknown error") if self._model_info else "Model not found or info unavailable"
            issues.append(f"Model Error: Could not retrieve info for model '{self.model}'. {err_msg}")
            return issues, cost # Cannot proceed without model info
        if not self._model_info.get("litellm_info_available"):
             issues.append(f"Warning: Limited info available for model '{self.model}'. Context length and cost checks may be inaccurate.")

        # 2. Render prompt
        try:
            rendered = await self._render_prompt()
        except (ValidationError, TemplateError) as e:
            issues.append(f"Template Error: {e}")
            return issues, cost

        # 3. Calculate token count
        try:
            self._input_tokens = await get_token_count(self.model, rendered)
        except Exception as e:
             issues.append(f"Tokenization Error: Could not count tokens for model '{self.model}'. {e}")
             # Attempt to continue with checks if possible, but context length check will likely fail
             self._input_tokens = None # Ensure it's None if counting failed

        # 4. Check context length
        if self._input_tokens is not None and self._model_info.get("max_input_tokens") is not None:
            max_len = self._model_info["max_input_tokens"]
            if self._input_tokens > max_len:
                msg = (
                    f"Context Length Error: Prompt length ({self._input_tokens} tokens) "
                    f"exceeds model '{self.model}' maximum ({max_len} tokens)."
                )
                issues.append(msg)
                # Find alternative models
                suggested_models_info = await find_models_for_prompt_length(self._input_tokens, current_model=self.model)
                if suggested_models_info:
                    suggestions = []
                    for info in suggested_models_info[:5]: # Limit suggestions
                         cost_info = "(Cost N/A)" 
                         if info.get("input_cost_per_token") is not None:
                             # Rough cost estimate for the current prompt with the suggested model
                             est_cost = await estimate_cost(info["model_name"], self._input_tokens)
                             cost_info = f"(Est. Cost: ${est_cost:.6f})" if est_cost is not None else "(Cost N/A)"
                         suggestions.append(f"  - {info['model_name']} (Max Tokens: {info.get('max_input_tokens', 'N/A')}) {cost_info}")
                    issues.append("Suggested models that might fit this prompt:\n" + "\n".join(suggestions))
                else:
                    issues.append("No alternative models found in litellm database that fit this prompt length.")
                # Raise the specific exception after gathering info
                # raise ContextLengthError(msg, self._input_tokens, max_len, suggested_models_info) # Decided to return issues instead of raising here
        elif self._input_tokens is None:
            issues.append(f"Warning: Could not check context length due to tokenization error.")
        elif self._model_info.get("max_input_tokens") is None:
             issues.append(f"Warning: Cannot check context length for model '{self.model}' as max_input_tokens is unknown.")

        # 5. Estimate cost (only if no major issues found, especially context length)
        if self._input_tokens is not None and not any("Context Length Error" in issue for issue in issues):
            # Note: Cost estimation requires assumptions about output tokens.
            # We'll estimate based on input only for now.
            cost = await estimate_cost(self.model, self._input_tokens)
            if cost is None:
                issues.append(f"Warning: Could not estimate cost for model '{self.model}'. Pricing info might be unavailable.")

        return issues if issues else None, cost

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
            context_error_msg = next((issue for issue in issues if "Context Length Error" in issue), None)
            if context_error_msg and self._input_tokens and self._model_info and self._model_info.get("max_input_tokens"):
                 # Re-fetch suggested models if needed, or reuse from check if stored
                 suggested = await find_models_for_prompt_length(self._input_tokens, self.model)
                 raise ContextLengthError(
                     context_error_msg,
                     prompt_length=self._input_tokens,
                     max_length=self._model_info["max_input_tokens"],
                     suggested_models=suggested
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
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            # Catch potential API errors, connection issues, etc. from litellm
            raise ModelError(f"Error sending prompt to model '{self.model}': {e}") from e

    # Potentially add a method to get the rendered prompt without sending
    async def get_rendered_prompt(self) -> str:
        """Returns the rendered prompt string."""
        return await self._render_prompt() 