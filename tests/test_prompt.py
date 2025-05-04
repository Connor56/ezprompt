import pytest
from unittest.mock import patch, AsyncMock
import os
from dotenv import load_dotenv
from ez_prompt.prompt import StatPrompt
from ez_prompt.exceptions import ValidationError, TemplateError

# Load the .env file
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(f"{DIR_PATH}/../.env"))


@pytest.fixture
def fake_prompt():
    template = "Hello, {{ name }}! How are you doing today?"
    inputs = {"name": "World"}
    model = "gpt-3.5-turbo"
    api_key = "fake-api-key"

    prompt = StatPrompt(template, model, api_key)

    # Format with some test inputs
    prompt.format(inputs)

    return prompt


def test_prompt_initialization():
    # Test successful initialization
    prompt = StatPrompt(
        template="Hello, {{ name }}!",
        model="gpt-3.5-turbo",
        api_key="fake-api-key",
    )

    prompt.format({"name": "World"})

    assert prompt.template == "Hello, {{ name }}!"
    assert prompt.inputs == {"name": "World"}
    assert prompt.model == "gpt-3.5-turbo"
    assert prompt.api_key == "fake-api-key"


def test_prompt_missing_inputs():
    # Test initialization with missing inputs
    with pytest.raises(ValidationError):
        prompt = StatPrompt(
            template="Hello, {{ name }}!",
            model="gpt-3.5-turbo",
            api_key="fake-api-key",
        )

        prompt.format({})


def test_prompt_template_syntax_error():
    # Test initialization with template syntax error
    with pytest.raises(TemplateError):
        StatPrompt(
            template="Hello, {{ name!",  # Missing closing bracket
            model="gpt-3.5-turbo",
            api_key="fake-api-key",
        )


def test__render(fake_prompt):
    # Test prompt rendering
    rendered = fake_prompt._render()
    assert rendered == "Hello, World! How are you doing today?"


def test__check_method(fake_prompt):
    # Test the check method returns no issues for valid prompt
    cost = fake_prompt._check()
    assert isinstance(cost, float)


@pytest.mark.asyncio
async def test_send_method():
    # Mock the OpenAI client response
    mock_response = AsyncMock()
    mock_response.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }

    with patch("openai.AsyncOpenAI") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.chat.completions.create = mock_response

        prompt = StatPrompt(
            template="Hello, {{ name }}!",
            model="gpt-3.5-turbo",
            api_key="fake-api-key",
        )

        prompt.format({"name": "World"})

        response = await prompt.send(temperature=0.7)

        # Verify the client was called with correct parameters
        mock_client.assert_called_once_with(api_key="fake-api-key")
        mock_instance.chat.completions.create.assert_called_once()

        # Check that we got a response
        assert response is not None


@pytest.mark.asyncio
async def test_send_method():

    api_key = os.getenv("OPENAI_API_KEY")

    prompt = StatPrompt(
        template="Hello, {{ name }}!",
        model="gpt-3.5-turbo",
        api_key=api_key,
        log=True,
    )

    prompt.format(
        {"name": "You big fat ugly sausage, what do you think of that eh?"},
    )

    response = await prompt.send(temperature=0.3)

    # Check that we got a response
    assert response is not None
    assert response.tool_cost == 0
    assert response.input_tokens == 23
