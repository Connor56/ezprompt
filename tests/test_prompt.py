import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import os
from dotenv import load_dotenv
from ezprompt.prompt import Prompt
from ezprompt.exceptions import ValidationError, TemplateError

# Load the .env file
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(f"{DIR_PATH}/../.env"))


@pytest.fixture
def fake_prompt():
    template = "Hello, {{ name }}! How are you doing today?"
    inputs = {"name": "World"}
    model = "gpt-3.5-turbo"
    api_key = "fake-api-key"
    return Prompt(template, inputs, model, api_key)


def test_prompt_initialization():
    # Test successful initialization
    prompt = Prompt(
        template="Hello, {{ name }}!",
        inputs={"name": "World"},
        model="gpt-3.5-turbo",
        api_key="fake-api-key",
    )
    assert prompt.template == "Hello, {{ name }}!"
    assert prompt.inputs == {"name": "World"}
    assert prompt.model == "gpt-3.5-turbo"
    assert prompt.api_key == "fake-api-key"


def test_prompt_missing_inputs():
    # Test initialization with missing inputs
    with pytest.raises(ValidationError):
        Prompt(
            template="Hello, {{ name }}!",
            inputs={},  # Missing 'name'
            model="gpt-3.5-turbo",
            api_key="fake-api-key",
        )


def test_prompt_template_syntax_error():
    # Test initialization with template syntax error
    with pytest.raises(TemplateError):
        Prompt(
            template="Hello, {{ name!",  # Missing closing bracket
            inputs={"name": "World"},
            model="gpt-3.5-turbo",
            api_key="fake-api-key",
        )


def test_render_prompt(fake_prompt):
    # Test prompt rendering
    rendered = fake_prompt._render_prompt()
    assert rendered == "Hello, World! How are you doing today?"


def test_check_method(fake_prompt):
    # Test the check method returns no issues for valid prompt
    issues, cost = fake_prompt.check()
    assert issues is None
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

        prompt = Prompt(
            template="Hello, {{ name }}!",
            inputs={"name": "World"},
            model="gpt-3.5-turbo",
            api_key="fake-api-key",
        )

        response = await prompt.send(temperature=0.7)

        # Verify the client was called with correct parameters
        mock_client.assert_called_once_with(api_key="fake-api-key")
        mock_instance.chat.completions.create.assert_called_once()

        # Check that we got a response
        assert response is not None


@pytest.mark.asyncio
async def test_send_method():

    api_key = os.getenv("OPENAI_API_KEY")

    prompt = Prompt(
        template="Hello, {{ name }}!",
        inputs={
            "name": "You big fat ugly sausage, what do you think of that eh?"
        },
        model="gpt-3.5-turbo",
        api_key=api_key,
        log=True,
    )

    response = await prompt.send(temperature=0.3)

    assert False

    # Check that we got a response
    assert response is not None
