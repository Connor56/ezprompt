import json


def extract_json(text: str) -> dict:
    """
    Extracts a JSON object from a string.
    """
    # First try loading it from the text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Then try loading it from the text between ```json and ```
    try:
        return json.loads(text.split("```json")[1].split("```")[0])
    except json.JSONDecodeError:
        pass

    # If all else fails, raise an error
    raise ValueError("No JSON object found in the text")
