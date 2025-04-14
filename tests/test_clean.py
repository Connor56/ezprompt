from ezprompt.clean import extract_json


def test_extract_json():
    text = """
    Thank you for your question, here's the answer:
    
    ```json
    {
        "name": "John",
        "age": 30
    }
    ```
    """

    assert extract_json(text) == {"name": "John", "age": 30}
