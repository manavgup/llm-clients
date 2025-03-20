from llm_clients import get_client
from pydantic import BaseModel

class Recipe(BaseModel):
    ingredients: list[str]
    steps: list[str]
    cook_time: int

# With any client
ollama = get_client("ollama", "llama3.1")
openai = get_client("openai")
anthropic = get_client("anthropic", "claude-3-5-sonnet-20240620")
watsonx = get_client("watsonx", "ibm/granite-3-2-8b-instruct")
for client in [ollama, watsonx]:
    response = client.generate_structured(
        "Give me a cookie recipe",
        Recipe
    )
    print(f"response from {client}: ",response.ingredients)
    print(f"Successfully completed for {client}")