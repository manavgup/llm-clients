# LLM Clients

A standardized interface for interacting with various Large Language Model providers.

## Supported Providers

- **OpenAI** - GPT-3.5, GPT-4 models
- **Anthropic** - Claude models
- **IBM WatsonX** - Granite, Flan, Llama, and Mistral models via IBM
- **Ollama** - Open source models (Llama, Mistral, etc.) via local Ollama server

## Installation

```bash
# From PyPI (coming soon)
pip install llm-clients

# From GitHub
pip install git+https://github.com/manavgup/llm-clients.git
```

## Quickstart
```
from llm_clients import get_client

# Load environment variables from .env file
# (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

# Get a client for a specific provider
openai_client = get_client("openai", model_id="gpt-4-turbo")
claude_client = get_client("anthropic", model_id="claude-3-opus-20240229")

# Generate text
response = openai_client.generate("Explain quantum computing in simple terms.")
print(response)

# Stream responses
for chunk in claude_client.generate_stream("Write a short poem about technology."):
    print(chunk, end="")
```

## Configuration
Set the following environment variables based on the providers you want to use:

```
OpenAI: OPENAI_API_KEY
Anthropic: ANTHROPIC_API_KEY
IBM WatsonX: WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID
Ollama: OLLAMA_BASE_URL (default: "http://localhost:11434")
```