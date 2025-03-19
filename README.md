# LLM Clients

A standardized interface for interacting with various Large Language Model providers.

## Features

- Unified API across multiple LLM providers
- Type hints throughout with Pydantic validation
- Automatic environment detection for credentials
- Streaming support for all providers
- Chat and completion model interfaces
- Context manager support for connection management
- Customizable prompt formatting for different model families
- Embeddings support where available

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
from llm_clients import get_client, ProviderType, GenerationParams

# Load environment variables from .env file
# (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

# Get a client for a specific provider
openai_client = get_client("openai", model_id="gpt-4-turbo")
claude_client = get_client("anthropic", model_id="claude-3-opus-20240229")

# Generate text with default settings
response = openai_client.generate("Explain quantum computing in simple terms.")
print(response)

# Generate with custom parameters
params = GenerationParams(
    temperature=0.7,
    max_tokens=300,
    top_p=0.95
)
response = claude_client.generate(
    "Write a short poem about technology.",
    params
)
print(response)

# Stream responses
for chunk in claude_client.generate_stream("Explain how neural networks work."):
    print(chunk, end="")

# Use with a context manager
with get_client(ProviderType.OLLAMA, "llama3") as client:
    response = client.generate("What are the main principles of good code design?")
    print(response)
```

## Chat Interfaces
All clients support a standard chat interface via the generate_with_messages method:

```
from llm_clients import get_client, Message, MessageRole

client = get_client("openai", "gpt-4")

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant specializing in Python."),
    Message(role=MessageRole.USER, content="What's the difference between a list and a tuple?")
]

response = client.generate_with_messages(messages)
print(response)
```

## Provider-Specific Configuration

Each provider supports custom configuration:

```
from llm_clients import get_client, OpenAIConfig, AnthropicConfig, OllamaConfig

# Custom OpenAI configuration
openai_config = OpenAIConfig(
    api_key="your-api-key",  # Override environment variable
    timeout=60.0,  # Longer timeout
    max_retries=5  # More retries
)
openai_client = get_client("openai", "gpt-4", config=openai_config)

# Custom Ollama configuration
ollama_config = OllamaConfig(
    base_url="http://192.168.1.100:11434",  # Custom Ollama server
    num_gpu=2,  # Use 2 GPUs
    num_thread=8  # Use 8 threads
)
ollama_client = get_client("ollama", "llama3", config=ollama_config)
```

## Embeddings Support

Generate embeddings from supported providers:
```
from llm_clients import get_embeddings

# Simple interface using factory function
texts = ["Hello world", "How are you today?"]
embeddings = get_embeddings(texts, provider="openai")

# Or use a client directly
from llm_clients import get_client

client = get_client("openai")
embeddings = client.get_embeddings(texts)
```

## Configuration
Set the following environment variables based on the providers you want to use:

```
OpenAI: OPENAI_API_KEY
Anthropic: ANTHROPIC_API_KEY
IBM WatsonX: WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID
Ollama: OLLAMA_BASE_URL (default: "http://localhost:11434")
```