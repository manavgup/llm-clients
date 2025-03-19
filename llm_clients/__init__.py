"""
LLM Clients Library - A unified interface for multiple LLM providers.

This library provides a standardized way to interact with various Large Language Model
providers including OpenAI, Anthropic Claude, IBM WatsonX, and Ollama.
"""
import logging
from importlib.metadata import version, PackageNotFoundError
from typing import Dict, List, Optional, Union

# Set up logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Try to get version
try:
    __version__ = version("llm_clients")
except PackageNotFoundError:
    __version__ = "0.1.0.dev"

# Import environment variables loader and apply it
from dotenv import load_dotenv
load_dotenv()

# Import key interfaces
from .interfaces import (
    ProviderType, 
    ModelInfo, 
    GenerationParams, 
    Message, 
    MessageRole,
    ClientConfig
)

# Import the base LLM client
from .llm_client import LLMClient

# Import factory functions
from .factory import (
    get_client, 
    list_available_models, 
    get_available_models,
    update_default_model
)

# Import client configs
from .anthropic_client import AnthropicConfig
from .openai_client import OpenAIConfig
from .ollama_client import OllamaConfig
from .watsonx_client import WatsonxConfig

# Add an alias for get_client to match the expected get_llm_client name
get_llm_client = get_client


# Import specific client implementations lazily
def _get_anthropic_client():
    from .anthropic_client import AnthropicClient
    return AnthropicClient

def _get_openai_client():
    from .openai_client import OpenAIClient
    return OpenAIClient

def _get_ollama_client():
    from .ollama_client import OllamaClient
    return OllamaClient

def _get_watsonx_client():
    from .watsonx_client import WatsonxClient
    return WatsonxClient


# Lazily load client classes when accessed
class LazyClientLoader:
    @property
    def AnthropicClient(self):
        return _get_anthropic_client()
    
    @property
    def OpenAIClient(self):
        return _get_openai_client()
    
    @property
    def OllamaClient(self):
        return _get_ollama_client()
    
    @property
    def WatsonxClient(self):
        return _get_watsonx_client()


# Create the lazy loader and export its properties
_lazy_loader = LazyClientLoader()
AnthropicClient = _lazy_loader.AnthropicClient
OpenAIClient = _lazy_loader.OpenAIClient
OllamaClient = _lazy_loader.OllamaClient
WatsonxClient = _lazy_loader.WatsonxClient


# Create a convenience function for embedding
def get_embeddings(
    texts: List[str], 
    provider: Union[str, ProviderType] = "openai",
    model_id: Optional[str] = None,
    config: Optional[ClientConfig] = None
) -> List[List[float]]:
    """
    Get embeddings for a list of texts from a specific provider.
    
    Args:
        texts: List of texts to get embeddings for
        provider: The LLM provider to use (default: "openai")
        model_id: Optional specific model ID to use
        config: Optional provider configuration
        
    Returns:
        List[List[float]]: List of embedding vectors
        
    Raises:
        NotImplementedError: If the specified provider doesn't support embeddings
        ValueError: If the provider is invalid or configuration is incorrect
    """
    client = get_client(provider, model_id, config)
    return client.get_embeddings(texts)


__all__ = [
    # Version
    "__version__",
    
    # Interfaces and types
    "LLMClient",
    "ProviderType",
    "ModelInfo",
    "GenerationParams",
    "Message",
    "MessageRole",
    "ClientConfig",
    
    # Configuration classes
    "AnthropicConfig",
    "OpenAIConfig",
    "OllamaConfig",
    "WatsonxConfig",
    
    # Factory functions
    "get_client",
    "get_llm_client",  # Alias
    "list_available_models",
    "get_available_models",
    "update_default_model",
    "get_embeddings",
    
    # Client implementations
    "AnthropicClient",
    "OpenAIClient",
    "WatsonxClient",
    "OllamaClient"
]