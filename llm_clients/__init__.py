"""
LLM Clients Library - A unified interface for multiple LLM providers.
"""
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import key interfaces
from .interfaces import ProviderType, ModelInfo, GenerationParams, Message
from .llm_client import LLMClient

# Import factory functions
from .factory import get_client, list_available_models, get_available_models

# Import specific client implementations
# (these are registered with the factory via decorators)
from .anthropic import AnthropicClient
from .openai import OpenAIClient
from .watsonx import WatsonxClient
from .ollama import OllamaClient

# Add an alias for get_client to match the expected get_llm_client name
get_llm_client = get_client

__all__ = [
    # Interfaces and types
    'LLMClient',
    'ProviderType',
    'ModelInfo',
    'GenerationParams',
    'Message',
    
    # Factory functions
    'get_client',
    'get_llm_client',  # Add the alias
    'list_available_models',
    'get_available_models',
    
    # Client implementations
    'AnthropicClient',
    'OpenAIClient',
    'WatsonxClient',
    'OllamaClient'
]