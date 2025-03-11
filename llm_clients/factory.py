"""
Factory methods for creating LLM clients.
"""
import logging
from typing import Dict, Optional, Type

from .interfaces import ProviderType, ModelInfo
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Registry of client classes keyed by provider type
CLIENT_REGISTRY: Dict[ProviderType, Type[LLMClient]] = {}


# Default model IDs for each provider
DEFAULT_MODELS = {
    ProviderType.ANTHROPIC: "claude-3-opus-20240229",
    ProviderType.OPENAI: "gpt-4-turbo",
    ProviderType.WATSONX: "ibm/granite-3-2-8b-instruct",
    ProviderType.OLLAMA: "llama3"
}


def register_client(client_class: Type[LLMClient]) -> Type[LLMClient]:
    """
    Register a client class with the factory.
    
    Args:
        client_class: The client class to register
        
    Returns:
        The client class (for decorator usage)
    """
    CLIENT_REGISTRY[client_class.provider_type] = client_class
    return client_class


def get_client(provider: str, model_id: Optional[str] = None) -> LLMClient:
    """
    Get a client for the specified provider and model.
    
    Args:
        provider: The provider name or ProviderType
        model_id: The model ID (uses default if None)
        
    Returns:
        An LLM client instance
    """
    # Convert string to ProviderType if needed
    if isinstance(provider, str):
        try:
            provider = ProviderType(provider.lower())
        except ValueError:
            raise ValueError(f"Unknown provider: {provider}")
    
    # Check if provider is supported
    if provider not in CLIENT_REGISTRY:
        raise ValueError(f"Provider {provider} is not registered")
    
    # Get client class
    client_class = CLIENT_REGISTRY[provider]
    
    # Use default model if not specified
    if model_id is None:
        model_id = DEFAULT_MODELS.get(provider)
    
    # Create and return client
    return client_class(model_id)


def list_available_models() -> str:
    """
    Get a formatted string listing all available models from all providers.
    
    Returns:
        str: Formatted list of all available models
    """
    result = "Available Models:\n"
    
    for provider_type in ProviderType:
        # Skip if provider not registered
        if provider_type not in CLIENT_REGISTRY:
            continue
            
        # Get client class
        client_class = CLIENT_REGISTRY[provider_type]
        
        # Format provider header
        provider_name = provider_type.name
        result += f"\n{provider_name}\n"
        result += "-" * len(provider_name) + "\n"
        
        try:
            # Get available models
            models = client_class.get_available_models()
            
            # List models
            default_model = DEFAULT_MODELS.get(provider_type)
            for key, model in models.items():
                default_marker = " (default)" if key == default_model or model.model_id == default_model else ""
                context_info = f", {model.context_length} tokens" if model.context_length else ""
                result += f"  - {key}{default_marker}: {model.name} ({model.description}{context_info})\n"
                
        except Exception as e:
            result += f"  Error retrieving models: {str(e)}\n"
    
    return result


def get_available_models(provider: str) -> Dict[str, ModelInfo]:
    """
    Get available models for a specific provider.
    
    Args:
        provider: The provider name or ProviderType
        
    Returns:
        Dict[str, ModelInfo]: Dictionary of model_key -> model_info
    """
    # Convert string to ProviderType if needed
    if isinstance(provider, str):
        try:
            provider = ProviderType(provider.lower())
        except ValueError:
            raise ValueError(f"Unknown provider: {provider}")
    
    # Check if provider is supported
    if provider not in CLIENT_REGISTRY:
        raise ValueError(f"Provider {provider} is not registered")
    
    # Get client class
    client_class = CLIENT_REGISTRY[provider]
    
    # Get available models
    return client_class.get_available_models()