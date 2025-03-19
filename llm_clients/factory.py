"""
Factory methods for creating LLM clients with improved type hints and error handling.
"""
import logging
import importlib
from typing import Dict, Optional, Type, TypeVar, Union, List, Any, cast
from functools import lru_cache

from .interfaces import ProviderType, ModelInfo, ClientConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Generic type variable for client configuration
T = TypeVar('T', bound=ClientConfig)

# Registry of client classes keyed by provider type
CLIENT_REGISTRY: Dict[ProviderType, Type[LLMClient]] = {}

# Default model IDs for each provider
DEFAULT_MODELS = {
    ProviderType.ANTHROPIC: "claude-3-5-sonnet-20240307",
    ProviderType.OPENAI: "gpt-4-turbo",
    ProviderType.WATSONX: "ibm/granite-13b-instruct-v2",
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
    if not hasattr(client_class, 'provider_type'):
        raise ValueError(f"Client class {client_class.__name__} does not have a provider_type attribute")
        
    CLIENT_REGISTRY[client_class.provider_type] = client_class
    return client_class


def get_client(
    provider: Union[str, ProviderType], 
    model_id: Optional[str] = None,
    config: Optional[ClientConfig] = None
) -> LLMClient:
    """
    Get a client for the specified provider and model.
    
    Args:
        provider: The provider name or ProviderType
        model_id: The model ID (uses default if None)
        config: Optional configuration for the client
        
    Returns:
        An LLM client instance
    """
    # Convert string to ProviderType if needed
    provider_type = _resolve_provider_type(provider)
    
    # Check if provider is supported and loaded
    client_class = _get_client_class(provider_type)
    
    # Use default model if not specified
    if model_id is None:
        model_id = DEFAULT_MODELS.get(provider_type)
        logger.info(f"Using default model {model_id} for provider {provider_type.name}")
    
    # Create and return client
    try:
        if config:
            return client_class(model_id, config)
        else:
            return client_class(model_id)
    except Exception as e:
        logger.error(f"Failed to create client for {provider_type.name}: {str(e)}")
        raise RuntimeError(f"Failed to create client for {provider_type.name}: {str(e)}") from e


def list_available_models() -> str:
    """
    Get a formatted string listing all available models from all providers.
    
    Returns:
        str: Formatted list of all available models
    """
    result = "Available Models:\n"
    
    for provider_type in ProviderType:
        # Skip if provider not registered or module not imported yet
        if provider_type not in CLIENT_REGISTRY:
            try:
                # Try to dynamically import the client module
                _import_client_module(provider_type)
            except ImportError:
                logger.warning(f"Provider {provider_type.name} module could not be imported")
                continue
        
        # Skip if still not registered after import attempt
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
                capabilities = f", Capabilities: {', '.join(model.capabilities)}" if model.capabilities else ""
                result += f"  - {key}{default_marker}: {model.name} ({model.description}{context_info}{capabilities})\n"
                
        except Exception as e:
            result += f"  Error retrieving models: {str(e)}\n"
    
    return result


@lru_cache(maxsize=8)
def get_available_models(provider: Union[str, ProviderType]) -> Dict[str, ModelInfo]:
    """
    Get available models for a specific provider. Results are cached.
    
    Args:
        provider: The provider name or ProviderType
        
    Returns:
        Dict[str, ModelInfo]: Dictionary of model_key -> model_info
    """
    # Convert string to ProviderType if needed
    provider_type = _resolve_provider_type(provider)
    
    # Check if provider is supported and loaded
    client_class = _get_client_class(provider_type)
    
    # Get available models
    try:
        return client_class.get_available_models()
    except Exception as e:
        logger.error(f"Error getting models for {provider_type.name}: {str(e)}")
        raise RuntimeError(f"Failed to get models for {provider_type.name}: {str(e)}") from e


def update_default_model(provider: Union[str, ProviderType], model_id: str) -> None:
    """
    Update the default model for a provider.
    
    Args:
        provider: The provider name or ProviderType
        model_id: The new default model ID
    """
    provider_type = _resolve_provider_type(provider)
    DEFAULT_MODELS[provider_type] = model_id
    logger.info(f"Updated default model for {provider_type.name} to {model_id}")


def _resolve_provider_type(provider: Union[str, ProviderType]) -> ProviderType:
    """
    Convert a string to ProviderType if needed.
    
    Args:
        provider: The provider name or ProviderType
        
    Returns:
        ProviderType: The resolved provider type
    """
    if isinstance(provider, str):
        try:
            return ProviderType(provider.lower())
        except ValueError:
            raise ValueError(f"Unknown provider: {provider}. "
                            f"Available providers: {', '.join(p.value for p in ProviderType)}")
    return provider


def _get_client_class(provider_type: ProviderType) -> Type[LLMClient]:
    """
    Get the client class for a provider type, importing if necessary.
    
    Args:
        provider_type: The provider type
        
    Returns:
        Type[LLMClient]: The client class
    """
    # Check if provider is in registry
    if provider_type not in CLIENT_REGISTRY:
        # Try to import the module
        try:
            _import_client_module(provider_type)
        except ImportError as e:
            raise ValueError(f"Provider {provider_type.name} is not available: {str(e)}")
    
    # Check again after import attempt
    if provider_type not in CLIENT_REGISTRY:
        raise ValueError(f"Provider {provider_type.name} is not registered after import")
    
    return CLIENT_REGISTRY[provider_type]


def _import_client_module(provider_type: ProviderType) -> None:
    """
    Dynamically import a client module by provider type.
    
    Args:
        provider_type: The provider type to import
    """
    # Map provider type to expected module name
    module_map = {
        ProviderType.ANTHROPIC: "anthropic_client",
        ProviderType.OPENAI: "openai_client",
        ProviderType.WATSONX: "watsonx_client",
        ProviderType.OLLAMA: "ollama_client"
    }
    
    if provider_type not in module_map:
        raise ImportError(f"No module mapping for provider {provider_type.name}")
    
    module_name = module_map[provider_type]
    
    try:
        # Import the module
        importlib.import_module(f".{module_name}", package="llm_clients")
    except ImportError as e:
        logger.error(f"Failed to import module for {provider_type.name}: {str(e)}")
        raise