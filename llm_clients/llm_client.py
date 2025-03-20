"""
Base LLM client definition with improved abstractions.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, ClassVar, List, Any, TypeVar, Generic, cast, Type

from pydantic import ValidationError, BaseModel

from .interfaces import ProviderType, ModelInfo, GenerationParams, Message

logger = logging.getLogger(__name__)

# Define a TypeVar for the client's configuration options
T = TypeVar('T')

class LLMClient(ABC, Generic[T]):
    """Abstract base class for LLM clients with improved abstractions."""
    
    # Class variables to be defined by subclasses
    provider_type: ClassVar[ProviderType]
    
    def __init__(self, model_id: str):
        """Initialize the LLM client with basic settings."""
        self.model_id = model_id
        self._validate_environment()
        self._init_client()
    
    @abstractmethod
    def _validate_environment(self) -> None:
        """
        Validate that the necessary environment variables are set.
        Raises ValueError if any required variables are missing.
        """
        pass
    
    @abstractmethod
    def _init_client(self) -> None:
        """
        Initialize the client with the provider's API.
        Should set up any necessary client instances and configurations.
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for this provider.
        
        Returns:
            Dict[str, ModelInfo]: Dictionary of model_key -> model_info
        """
        pass
    
    @abstractmethod
    def _generate_impl(self, prompt: str, params: GenerationParams) -> str:
        """
        Implement the provider-specific logic for text generation.
        
        Args:
            prompt: The text prompt to send to the model
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        pass
    
    @abstractmethod
    def _generate_stream_impl(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """
        Implement the provider-specific logic for streaming text generation.
        
        Args:
            prompt: The text prompt to send to the model
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        pass
    
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """
        Generate text based on the prompt. Includes common error handling.
        
        Args:
            prompt: The text prompt to send to the model
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        try:
            # Use default parameters if none provided
            if params is None:
                params = GenerationParams()
                
            # Validate params if needed
            params = self._validate_params(params)
            
            # Call the implementation
            return self._generate_impl(prompt, params)
        except Exception as e:
            logger.error(f"Error generating text with {self.provider_type.name}: {str(e)}")
            raise
    
    def generate_stream(self, prompt: str, params: Optional[GenerationParams] = None) -> Generator[str, None, None]:
        """
        Stream generation results. Includes common error handling.
        
        Args:
            prompt: The text prompt to send to the model
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        try:
            # Use default parameters if none provided
            if params is None:
                params = GenerationParams()
                
            # Validate params if needed
            params = self._validate_params(params)
            
            # Yield from the implementation
            yield from self._generate_stream_impl(prompt, params)
        except Exception as e:
            logger.error(f"Error streaming text with {self.provider_type.name}: {str(e)}")
            raise
    
    def _validate_params(self, params: GenerationParams) -> GenerationParams:
        """
        Validate and potentially modify generation parameters for the specific provider.
        
        Args:
            params: The parameters to validate
            
        Returns:
            GenerationParams: Validated (and potentially modified) parameters
        """
        # Default implementation just returns the params as is
        # Subclasses can override to add provider-specific validation
        return params
    
    def generate_with_messages(self, messages: List[Message], params: Optional[GenerationParams] = None) -> str:
        """
        Generate text based on a list of messages in a conversation.
        
        Args:
            messages: List of messages in the conversation
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        # Default implementation converts to a simple prompt
        # Subclasses should override this for chat-specific implementations
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, params)
    
    @abstractmethod
    def generate_structured(self, 
                          prompt: str, 
                          response_model: Type[BaseModel],
                          params: Optional[GenerationParams] = None) -> BaseModel:
        """Generate and parse structured response using Pydantic model"""
        pass
    
    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """
        Convert a list of messages to a simple prompt.
        
        Args:
            messages: List of messages to convert
            
        Returns:
            str: A prompt string representing the messages
        """
        # Basic implementation - subclasses can override for better formatting
        formatted_messages = []
        for message in messages:
            formatted_messages.append(f"{message.role}: {message.content}")
        return "\n".join(formatted_messages)
    
    def get_embeddings(self, texts: List[str], model_id: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            model_id: Optional embedding model ID. If None, uses the provider's default.

        Returns:
            List[List[float]]: Corresponding embeddings for each text.
        """
        # Default implementation raises NotImplementedError
        # Providers that support embeddings should override this
        raise NotImplementedError(
            f"Embeddings not implemented for {self.provider_type.name} provider"
        )
    
    def close(self) -> None:
        """Close any open connections. Implement if needed."""
        pass
    
    def __enter__(self) -> 'LLMClient[T]':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()