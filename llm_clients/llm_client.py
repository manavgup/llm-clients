"""
Base LLM client definition.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, ClassVar

from .interfaces import ProviderType, ModelInfo, GenerationParams

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    # Class variables to be defined by subclasses
    provider_type: ClassVar[ProviderType]
    
    def __init__(self, model_id: str):
        """Initialize the LLM client."""
        self.model_id = model_id
    
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
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: The text prompt to send to the model
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, params: Optional[GenerationParams] = None) -> Generator[str, None, None]:
        """
        Stream generation results.
        
        Args:
            prompt: The text prompt to send to the model
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        pass
    
    def close(self):
        """Close any open connections. Implement if needed."""
        pass