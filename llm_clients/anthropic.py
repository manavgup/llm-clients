import os
import logging
import anthropic
from typing import Dict, Optional, Generator, ClassVar

from .llm_client import LLMClient
from .interfaces import ProviderType, ModelInfo, GenerationParams
from .factory import register_client

logger = logging.getLogger(__name__)


@register_client
class AnthropicClient(LLMClient):
    """Client for Anthropic Claude models."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.ANTHROPIC
    
    def __init__(self, model_id: str):
        """
        Initialize Anthropic Claude client.
        
        Args:
            model_id: The model ID to use
        """
        super().__init__(model_id)
        
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        # Initialize client
        self.client = anthropic.Anthropic(api_key=api_key)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for Anthropic.
        
        Returns:
            Dict[str, ModelInfo]: Dictionary of model_key -> model_info
        """
        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, returning default models")
            return cls._get_default_models()
        
        try:
            # Anthropic API doesn't currently have a models endpoint
            # If they add one in the future, we'd call it here
            # For now, return default models
            return cls._get_default_models()
            
        except Exception as e:
            logger.error(f"Error getting Anthropic models: {str(e)}")
            return cls._get_default_models()
    
    @staticmethod
    def _get_default_models() -> Dict[str, ModelInfo]:
        """Return default Anthropic models."""
        return {
            "claude-3-opus": ModelInfo(
                name="Claude 3 Opus",
                model_id="claude-3-opus-20240229",
                description="Claude 3 Opus by Anthropic (Most powerful)",
                context_length=200000,
                provider=ProviderType.ANTHROPIC
            ),
            "claude-3-sonnet": ModelInfo(
                name="Claude 3 Sonnet",
                model_id="claude-3-sonnet-20240229",
                description="Claude 3 Sonnet by Anthropic (Balanced)",
                context_length=180000,
                provider=ProviderType.ANTHROPIC
            ),
            "claude-3-haiku": ModelInfo(
                name="Claude 3 Haiku",
                model_id="claude-3-haiku-20240307",
                description="Claude 3 Haiku by Anthropic (Fast)",
                context_length=150000,
                provider=ProviderType.ANTHROPIC
            ),
            "claude-2": ModelInfo(
                name="Claude 2",
                model_id="claude-2.0",
                description="Claude 2 by Anthropic (Legacy)",
                context_length=100000,
                provider=ProviderType.ANTHROPIC
            )
        }
    
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """
        Generate text using Claude.
        
        Args:
            prompt: The text prompt to send to Claude
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        try:
            # Use default parameters if none provided
            if params is None:
                params = GenerationParams()
            
            # Create messages
            messages = [{"role": "user", "content": prompt}]
            
            # Build API parameters - only include parameters that Claude supports
            api_params = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": params.max_tokens,
            }
            
            # Only add supported parameters if they have values
            if params.temperature is not None:
                api_params["temperature"] = params.temperature
                
            if params.top_p is not None:
                api_params["top_p"] = params.top_p
                
            if params.stop_sequences:
                api_params["stop_sequences"] = params.stop_sequences
                
            # Claude doesn't support top_k parameter, so don't include it
            
            # Call API
            response = self.client.messages.create(**api_params)
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating text with Claude: {str(e)}")
            raise
    
    def generate_stream(self, prompt: str, params: Optional[GenerationParams] = None) -> Generator[str, None, None]:
        """
        Stream generation results from Claude.
        
        Args:
            prompt: The text prompt to send to Claude
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        try:
            # Use default parameters if none provided
            if params is None:
                params = GenerationParams()
            
            # Create messages
            messages = [{"role": "user", "content": prompt}]
            
            # Build API parameters - only include parameters that Claude supports
            api_params = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": params.max_tokens,
            }
            
            # Only add supported parameters if they have values
            if params.temperature is not None:
                api_params["temperature"] = params.temperature
                
            if params.top_p is not None:
                api_params["top_p"] = params.top_p
                
            if params.stop_sequences:
                api_params["stop_sequences"] = params.stop_sequences
            
            # Stream response
            with self.client.messages.stream(**api_params) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Error streaming text with Claude: {str(e)}")
            raise