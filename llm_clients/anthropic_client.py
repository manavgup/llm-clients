"""
Enhanced Anthropic Claude client implementation.
"""
import os
import logging
from typing import Dict, Optional, Generator, ClassVar, List, Any
from contextlib import asynccontextmanager

import anthropic
from pydantic import BaseModel, Field, model_validator

from .llm_client import LLMClient
from .interfaces import ProviderType, ModelInfo, GenerationParams, Message, MessageRole, ClientConfig
from .factory import register_client

logger = logging.getLogger(__name__)


class AnthropicConfig(ClientConfig):
    """Configuration options specific to Anthropic Claude."""
    api_key: Optional[str] = Field(None, description="Anthropic API key")
    api_url: Optional[str] = Field(None, description="Anthropic API URL")
    default_system_prompt: str = Field(
        "You are Claude, a helpful, harmless, and honest AI assistant.", 
        description="Default system prompt to use when none is provided"
    )
    # Future: proxy configuration, etc.


@register_client
class AnthropicClient(LLMClient[AnthropicConfig]):
    """Enhanced client for Anthropic Claude models."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.ANTHROPIC
    
    def __init__(self, model_id: str, config: Optional[AnthropicConfig] = None):
        """
        Initialize Anthropic Claude client.
        
        Args:
            model_id: The model ID to use
            config: Optional configuration for the client
        """
        self.config = config or AnthropicConfig()
        self.client = None
        super().__init__(model_id)
    
    def _validate_environment(self) -> None:
        """Validate required environment variables are set."""
        if not self.config.api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set and no API key provided in config")
            self.config.api_key = api_key
    
    def _init_client(self) -> None:
        """Initialize Anthropic client."""
        client_kwargs = {"api_key": self.config.api_key}
        
        # Add optional parameters if provided
        if self.config.api_url:
            client_kwargs["base_url"] = self.config.api_url
            
        if self.config.timeout:
            client_kwargs["timeout"] = self.config.timeout
            
        self.client = anthropic.Anthropic(**client_kwargs)
    
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
            # Create a client to fetch models
            client = anthropic.Anthropic(api_key=api_key)
            
            # Anthropic API doesn't currently have a models endpoint
            # If they add one in the future, we'd call it here
            # For now, return default models
            models = cls._get_default_models()
            
            # Try to validate if the models are available by making a minimal API call
            # This improves the reliability of the model list
            try:
                # Just verify API is working
                client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
            except Exception as e:
                logger.warning(f"API check failed, models may be unavailable: {str(e)}")
                
            return models
            
        except Exception as e:
            logger.error(f"Error getting Anthropic models: {str(e)}")
            return cls._get_default_models()
    
    @staticmethod
    def _get_default_models() -> Dict[str, ModelInfo]:
        """Return default Anthropic models with enhanced information."""
        return {
            "claude-3-opus-20240229": ModelInfo(
                name="Claude 3 Opus",
                model_id="claude-3-opus-20240229",
                description="Claude 3 Opus by Anthropic (Most capable model)",
                context_length=200000,
                provider=ProviderType.ANTHROPIC,
                model_type="chat",
                capabilities=["chat", "vision", "function-calling"]
            ),
            "claude-3-sonnet-20240229": ModelInfo(
                name="Claude 3 Sonnet",
                model_id="claude-3-sonnet-20240229",
                description="Claude 3 Sonnet by Anthropic (Balanced performance and speed)",
                context_length=180000,
                provider=ProviderType.ANTHROPIC,
                model_type="chat",
                capabilities=["chat", "vision", "function-calling"]
            ),
            "claude-3-haiku-20240307": ModelInfo(
                name="Claude 3 Haiku",
                model_id="claude-3-haiku-20240307",
                description="Claude 3 Haiku by Anthropic (Fastest, most compact)",
                context_length=150000,
                provider=ProviderType.ANTHROPIC,
                model_type="chat",
                capabilities=["chat", "vision"]
            ),
            "claude-3-5-sonnet-20240307": ModelInfo(
                name="Claude 3.5 Sonnet",
                model_id="claude-3-5-sonnet-20240307",
                description="Claude 3.5 Sonnet by Anthropic (Latest generation with best reasoning)",
                context_length=200000,
                provider=ProviderType.ANTHROPIC,
                model_type="chat",
                capabilities=["chat", "vision", "function-calling"]
            ),
            "claude-2.1": ModelInfo(
                name="Claude 2.1",
                model_id="claude-2.1",
                description="Claude 2.1 by Anthropic (Legacy)",
                context_length=100000,
                provider=ProviderType.ANTHROPIC,
                model_type="chat",
                capabilities=["chat"]
            )
        }
    
    def _validate_params(self, params: GenerationParams) -> GenerationParams:
        """Validate and adjust parameters for Anthropic API."""
        # Make a copy to avoid modifying the original
        params_dict = params.model_dump()
        
        # Claude doesn't support top_k parameter, so remove it if present
        if params.top_k is not None:
            logger.info("Removing unsupported top_k parameter for Claude API")
            params_dict["top_k"] = None
        
        # Recreate the GenerationParams with validated values
        return GenerationParams(**params_dict)
    
    def _generate_impl(self, prompt: str, params: GenerationParams) -> str:
        """
        Implement generation for Claude.
        
        Args:
            prompt: The text prompt to send to Claude
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
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
            
        # Call API
        response = self.client.messages.create(**api_params)
        
        return response.content[0].text
    
    def _generate_stream_impl(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """
        Implement streaming generation for Claude.
        
        Args:
            prompt: The text prompt to send to Claude
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
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
    
    def generate_with_messages(self, messages: List[Message], params: Optional[GenerationParams] = None) -> str:
        """
        Generate text based on a list of messages in a conversation.
        
        Args:
            messages: List of messages in the conversation
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        if params is None:
            params = GenerationParams()
        
        params = self._validate_params(params)
        
        # Convert to Anthropic message format
        api_messages = []
        
        # Check if we need to add a system message
        has_system = any(msg.role == MessageRole.SYSTEM for msg in messages)
        
        if not has_system:
            # Add default system prompt
            api_params = {
                "model": self.model_id,
                "messages": self._convert_messages(messages),
                "max_tokens": params.max_tokens,
                "system": self.config.default_system_prompt
            }
        else:
            # System message is already included in the messages
            api_params = {
                "model": self.model_id,
                "messages": self._convert_messages(messages),
                "max_tokens": params.max_tokens,
            }
        
        # Add optional parameters
        if params.temperature is not None:
            api_params["temperature"] = params.temperature
            
        if params.top_p is not None:
            api_params["top_p"] = params.top_p
            
        if params.stop_sequences:
            api_params["stop_sequences"] = params.stop_sequences
        
        # Call API
        response = self.client.messages.create(**api_params)
        
        return response.content[0].text
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert our Message objects to the format expected by Anthropic API.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List[Dict[str, Any]]: Messages in Anthropic format
        """
        api_messages = []
        
        for msg in messages:
            # Skip system messages, they're handled separately in Anthropic
            if msg.role == MessageRole.SYSTEM:
                continue
                
            api_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return api_messages
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using Claude Messages API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings for each text
        """
        # Claude embedding is not yet fully supported, but could be implemented
        # when Anthropic releases an embedding endpoint
        raise NotImplementedError(
            "Embeddings are not yet supported for Anthropic Claude"
        )