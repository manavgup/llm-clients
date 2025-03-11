"""
OpenAI client implementation with support for both chat and completion models.
"""
import os
import logging
import openai
from typing import Dict, Optional, Generator, ClassVar, List, Set

from .llm_client import LLMClient
from .interfaces import ProviderType, ModelInfo, GenerationParams
from .factory import register_client

logger = logging.getLogger(__name__)

# Models known to be completion-only models
COMPLETION_MODELS: Set[str] = {
    "babbage-002", 
    "davinci-002", 
    "gpt-3.5-turbo-instruct", 
    "gpt-3.5-turbo-instruct-0914"
}

# Models known to use different API endpoint patterns
SPECIAL_MODEL_PREFIXES: List[str] = [
    "o1",
    "o3"
]

@register_client
class OpenAIClient(LLMClient):
    """Client for OpenAI models."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.OPENAI
    
    def __init__(self, model_id: str):
        """
        Initialize OpenAI client.
        
        Args:
            model_id: The model ID to use
        """
        super().__init__(model_id)
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Determine if this is a chat model or completion model
        self.is_chat_model = self._is_chat_model(model_id)
    
    def _is_chat_model(self, model_id: str) -> bool:
        """
        Determine if a model is a chat model or completion model.
        
        Args:
            model_id: The model ID to check
            
        Returns:
            bool: True if it's a chat model, False if it's a completion model
        """
        # Known completion models
        if model_id in COMPLETION_MODELS:
            return False
        
        # Special models that use different endpoint patterns
        if any(model_id.startswith(prefix) for prefix in SPECIAL_MODEL_PREFIXES):
            # Need to check API documentation for each model
            # Default to chat for now, but we might need to update this
            return True
        
        # Most GPT models are chat models
        if model_id.startswith("gpt-") and "instruct" not in model_id:
            return True
        
        # Default to chat model for unknown cases (most models are chat these days)
        return True
    
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for OpenAI.
        
        Returns:
            Dict[str, ModelInfo]: Dictionary of model_key -> model_info
        """
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, returning default models")
            return cls._get_default_models()
        
        try:
            client = openai.OpenAI(api_key=api_key)
            models_dict = {}
            
            # Get models from API
            models = client.models.list()
            
            # Filter for models we might want to use
            excluded_models = ["whisper", "tts", "dall-e", "embedding", "moderation"]
            
            for model in models.data:
                model_id = model.id
                
                # Skip non-completion models
                if any(excluded in model_id.lower() for excluded in excluded_models):
                    continue
                    
                # Skip deprecated models
                if "deprecated" in model_id.lower():
                    continue
                    
                # Add to our models dict
                key = model_id
                name = model_id
                
                # Better naming for common models
                if model_id.startswith("gpt-4"):
                    name = f"GPT-4 {model_id.replace('gpt-4-', '')}"
                    description = "GPT-4 model by OpenAI"
                    context_length = 128000 if "turbo" in model_id else 8192
                elif model_id.startswith("gpt-3.5"):
                    name = f"GPT-3.5 {model_id.replace('gpt-3.5-', '')}"
                    description = "GPT-3.5 model by OpenAI"
                    context_length = 16384
                else:
                    description = f"OpenAI model: {model_id}"
                    context_length = None
                
                # Add model type (chat or completion)
                is_chat = model_id not in COMPLETION_MODELS and "instruct" not in model_id
                model_type = "chat" if is_chat else "completion"
                if description and not description.endswith(")"):
                    description += f" ({model_type})"
                
                # Add to dictionary
                models_dict[key] = ModelInfo(
                    name=name,
                    model_id=model_id,
                    description=description,
                    context_length=context_length,
                    provider=ProviderType.OPENAI
                )
            
            # If no models were found (unlikely), use fallback
            if not models_dict:
                logger.warning("No OpenAI models discovered, using default models")
                return cls._get_default_models()
                
            return models_dict
            
        except Exception as e:
            logger.error(f"Error getting OpenAI models: {str(e)}")
            return cls._get_default_models()
    
    @staticmethod
    def _get_default_models() -> Dict[str, ModelInfo]:
        """Return default OpenAI models."""
        return {
            "gpt-4-turbo": ModelInfo(
                name="GPT-4 Turbo",
                model_id="gpt-4-turbo",
                description="GPT-4 Turbo by OpenAI (Most powerful) (chat)",
                context_length=128000,
                provider=ProviderType.OPENAI
            ),
            "gpt-4": ModelInfo(
                name="GPT-4",
                model_id="gpt-4",
                description="GPT-4 by OpenAI (Original) (chat)",
                context_length=8192,
                provider=ProviderType.OPENAI
            ),
            "gpt-3.5-turbo": ModelInfo(
                name="GPT-3.5 Turbo",
                model_id="gpt-3.5-turbo",
                description="GPT-3.5 Turbo by OpenAI (Fast) (chat)",
                context_length=16384,
                provider=ProviderType.OPENAI
            ),
            "gpt-3.5-turbo-instruct": ModelInfo(
                name="GPT-3.5 Turbo Instruct",
                model_id="gpt-3.5-turbo-instruct",
                description="GPT-3.5 Turbo Instruct by OpenAI (completion)",
                context_length=4096,
                provider=ProviderType.OPENAI
            )
        }
    
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """
        Generate text using OpenAI models.
        
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
            
            # Call the appropriate endpoint based on model type
            if self.is_chat_model:
                return self._generate_chat(prompt, params)
            else:
                return self._generate_completion(prompt, params)
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise
    
    def _generate_chat(self, prompt: str, params: GenerationParams) -> str:
        """Generate using the chat completions endpoint."""
        # Create messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build API parameters - only include parameters that are supported and have values
        api_params = {
            "model": self.model_id,
            "messages": messages,
        }
        
        # Add optional parameters if they have values
        if params.max_tokens is not None:
            api_params["max_tokens"] = params.max_tokens
            
        if params.temperature is not None:
            api_params["temperature"] = params.temperature
            
        if params.top_p is not None:
            api_params["top_p"] = params.top_p
            
        # Only add stop sequences if they exist and aren't None
        if params.stop_sequences and all(s is not None for s in params.stop_sequences):
            api_params["stop"] = params.stop_sequences
            
        if params.seed is not None:
            api_params["seed"] = params.seed
        
        # Call API
        response = self.client.chat.completions.create(**api_params)
        
        return response.choices[0].message.content
    
    def _generate_completion(self, prompt: str, params: GenerationParams) -> str:
        """Generate using the completions endpoint."""
        # Build API parameters - only include parameters that are supported and have values
        api_params = {
            "model": self.model_id,
            "prompt": prompt,
        }
        
        # Add optional parameters if they have values
        if params.max_tokens is not None:
            api_params["max_tokens"] = params.max_tokens
            
        if params.temperature is not None:
            api_params["temperature"] = params.temperature
            
        if params.top_p is not None:
            api_params["top_p"] = params.top_p
            
        # Only add stop sequences if they exist and aren't None
        if params.stop_sequences and all(s is not None for s in params.stop_sequences):
            api_params["stop"] = params.stop_sequences
            
        if params.seed is not None:
            api_params["seed"] = params.seed
        
        # Call API
        response = self.client.completions.create(**api_params)
        
        return response.choices[0].text.strip()
    
    def generate_stream(self, prompt: str, params: Optional[GenerationParams] = None) -> Generator[str, None, None]:
        """
        Stream generation results from OpenAI models.
        
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
            
            # Call the appropriate endpoint based on model type
            if self.is_chat_model:
                yield from self._generate_chat_stream(prompt, params)
            else:
                yield from self._generate_completion_stream(prompt, params)
                    
        except Exception as e:
            logger.error(f"Error streaming text with OpenAI: {str(e)}")
            raise
    
    def _generate_chat_stream(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """Stream using the chat completions endpoint."""
        # Create messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build API parameters - only include parameters that are supported and have values
        api_params = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
        }
        
        # Add optional parameters if they have values
        if params.max_tokens is not None:
            api_params["max_tokens"] = params.max_tokens
            
        if params.temperature is not None:
            api_params["temperature"] = params.temperature
            
        if params.top_p is not None:
            api_params["top_p"] = params.top_p
            
        # Only add stop sequences if they exist and aren't None
        if params.stop_sequences and all(s is not None for s in params.stop_sequences):
            api_params["stop"] = params.stop_sequences
            
        if params.seed is not None:
            api_params["seed"] = params.seed
        
        # Stream response
        stream = self.client.chat.completions.create(**api_params)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _generate_completion_stream(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """Stream using the completions endpoint."""
        # Build API parameters - only include parameters that are supported and have values
        api_params = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": True,
        }
        
        # Add optional parameters if they have values
        if params.max_tokens is not None:
            api_params["max_tokens"] = params.max_tokens
            
        if params.temperature is not None:
            api_params["temperature"] = params.temperature
            
        if params.top_p is not None:
            api_params["top_p"] = params.top_p
            
        # Only add stop sequences if they exist and aren't None
        if params.stop_sequences and all(s is not None for s in params.stop_sequences):
            api_params["stop"] = params.stop_sequences
            
        if params.seed is not None:
            api_params["seed"] = params.seed
        
        # Stream response
        stream = self.client.completions.create(**api_params)
        
        for chunk in stream:
            if chunk.choices[0].text:
                yield chunk.choices[0].text