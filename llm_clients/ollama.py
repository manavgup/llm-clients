"""
Ollama client implementation.
"""
import os
import json
import logging
import requests
from typing import Dict, Optional, Generator, ClassVar, Any

from .llm_client import LLMClient
from .interfaces import ProviderType, ModelInfo, GenerationParams
from .factory import register_client

logger = logging.getLogger(__name__)


@register_client
class OllamaClient(LLMClient):
    """Client for Ollama models."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.OLLAMA
    
    def __init__(self, model_id: str):
        """
        Initialize Ollama client.
        
        Args:
            model_id: The model ID to use
        """
        super().__init__(model_id)
        
        # Get base URL from environment or use default
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for Ollama.
        
        Returns:
            Dict[str, ModelInfo]: Dictionary of model_key -> model_info
        """
        # Get base URL from environment or use default
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        try:
            # Get models from Ollama API
            response = requests.get(f"{base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models_dict = {}
            
            if "models" not in data:
                logger.warning("Unexpected response format from Ollama API")
                return cls._get_default_models()
            
            for model in data["models"]:
                # Extract model name without version tag
                if "name" not in model:
                    continue
                    
                model_id = model["name"]
                key = model_id.split(":")[0]  # Remove any version tag
                
                # Build description
                if "details" in model and "family" in model["details"]:
                    description = f"{key.capitalize()} - {model['details']['family']}"
                else:
                    description = f"{key.capitalize()} model via Ollama"
                
                # Add to dictionary
                models_dict[key] = ModelInfo(
                    name=key.capitalize(),
                    model_id=model_id,
                    description=description,
                    provider=ProviderType.OLLAMA
                )
            
            # If no models were found, use default
            if not models_dict:
                logger.warning("No Ollama models discovered, using default models")
                return cls._get_default_models()
                
            return models_dict
            
        except Exception as e:
            logger.error(f"Error getting Ollama models: {str(e)}")
            return cls._get_default_models()
    
    @staticmethod
    def _get_default_models() -> Dict[str, ModelInfo]:
        """Return default Ollama models."""
        return {
            "llama3": ModelInfo(
                name="Llama 3",
                model_id="llama3",
                description="Llama 3 by Meta (via Ollama)",
                provider=ProviderType.OLLAMA
            ),
            "mistral": ModelInfo(
                name="Mistral",
                model_id="mistral",
                description="Mistral 7B by Mistral AI (via Ollama)",
                provider=ProviderType.OLLAMA
            ),
            "mixtral": ModelInfo(
                name="Mixtral",
                model_id="mixtral",
                description="Mixtral 8x7B by Mistral AI (via Ollama)",
                provider=ProviderType.OLLAMA
            ),
            "llama2": ModelInfo(
                name="Llama 2",
                model_id="llama2",
                description="Llama 2 by Meta (via Ollama)",
                provider=ProviderType.OLLAMA
            ),
            "codellama": ModelInfo(
                name="CodeLlama",
                model_id="codellama",
                description="CodeLlama by Meta (via Ollama)",
                provider=ProviderType.OLLAMA
            )
        }
    
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The text prompt to send to Ollama
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        try:
            # Use default parameters if none provided
            if params is None:
                params = GenerationParams()
            
            # Prepare request data
            data = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False
            }
            
            # Add optional parameters
            if params.max_tokens is not None:
                data["max_tokens"] = params.max_tokens
            if params.temperature is not None:
                data["temperature"] = params.temperature
            if params.top_p is not None:
                data["top_p"] = params.top_p
            if params.top_k is not None:
                data["top_k"] = params.top_k
            if params.stop_sequences:
                data["stop"] = params.stop_sequences
            
            # Send request to Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {str(e)}")
            raise
    
    def generate_stream(self, prompt: str, params: Optional[GenerationParams] = None) -> Generator[str, None, None]:
        """
        Stream generation results from Ollama.
        
        Args:
            prompt: The text prompt to send to Ollama
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        try:
            # Use default parameters if none provided
            if params is None:
                params = GenerationParams()
            
            # Prepare request data
            data = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": True
            }
            
            # Add optional parameters
            if params.max_tokens is not None:
                data["max_tokens"] = params.max_tokens
            if params.temperature is not None:
                data["temperature"] = params.temperature
            if params.top_p is not None:
                data["top_p"] = params.top_p
            if params.top_k is not None:
                data["top_k"] = params.top_k
            if params.stop_sequences:
                data["stop"] = params.stop_sequences
            
            # Send request to Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from Ollama: {line}")
                    
        except Exception as e:
            logger.error(f"Error streaming text with Ollama: {str(e)}")
            raise