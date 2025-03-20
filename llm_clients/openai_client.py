"""
Enhanced OpenAI client implementation with support for all model types.
"""
import os
import json
import logging
from typing import Dict, Optional, Generator, ClassVar, List, Any, Set, Union, Type
from pydantic import BaseModel, Field, ValidationError

from openai import OpenAI
from .llm_client import LLMClient
from .interfaces import ProviderType, ModelInfo, GenerationParams, Message, MessageRole, ClientConfig
from .factory import register_client

logger = logging.getLogger(__name__)

# Models known to be completion-only models
COMPLETION_MODELS: Set[str] = {
    "babbage-002", 
    "davinci-002", 
    "gpt-3.5-turbo-instruct", 
    "gpt-3.5-turbo-instruct-0914"
}

# Models with specific API patterns
SPECIAL_MODEL_PREFIXES: List[str] = [
    "o1",  # Claude Opus models via Azure OpenAI
    "o3",  # Claude models via OpenAI
    "trt-llm"  # TensorRT-LLM models
]


class OpenAIConfig(ClientConfig):
    """Configuration options specific to OpenAI."""
    api_key: Optional[str] = Field(None, description="OpenAI API key")
    api_base: Optional[str] = Field(None, description="OpenAI API base URL")
    organization: Optional[str] = Field(None, description="OpenAI organization ID")
    default_system_prompt: str = Field(
        "You are a helpful assistant.", 
        description="Default system prompt to use when none is provided"
    )


@register_client
class OpenAIClient(LLMClient[OpenAIConfig]):
    """Enhanced client for OpenAI models."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.OPENAI
    
    def __init__(self, model_id: str, config: Optional[OpenAIConfig] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model_id: The model ID to use
            config: Optional configuration for the client
        """
        self.config = config or OpenAIConfig()
        self.client = None
        super().__init__(model_id)
        
        # Determine if this is a chat model or completion model
        self.is_chat_model = self._is_chat_model(model_id)
        logger.info(f"Using model {model_id} as a {'chat' if self.is_chat_model else 'completion'} model")
    
    def _validate_environment(self) -> None:
        """Validate required environment variables are set."""
        if not self.config.api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set and no API key provided in config")
            self.config.api_key = api_key
    
    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        client_kwargs = {"api_key": self.config.api_key}
        
        # Add optional parameters if provided
        if self.config.api_base:
            client_kwargs["base_url"] = self.config.api_base
            
        if self.config.organization:
            client_kwargs["organization"] = self.config.organization
            
        if self.config.timeout:
            client_kwargs["timeout"] = self.config.timeout
            
        if self.config.max_retries:
            client_kwargs["max_retries"] = self.config.max_retries
            
        self.client = OpenAI(**client_kwargs)
    
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
            # Check documentation for each special model
            # Default to chat for now, but might need updating for specific models
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
            client = OpenAI(api_key=api_key)
            models_dict = {}
            
            # Get models from API
            models = client.models.list()
            
            # Filter for models we want to use
            excluded_keywords = ["whisper", "tts", "dall-e", "embedding", "moderation", "deprecated"]
            
            for model in models.data:
                model_id = model.id
                
                # Skip models with excluded keywords
                if any(excluded in model_id.lower() for excluded in excluded_keywords):
                    continue
                
                # Add to our models dict
                key = model_id
                name = model_id
                
                # Better naming for common models
                if model_id.startswith("gpt-4"):
                    name = f"GPT-4 {model_id.replace('gpt-4-', '')}"
                    description = "GPT-4 model by OpenAI"
                    context_length = 128000 if "turbo" in model_id or "vision" in model_id else 8192
                    capabilities = ["chat"]
                    if "vision" in model_id:
                        capabilities.append("vision")
                elif model_id.startswith("gpt-3.5"):
                    name = f"GPT-3.5 {model_id.replace('gpt-3.5-', '')}"
                    description = "GPT-3.5 model by OpenAI"
                    context_length = 16384
                    capabilities = ["chat"]
                else:
                    description = f"OpenAI model: {model_id}"
                    context_length = None
                    capabilities = []
                
                # Add model type (chat or completion)
                is_chat = model_id not in COMPLETION_MODELS and "instruct" not in model_id
                model_type = "chat" if is_chat else "completion"
                
                # Add to dictionary
                models_dict[key] = ModelInfo(
                    name=name,
                    model_id=model_id,
                    description=description,
                    context_length=context_length,
                    provider=ProviderType.OPENAI,
                    model_type=model_type,
                    capabilities=capabilities
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
        """Return default OpenAI models with enhanced information."""
        return {
            "gpt-4-turbo": ModelInfo(
                name="GPT-4 Turbo",
                model_id="gpt-4-turbo",
                description="GPT-4 Turbo by OpenAI (Most capable)",
                context_length=128000,
                provider=ProviderType.OPENAI,
                model_type="chat",
                capabilities=["chat", "function-calling"]
            ),
            "gpt-4-vision-preview": ModelInfo(
                name="GPT-4 Vision",
                model_id="gpt-4-vision-preview",
                description="GPT-4 Vision by OpenAI (Vision capabilities)",
                context_length=128000,
                provider=ProviderType.OPENAI,
                model_type="chat",
                capabilities=["chat", "vision", "function-calling"]
            ),
            "gpt-4": ModelInfo(
                name="GPT-4",
                model_id="gpt-4",
                description="GPT-4 by OpenAI (Original)",
                context_length=8192,
                provider=ProviderType.OPENAI,
                model_type="chat",
                capabilities=["chat", "function-calling"]
            ),
            "gpt-3.5-turbo": ModelInfo(
                name="GPT-3.5 Turbo",
                model_id="gpt-3.5-turbo",
                description="GPT-3.5 Turbo by OpenAI (Fast)",
                context_length=16384,
                provider=ProviderType.OPENAI,
                model_type="chat",
                capabilities=["chat", "function-calling"]
            ),
            "gpt-3.5-turbo-instruct": ModelInfo(
                name="GPT-3.5 Turbo Instruct",
                model_id="gpt-3.5-turbo-instruct",
                description="GPT-3.5 Turbo Instruct by OpenAI",
                context_length=4096,
                provider=ProviderType.OPENAI,
                model_type="completion",
                capabilities=["completion"]
            )
        }
    
    def _validate_params(self, params: GenerationParams) -> GenerationParams:
        """Validate and adjust parameters for OpenAI API."""
        # No special validation needed for OpenAI
        return params
    
    def _generate_impl(self, prompt: str, params: GenerationParams) -> str:
        """
        Implement generation for OpenAI.
        
        Args:
            prompt: The text prompt to send to the model
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        # Call the appropriate endpoint based on model type
        if self.is_chat_model:
            return self._generate_chat(prompt, params)
        else:
            return self._generate_completion(prompt, params)
    
    def _generate_chat(self, prompt: str, params: GenerationParams) -> str:
        """Generate using the chat completions endpoint."""
        # Create messages with system prompt
        messages = [
            {"role": "system", "content": self.config.default_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Build API parameters
        api_params = self._build_chat_api_params(messages, params)
        
        # Call API
        response = self.client.chat.completions.create(**api_params)
        
        return response.choices[0].message.content or ""
    
    def _generate_completion(self, prompt: str, params: GenerationParams) -> str:
        """Generate using the completions endpoint."""
        # Build API parameters
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
            
        if params.frequency_penalty is not None:
            api_params["frequency_penalty"] = params.frequency_penalty
            
        if params.presence_penalty is not None:
            api_params["presence_penalty"] = params.presence_penalty
        
        # Call API
        response = self.client.completions.create(**api_params)
        
        return response.choices[0].text.strip()
    
    def _generate_stream_impl(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """
        Implement streaming generation for OpenAI.
        
        Args:
            prompt: The text prompt to send to the model
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        # Call the appropriate endpoint based on model type
        if self.is_chat_model:
            yield from self._generate_chat_stream(prompt, params)
        else:
            yield from self._generate_completion_stream(prompt, params)
    
    def _generate_chat_stream(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """Stream using the chat completions endpoint."""
        # Create messages with system prompt
        messages = [
            {"role": "system", "content": self.config.default_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Build API parameters
        api_params = self._build_chat_api_params(messages, params)
        api_params["stream"] = True
        
        # Stream response
        stream = self.client.chat.completions.create(**api_params)
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _generate_completion_stream(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """Stream using the completions endpoint."""
        # Build API parameters
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
            
        if params.frequency_penalty is not None:
            api_params["frequency_penalty"] = params.frequency_penalty
            
        if params.presence_penalty is not None:
            api_params["presence_penalty"] = params.presence_penalty
        
        # Stream response
        stream = self.client.completions.create(**api_params)
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text
    
    def _build_chat_api_params(self, messages: List[Dict[str, str]], params: GenerationParams) -> Dict[str, Any]:
        """Build API parameters for chat completions."""
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
            
        if params.frequency_penalty is not None:
            api_params["frequency_penalty"] = params.frequency_penalty
            
        if params.presence_penalty is not None:
            api_params["presence_penalty"] = params.presence_penalty
            
        return api_params
    
    def generate_with_messages(self, messages: List[Message], params: Optional[GenerationParams] = None) -> str:
        """
        Generate text based on a list of messages in a conversation.
        
        Args:
            messages: List of messages in the conversation
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        if not self.is_chat_model:
            # For completion models, convert messages to a single prompt
            return super().generate_with_messages(messages, params)
        
        if params is None:
            params = GenerationParams()
        
        params = self._validate_params(params)
        
        # Convert to OpenAI message format
        api_messages = self._convert_messages(messages)
        
        # Check if we need to add a system message
        has_system = any(msg["role"] == "system" for msg in api_messages)
        
        if not has_system:
            # Add default system prompt
            api_messages.insert(0, {"role": "system", "content": self.config.default_system_prompt})
        
        # Build API parameters
        api_params = self._build_chat_api_params(api_messages, params)
        
        # Call API
        response = self.client.chat.completions.create(**api_params)
        
        return response.choices[0].message.content or ""
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Convert our Message objects to the format expected by OpenAI API.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List[Dict[str, str]]: Messages in OpenAI format
        """
        api_messages = []
        
        for msg in messages:
            api_message = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Add name if provided (for function calls)
            if msg.name:
                api_message["name"] = msg.name
                
            api_messages.append(api_message)
        
        return api_messages
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using OpenAI's embedding models.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings for each text
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small", 
                input=texts
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            raise
    
    def generate_structured(self, 
                      prompt: str,
                      response_model: Type[BaseModel],
                      params: Optional[GenerationParams] = None) -> BaseModel:
        if params is None:
            params = GenerationParams()
        
        try:
            # Get JSON schema and create system message
            schema = response_model.model_json_schema()
            system_content = (
                f"Always respond with a valid JSON object that matches this exact schema:\n"
                f"{json.dumps(schema, indent=2)}\n"
                "Use proper JSON syntax with double quotes and correct data types."
            )
            
            # Build messages with schema instructions
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            # Create API parameters
            api_params = self._build_chat_api_params(messages, params)
            api_params.update({
                "response_format": {"type": "json_object"},
            })
            
            # Remove deprecated parameters that might conflict
            api_params.pop("functions", None)
            api_params.pop("function_call", None)
            api_params.pop("tools", None)
            
            # Call API
            response = self.client.chat.completions.create(**api_params)
            content = response.choices[0].message.content
            
            # Parse and validate
            parsed = json.loads(content)
            return response_model.model_validate(parsed)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {content}")
            raise ValueError(f"Failed to parse JSON response: {e}") from e
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e.errors()}")
            raise