"""
Enhanced Ollama client implementation.
"""
import os
import json
import logging
import requests
from typing import Dict, Optional, Generator, ClassVar, Any, List, Union
from pydantic import BaseModel, Field

from .llm_client import LLMClient
from .interfaces import ProviderType, ModelInfo, GenerationParams, Message, MessageRole, ClientConfig
from .factory import register_client
from .prompt_formatters import PromptFormatter

logger = logging.getLogger(__name__)


class OllamaConfig(ClientConfig):
    """Configuration options specific to Ollama."""
    base_url: str = Field("http://localhost:11434", description="Ollama server base URL")
    request_timeout: float = Field(60.0, description="Timeout for requests in seconds")
    use_mmap: Optional[bool] = Field(None, description="Use memory mapping for model loading")
    num_gpu: Optional[int] = Field(None, description="Number of GPUs to use")
    num_thread: Optional[int] = Field(None, description="Number of threads to use")
    use_formatter: bool = Field(True, description="Whether to use model-specific prompt formatting")


@register_client
class OllamaClient(LLMClient[OllamaConfig]):
    """Enhanced client for Ollama models."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.OLLAMA
    
    def __init__(self, model_id: str, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.
        
        Args:
            model_id: The model ID to use
            config: Optional configuration for the client
        """
        self.config = config or OllamaConfig()
        super().__init__(model_id)
        
        # Get prompt formatter for this model
        if self.config.use_formatter:
            self.formatter = PromptFormatter.get_formatter_for_model(model_id)
        else:
            self.formatter = PromptFormatter()
    
    def _validate_environment(self) -> None:
        """Validate environment configuration, overriding from env vars if needed."""
        # Get base URL from environment or use default
        env_base_url = os.getenv("OLLAMA_BASE_URL")
        if env_base_url and not self.config.base_url:
            self.config.base_url = env_base_url
    
    def _init_client(self) -> None:
        """Initialize client - for Ollama, there's no specific client initialization."""
        # Validate the base URL is reachable
        try:
            response = requests.get(
                f"{self.config.base_url}/api/version",
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            logger.info(f"Connected to Ollama server {response.json().get('version', 'unknown')}")
        except requests.RequestException as e:
            logger.warning(f"Could not connect to Ollama server: {str(e)}")
            logger.warning("Make sure Ollama is running and accessible")
    
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
            response = requests.get(
                f"{base_url}/api/tags",
                timeout=10.0
            )
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
                
                # Handle model metadata and capabilities
                model_type = "chat"  # Default type for most models
                capabilities = ["chat"]
                
                # Add specialized capabilities based on model family
                if "details" in model:
                    details = model.get("details", {})
                    model_family = details.get("family", "").lower()
                    
                    # Add specialized capabilities
                    if "code" in model_id.lower() or "codellama" in model_id.lower():
                        capabilities.append("code")
                        
                    if "vision" in model_id.lower():
                        capabilities.append("vision")
                        
                    # Infer context length from model details if available
                    context_length = None
                    if "parameter_size" in details:
                        # Very rough estimation based on model size
                        param_size = details.get("parameter_size", "")
                        if "b" in param_size.lower():
                            try:
                                size = float(param_size.lower().replace("b", ""))
                                if size <= 7:
                                    context_length = 4096
                                elif size <= 13:
                                    context_length = 8192
                                else:
                                    context_length = 16384
                            except (ValueError, TypeError):
                                pass
                else:
                    model_family = ""
                    context_length = None
                
                # Build description
                if model_family:
                    description = f"{key.capitalize()} - {model_family}"
                else:
                    description = f"{key.capitalize()} model via Ollama"
                
                # Add to dictionary
                models_dict[key] = ModelInfo(
                    name=key.capitalize(),
                    model_id=model_id,
                    description=description,
                    context_length=context_length,
                    provider=ProviderType.OLLAMA,
                    model_type=model_type,
                    capabilities=capabilities
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
        """Return default Ollama models with enhanced information."""
        return {
            "llama3": ModelInfo(
                name="Llama 3",
                model_id="llama3",
                description="Llama 3 by Meta (via Ollama)",
                context_length=8192,
                provider=ProviderType.OLLAMA,
                model_type="chat",
                capabilities=["chat"]
            ),
            "llama3:8b": ModelInfo(
                name="Llama 3 8B",
                model_id="llama3:8b",
                description="Llama 3 8B by Meta (via Ollama)",
                context_length=8192,
                provider=ProviderType.OLLAMA,
                model_type="chat",
                capabilities=["chat"]
            ),
            "llama3:70b": ModelInfo(
                name="Llama 3 70B",
                model_id="llama3:70b",
                description="Llama 3 70B by Meta (via Ollama)",
                context_length=8192,
                provider=ProviderType.OLLAMA,
                model_type="chat",
                capabilities=["chat"]
            ),
            "mistral": ModelInfo(
                name="Mistral",
                model_id="mistral",
                description="Mistral 7B by Mistral AI (via Ollama)",
                context_length=8192,
                provider=ProviderType.OLLAMA,
                model_type="chat",
                capabilities=["chat"]
            ),
            "mixtral": ModelInfo(
                name="Mixtral",
                model_id="mixtral",
                description="Mixtral 8x7B by Mistral AI (via Ollama)",
                context_length=32768,
                provider=ProviderType.OLLAMA,
                model_type="chat",
                capabilities=["chat"]
            ),
            "codellama": ModelInfo(
                name="CodeLlama",
                model_id="codellama",
                description="CodeLlama by Meta (via Ollama)",
                context_length=16384,
                provider=ProviderType.OLLAMA,
                model_type="chat",
                capabilities=["chat", "code"]
            ),
        }
    
    def _validate_params(self, params: GenerationParams) -> GenerationParams:
        """Validate and adjust parameters for Ollama API."""
        # No special validation needed for Ollama
        return params
    
    def _generate_impl(self, prompt: str, params: GenerationParams) -> str:
        """
        Implement generation for Ollama.
        
        Args:
            prompt: The text prompt to send to Ollama
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        # Format the prompt if configured to use formatters
        if self.config.use_formatter:
            formatted_prompt = self.formatter.format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        # Prepare request data
        data = self._prepare_request_data(formatted_prompt, params, stream=False)
        
        # Send request to Ollama API
        response = requests.post(
            f"{self.config.base_url}/api/generate",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=self.config.request_timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract generated text
        return result.get("response", "")
    
    def _generate_stream_impl(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """
        Implement streaming generation for Ollama.
        
        Args:
            prompt: The text prompt to send to Ollama
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        # Format the prompt if configured to use formatters
        if self.config.use_formatter:
            formatted_prompt = self.formatter.format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        # Prepare request data
        data = self._prepare_request_data(formatted_prompt, params, stream=True)
        
        # Send request to Ollama API
        response = requests.post(
            f"{self.config.base_url}/api/generate",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=self.config.request_timeout,
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
    
    def _prepare_request_data(self, prompt: str, params: GenerationParams, stream: bool) -> Dict[str, Any]:
        """
        Prepare request data for Ollama API.
        
        Args:
            prompt: The formatted prompt to send
            params: Generation parameters
            stream: Whether to stream the response
            
        Returns:
            Dict[str, Any]: Request data for Ollama API
        """
        data = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": stream
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
        
        # Add Ollama-specific parameters from config
        options = {}
        
        if self.config.num_gpu is not None:
            options["num_gpu"] = self.config.num_gpu
        if self.config.num_thread is not None:
            options["num_thread"] = self.config.num_thread
        if self.config.use_mmap is not None:
            options["use_mmap"] = self.config.use_mmap
        
        if options:
            data["options"] = options
        
        return data
    
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
        
        # Convert to Ollama message format
        api_messages = self._convert_messages(messages)
        
        # Prepare request data
        data = {
            "model": self.model_id,
            "messages": api_messages,
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
        
        # Add Ollama-specific parameters from config
        options = {}
        
        if self.config.num_gpu is not None:
            options["num_gpu"] = self.config.num_gpu
        if self.config.num_thread is not None:
            options["num_thread"] = self.config.num_thread
        if self.config.use_mmap is not None:
            options["use_mmap"] = self.config.use_mmap
        
        if options:
            data["options"] = options
        
        # Send request to Ollama API using chat endpoint if available
        try:
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=self.config.request_timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            return result.get("message", {}).get("content", "")
            
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            # Fall back to standard prompt method if chat endpoint fails
            logger.warning(f"Chat endpoint failed, falling back to standard generate: {str(e)}")
            formatted_prompt = self._messages_to_prompt(messages)
            return self.generate(formatted_prompt, params)
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Convert our Message objects to the format expected by Ollama API.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List[Dict[str, str]]: Messages in Ollama format
        """
        # Convert roles to Ollama's expected format
        role_mapping = {
            MessageRole.SYSTEM: "system",
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            # Function isn't supported by Ollama
        }
        
        api_messages = []
        
        for msg in messages:
            role = role_mapping.get(msg.role, msg.role)
            api_messages.append({
                "role": role,
                "content": msg.content
            })
        
        return api_messages
    
    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """
        Convert a list of messages to a simple prompt for fallback.
        
        Args:
            messages: List of messages to convert
            
        Returns:
            str: A prompt string representing the messages
        """
        # Format each message according to its role
        formatted_parts = []
        
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                formatted_parts.append(f"<system>\n{message.content}\n</system>")
            elif message.role == MessageRole.USER:
                formatted_parts.append(f"<human>\n{message.content}\n</human>")
            elif message.role == MessageRole.ASSISTANT:
                formatted_parts.append(f"<assistant>\n{message.content}\n</assistant>")
            else:
                formatted_parts.append(f"{message.role}: {message.content}")
        
        # Combine the parts
        prompt = "\n".join(formatted_parts)
        
        # Add final assistant prompt
        prompt += "\n<assistant>\n"
        
        return prompt
    
    def get_embeddings(self, texts: List[str], model_id: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings using Ollama's embedding endpoint.
        
        Args:
            texts: List of texts to embed
            model_id: Optional specific embedding model ID
            
        Returns:
            List[List[float]]: List of embeddings for each text
        """
        # Use specified model or default embedding model
        embed_model = model_id or self.model_id
        
        # Log which model we're using for embeddings
        logger.info(f"Generating embeddings using model: {embed_model}")
        
        embeddings = []
        
        try:
            for text in texts:
                # Based on successful curl test, use the /api/embed endpoint with 'input' parameter
                response = requests.post(
                    f"{self.config.base_url}/api/embed",
                    json={"model": embed_model, "input": text},
                    headers={"Content-Type": "application/json"},
                    timeout=self.config.request_timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                # The response has 'embeddings' (plural) key containing one embedding
                if "embeddings" in result and len(result["embeddings"]) > 0:
                    embeddings.append(result["embeddings"][0])
                else:
                    raise ValueError(f"No embeddings found in response: {result}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Ollama: {str(e)}")
            raise