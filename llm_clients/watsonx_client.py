"""
Enhanced IBM WatsonX client implementation with model-specific prompt formatting.
"""
import os
import logging
from typing import Dict, Optional, Generator, ClassVar, Any, List
from pydantic import BaseModel, Field

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

from .llm_client import LLMClient
from .interfaces import ProviderType, ModelInfo, GenerationParams, Message, MessageRole, ClientConfig
from .factory import register_client
from .prompt_formatters import PromptFormatter

logger = logging.getLogger(__name__)


class WatsonxConfig(ClientConfig):
    """Configuration options specific to IBM WatsonX."""
    api_key: Optional[str] = Field(None, description="WatsonX API key")
    url: Optional[str] = Field("https://us-south.ml.cloud.ibm.com", description="WatsonX API URL")
    project_id: Optional[str] = Field(None, description="WatsonX project ID")
    default_system_prompt: str = Field(
        "You are an AI assistant that follows instructions precisely and accurately.", 
        description="Default system prompt to use when none is provided"
    )
    persistent_connection: bool = Field(True, description="Whether to use a persistent connection")
    use_formatter: bool = Field(True, description="Whether to use model-specific prompt formatting")


@register_client
class WatsonxClient(LLMClient[WatsonxConfig]):
    """Enhanced client for IBM WatsonX models."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.WATSONX
    
    def __init__(self, model_id: str, config: Optional[WatsonxConfig] = None):
        """
        Initialize WatsonX client.
        
        Args:
            model_id: The model ID to use
            config: Optional configuration for the client
        """
        self.config = config or WatsonxConfig()
        self.client = None
        self.model = None
        self.GenParams = GenTextParamsMetaNames
        super().__init__(model_id)
        
        # Get prompt formatter if enabled
        if self.config.use_formatter:
            self.formatter = PromptFormatter.get_formatter_for_model(model_id)
        else:
            self.formatter = PromptFormatter()
    
    def _validate_environment(self) -> None:
        """Validate environment configuration, overriding from env vars if needed."""
        # Check API key
        if not self.config.api_key:
            api_key = os.getenv("WATSONX_API_KEY")
            if not api_key:
                raise ValueError("WATSONX_API_KEY environment variable not set and no API key provided in config")
            self.config.api_key = api_key
            
        # Check project ID
        if not self.config.project_id:
            project_id = os.getenv("WATSONX_PROJECT_ID")
            if not project_id:
                raise ValueError("WATSONX_PROJECT_ID environment variable not set and no project ID provided in config")
            self.config.project_id = project_id
            
        # Check URL (use default if not provided)
        if not self.config.url:
            url = os.getenv("WATSONX_URL")
            if url:
                self.config.url = url
    
    def _init_client(self) -> None:
        """Initialize WatsonX client."""
        # Initialize API client
        self.client = APIClient(
            project_id=self.config.project_id,
            credentials=Credentials(api_key=self.config.api_key, url=self.config.url)
        )
    
    def _get_model(self, params: Optional[Dict[Any, Any]] = None) -> ModelInference:
        """
        Get or create the model inference instance.
        
        Args:
            params: Generation parameters
            
        Returns:
            ModelInference: The model inference instance
        """
        if self.model is None:
            # Default parameters if none provided
            if params is None:
                params = {
                    self.GenParams.MAX_NEW_TOKENS: 512,
                    self.GenParams.MIN_NEW_TOKENS: 1,
                    self.GenParams.TEMPERATURE: 0.7,
                    self.GenParams.TOP_K: 50,
                    self.GenParams.RANDOM_SEED: 42,
                }
            
            model = ModelInference(
                persistent_connection=self.config.persistent_connection,
                model_id=self.model_id,
                params=params,
                credentials=Credentials(api_key=self.config.api_key, url=self.config.url),
                project_id=self.config.project_id
            )
            model.set_api_client(api_client=self.client)
            self.model = model
        
        return self.model
    
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for WatsonX.
        
        Returns:
            Dict[str, ModelInfo]: Dictionary of model_key -> model_info
        """
        # Check for credentials
        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        
        if not api_key or not project_id:
            logger.warning("WatsonX credentials not set, returning default models")
            return cls._get_default_models()
        
        try:
            # Initialize API client
            client = APIClient(
                project_id=project_id,
                credentials=Credentials(api_key=api_key, url=url)
            )
            
            models_dict = {}
            
            # Get model specs from the WatsonX API
            model_specs = client.foundation_models.get_model_specs()
            for model_spec in model_specs['resources']:
                model_id = model_spec.get("model_id")
                if not model_id:
                    continue
                    
                # Create a key in the format that works with our system
                key = model_id.replace('/', '-').lower()
                
                # Extract model info
                name = model_spec.get("label", model_id.split("/")[-1])
                description = model_spec.get("short_description", f"{model_id} model")

                # Get context length if available
                context_length = None
                if "model_limits" in model_spec and "max_sequence_length" in model_spec["model_limits"]:
                    context_length = model_spec["model_limits"]["max_sequence_length"]
                
                # Determine model type and capabilities
                model_type = "chat"  # Default to chat
                capabilities = ["chat"]
                
                # Add specialized capabilities based on model family
                if "code" in model_id.lower() or "starcoder" in model_id.lower():
                    capabilities.append("code")
                    
                if "flan" in model_id.lower():
                    capabilities.append("instruction-following")

                models_dict[key] = ModelInfo(
                    name=name,
                    model_id=model_id,
                    description=description,
                    context_length=context_length,
                    provider=ProviderType.WATSONX,
                    model_type=model_type,
                    capabilities=capabilities
                )

            # If no models were found, use default
            if not models_dict:
                logger.warning("No WatsonX models discovered, using default models")
                return cls._get_default_models()
                
            return models_dict
            
        except Exception as e:
            logger.error(f"Error getting WatsonX models: {str(e)}")
            return cls._get_default_models()
    
    @staticmethod
    def _get_default_models() -> Dict[str, ModelInfo]:
        """Return default WatsonX models with enhanced information."""
        return {
            "ibm-granite-13b-instruct-v2": ModelInfo(
                name="Granite 13B Instruct",
                model_id="ibm/granite-13b-instruct-v2",
                description="Granite 13B Instruct v2 by IBM",
                context_length=8192,
                provider=ProviderType.WATSONX,
                model_type="chat",
                capabilities=["chat", "instruction-following"]
            ),
            "ibm-granite-3-8b-instruct": ModelInfo(
                name="Granite 3 8B Instruct",
                model_id="ibm/granite-3-8b-instruct",
                description="Granite 3 8B Instruct by IBM",
                context_length=8192,
                provider=ProviderType.WATSONX,
                model_type="chat",
                capabilities=["chat", "instruction-following"]
            ),
            "meta-llama-llama-3-8b-instruct": ModelInfo(
                name="Llama 3 8B",
                model_id="meta-llama/llama-3-8b-instruct",
                description="Llama 3 8B Instruct by Meta (via IBM)",
                context_length=8192,
                provider=ProviderType.WATSONX,
                model_type="chat",
                capabilities=["chat"]
            ),
            "meta-llama-llama-3-70b-instruct": ModelInfo(
                name="Llama 3 70B",
                model_id="meta-llama/llama-3-70b-instruct",
                description="Llama 3 70B Instruct by Meta (via IBM)",
                context_length=8192,
                provider=ProviderType.WATSONX,
                model_type="chat",
                capabilities=["chat"]
            ),
            "mistralai-mixtral-8x7b-instruct-v01": ModelInfo(
                name="Mixtral 8x7B",
                model_id="mistralai/mixtral-8x7b-instruct-v01",
                description="Mixtral 8x7B Instruct by Mistral AI (via IBM)",
                context_length=32768,
                provider=ProviderType.WATSONX,
                model_type="chat",
                capabilities=["chat"]
            ),
            "google-flan-ul2": ModelInfo(
                name="Flan-UL2",
                model_id="google/flan-ul2",
                description="Flan-UL2 by Google (via IBM)",
                context_length=2048,
                provider=ProviderType.WATSONX,
                model_type="chat",
                capabilities=["chat", "instruction-following"]
            )
        }
    
    def _validate_params(self, params: GenerationParams) -> GenerationParams:
        """Validate and adjust parameters for WatsonX API."""
        # No special validation needed for WatsonX
        return params
    
    def _generate_impl(self, prompt: str, params: GenerationParams) -> str:
        """
        Implement generation for WatsonX.
        
        Args:
            prompt: The text prompt to send to WatsonX
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        # Format the prompt based on the model type
        formatted_prompt = self.formatter.format_prompt(prompt, self.config.default_system_prompt)
        
        # Log the formatted prompt for debugging
        logger.debug(f"Using formatted prompt for {self.model_id}:")
        logger.debug(f"{formatted_prompt[:200]}...")
        
        # Convert parameters to WatsonX format
        watson_params = self._convert_params_to_watson_format(params)
        
        # Get or create model
        model = self._get_model(watson_params)
        
        # Generate text - use the formatted prompt directly for all models
        response = model.generate_text(prompt=formatted_prompt)
        
        # Process the response
        return self._process_response(response)
    
    def _generate_stream_impl(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """
        Implement streaming generation for WatsonX.
        
        Args:
            prompt: The text prompt to send to WatsonX
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        # Format the prompt based on the model type
        formatted_prompt = self.formatter.format_prompt(prompt, self.config.default_system_prompt)
        
        # Convert parameters to WatsonX format
        watson_params = self._convert_params_to_watson_format(params)
        
        # Get or create model
        model = self._get_model(watson_params)
        
        # Generate text stream
        stream = model.generate_text_stream(prompt=formatted_prompt)
        
        for chunk in stream:
            if isinstance(chunk, dict) and 'generated_text' in chunk:
                yield chunk['generated_text']
            else:
                yield str(chunk)
    
    def _convert_params_to_watson_format(self, params: GenerationParams) -> Dict[Any, Any]:
        """
        Convert GenerationParams to WatsonX format.
        
        Args:
            params: Generation parameters
            
        Returns:
            Dict[Any, Any]: Parameters in WatsonX format
        """
        watson_params = {}
        
        # Map parameters to WatsonX format
        if params.max_tokens is not None:
            watson_params[self.GenParams.MAX_NEW_TOKENS] = params.max_tokens
            watson_params[self.GenParams.MIN_NEW_TOKENS] = 1
            
        if params.temperature is not None:
            watson_params[self.GenParams.TEMPERATURE] = params.temperature
            
        if params.top_p is not None:
            watson_params[self.GenParams.TOP_P] = params.top_p
            
        if params.top_k is not None:
            watson_params[self.GenParams.TOP_K] = params.top_k
            
        if params.seed is not None:
            watson_params[self.GenParams.RANDOM_SEED] = params.seed
            
        # Add any additional parameters that WatsonX supports
        # Currently there are no other mappings, but they can be added here
        
        return watson_params
    
    def _process_response(self, response: Any) -> str:
        """
        Process WatsonX response to extract the generated text.
        
        Args:
            response: The response from WatsonX
            
        Returns:
            str: The extracted text
        """
        if isinstance(response, dict):
            if 'results' in response and response['results']:
                return response['results'][0]['generated_text'].strip()
            elif 'generated_text' in response:
                return response['generated_text'].strip()
            else:
                return str(response).strip()
        else:
            return str(response).strip()
    
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
        
        # Extract system message if present
        system_content = None
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
                break
        
        # Use configured system prompt if no system message
        if not system_content:
            system_content = self.config.default_system_prompt
        
        # Format messages using the appropriate formatter
        formatted_prompt = self.formatter.format_messages(
            self._convert_messages(messages),
            system_content
        )
        
        # Log the formatted prompt for debugging
        logger.debug(f"Using formatted messages for {self.model_id}:")
        logger.debug(f"{formatted_prompt[:200]}...")
        
        # Convert parameters to WatsonX format
        watson_params = self._convert_params_to_watson_format(params)
        
        # Get or create model
        model = self._get_model(watson_params)
        
        # Generate text
        response = model.generate_text(prompt=formatted_prompt)
        
        # Process the response
        return self._process_response(response)
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Convert our Message objects to the format expected by formatter.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List[Dict[str, str]]: Messages in expected format
        """
        return [{"role": str(msg.role), "content": msg.content} for msg in messages]
    
    def get_embeddings(self, texts: List[str], model_id: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings using a specified model.
        
        Args:
            texts: List of texts to embed
            model_id: Optional embedding model ID
            
        Returns:
            List[List[float]]: List of embeddings for each text
        """
        # WatsonX doesn't currently provide a simple embeddings API
        # but we could implement this in the future
        raise NotImplementedError(
            "Embeddings are not yet supported for IBM WatsonX"
        )
    
    def close(self) -> None:
        """Close the persistent connection if open."""
        if self.model and self.config.persistent_connection:
            try:
                self.model.close_persistent_connection()
                logger.info("Closed WatsonX persistent connection")
            except Exception as e:
                logger.warning(f"Error closing WatsonX connection: {str(e)}")