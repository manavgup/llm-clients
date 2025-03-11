"""
IBM WatsonX client implementation with model-specific prompt formatting.
"""
import os
import logging
from typing import Dict, Optional, Generator, ClassVar, Any

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

from .llm_client import LLMClient
from .interfaces import ProviderType, ModelInfo, GenerationParams
from .factory import register_client
from .prompt_formatters import PromptFormatter

logger = logging.getLogger(__name__)


@register_client
class WatsonxClient(LLMClient):
    """Client for IBM WatsonX models."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.WATSONX
    
    def __init__(self, model_id: str):
        """
        Initialize WatsonX client.
        
        Args:
            model_id: The model ID to use
        """
        super().__init__(model_id)
        
        # Get credentials from environment
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        
        if not all([self.api_key, self.url, self.project_id]):
            raise ValueError("Missing required WatsonX credentials. Set WATSONX_API_KEY, WATSONX_URL, and WATSONX_PROJECT_ID environment variables.")
        
        # Initialize API client
        self.client: APIClient = APIClient(
            project_id=self.project_id,
            credentials=Credentials(api_key=self.api_key, url=self.url)
        )
        
        # GenParams for easy access
        self.GenParams = GenTextParamsMetaNames
        
        # Initialize model (will be created on first use)
        self.model = None
        
        # Get prompt formatter for this model
        self.formatter = PromptFormatter.get_formatter_for_model(model_id)
    
    def _get_model(self, params: Optional[Dict[Any, Any]] = None):
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
                persistent_connection=True,
                model_id=self.model_id,
                params=params,
                credentials=Credentials(api_key=self.api_key, url=self.url),
                project_id=self.project_id
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
            
            # Get text models from the TextModels enum - directly access the dictionary
            # Logs show this is returned as a dictionary by show() method
            model_specs = client.foundation_models.get_model_specs()
            for model_spec in model_specs['resources']:
                model_id = model_spec.get("model_id")
                # Create a key in the format that works with our system
                key = model_id.replace('/', '-').lower()
                
                # Extract model info
                name = model_spec.get("label", model_id.split("/")[-1])
                description = model_spec.get("short_description", f"{model_id} model")

                # Get context length if available
                context_length = None
                if "model_limits" in model_spec and "max_sequence_length" in model_spec["model_limits"]:
                    context_length = model_spec["model_limits"]["max_sequence_length"]

                models_dict[key] = ModelInfo(
                        name=name,
                        model_id=model_id,
                        description=description,
                        context_length=context_length,
                        provider=ProviderType.WATSONX
                    )

            return models_dict
        except Exception as e:
            logger.error(f"Error getting WatsonX models: {str(e)}")
            return cls._get_default_models()
    
    @staticmethod
    def _get_default_models() -> Dict[str, ModelInfo]:
        """Return default WatsonX models."""
        return {
            "ibm-granite-13b-instruct-v2": ModelInfo(
                name="Granite 13B Instruct",
                model_id="ibm/granite-13b-instruct-v2",
                description="Granite 13B Instruct v2 by IBM",
                provider=ProviderType.WATSONX
            ),
            "ibm-granite-3-8b-instruct": ModelInfo(
                name="Granite 3 8B Instruct",
                model_id="ibm/granite-3-8b-instruct",
                description="Granite 3 8B Instruct by IBM",
                provider=ProviderType.WATSONX
            ),
            "meta-llama-llama-3-3-70b-instruct": ModelInfo(
                name="Llama 3.3 70B",
                model_id="meta-llama/llama-3-3-70b-instruct",
                description="Llama 3.3 70B Instruct by Meta (via IBM)",
                provider=ProviderType.WATSONX
            ),
            "mistralai-mixtral-8x7b-instruct-v01": ModelInfo(
                name="Mixtral 8x7B",
                model_id="mistralai/mixtral-8x7b-instruct-v01",
                description="Mixtral 8x7B Instruct by Mistral AI (via IBM)",
                provider=ProviderType.WATSONX
            ),
            "mistralai-mistral-large": ModelInfo(
                name="Mistral Large",
                model_id="mistralai/mistral-large",
                description="Mistral Large by Mistral AI (via IBM)",
                provider=ProviderType.WATSONX
            ),
            "google-flan-ul2": ModelInfo(
                name="Flan-UL2",
                model_id="google/flan-ul2",
                description="Flan-UL2 by Google (via IBM)",
                provider=ProviderType.WATSONX
            )
        }
    
    def generate(self, prompt: str, params: Optional[GenerationParams] = None) -> str:
        """
        Generate text using WatsonX.
        
        Args:
            prompt: The text prompt to send to WatsonX
            params: Generation parameters
            
        Returns:
            str: The generated text response
        """
        try:
            # Use default parameters if none provided
            if params is None:
                params = GenerationParams()
            
            # Format the prompt based on the model type
            system_prompt = "You are an AI assistant that follows instructions precisely and accurately."
            formatted_prompt = self.formatter.format_prompt(prompt, system_prompt)
            
            # Log the formatted prompt for debugging
            logger.info(f"Using formatted prompt for {self.model_id}:")
            logger.info(f"{formatted_prompt[:200]}...")
            
            # Convert parameters to WatsonX format
            watson_params = {}
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
            
            # Get or create model
            model = self._get_model(watson_params)
            
            # Generate text - use the formatted prompt directly for all models
            response = model.generate_text(prompt=formatted_prompt)
            
            # Process the response
            if isinstance(response, dict):
                if 'results' in response:
                    result = response['results'][0]['generated_text'].strip()
                elif 'generated_text' in response:
                    result = response['generated_text'].strip()
                else:
                    result = response.strip() if isinstance(response, str) else str(response)
            else:
                result = response.strip() if isinstance(response, str) else str(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating text with WatsonX: {str(e)}")
            raise
    
    def generate_stream(self, prompt: str, params: Optional[GenerationParams] = None) -> Generator[str, None, None]:
        """
        Stream generation results from WatsonX.
        
        Args:
            prompt: The text prompt to send to WatsonX
            params: Generation parameters
            
        Returns:
            Generator yielding text chunks as they're generated
        """
        try:
            # Use default parameters if none provided
            if params is None:
                params = GenerationParams()
            
            # Format the prompt based on the model type
            system_prompt = "You are a helpful, accurate assistant that follows instructions precisely."
            formatted_prompt = self.formatter.format_prompt(prompt, system_prompt)
            
            # Convert parameters to WatsonX format
            watson_params = {}
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
            
            # Get or create model
            model = self._get_model(watson_params)
            
            # Generate text stream
            stream = model.generate_text_stream(prompt=formatted_prompt)
            
            for chunk in stream:
                if isinstance(chunk, dict) and 'generated_text' in chunk:
                    yield chunk['generated_text']
                else:
                    yield str(chunk)
                    
        except Exception as e:
            logger.error(f"Error streaming text with WatsonX: {str(e)}")
            raise
    
    def close(self):
        """Close the persistent connection if open."""
        if self.model:
            self.model.close_persistent_connection()
            logger.info("Closed WatsonX persistent connection")