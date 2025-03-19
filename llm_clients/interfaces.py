"""
Type definitions and common interfaces for LLM clients with Pydantic 2.0 enhancements.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Literal, Union, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ProviderType(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    WATSONX = "watsonx"
    OLLAMA = "ollama"


class ModelInfo(BaseModel):
    """Information about an LLM model with improved validation."""
    name: str = Field(..., description="Display name of the model")
    model_id: str = Field(..., description="Provider-specific model identifier")
    description: str = Field("", description="Description of the model")
    context_length: Optional[int] = Field(None, description="Maximum context length in tokens")
    provider: ProviderType = Field(..., description="Provider of this model")
    model_type: Optional[str] = Field(None, description="Type of model (e.g., 'chat', 'completion', 'embedding')")
    capabilities: List[str] = Field(default_factory=list, description="List of model capabilities")
    
    @field_validator('context_length')
    @classmethod
    def validate_context_length(cls, v: Optional[int]) -> Optional[int]:
        """Validate context length is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Context length must be a positive integer")
        return v
    
    model_config = ConfigDict(
        frozen=True,  # Make instances immutable
        extra="forbid"  # Prevent extra attributes
    )


class GenerationParams(BaseModel):
    """Common parameters for text generation across providers with improved validation."""
    max_tokens: Optional[int] = Field(1000, description="Maximum number of tokens to generate", ge=1)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature (0-1)", ge=0.0, le=1.0)
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter (0-1)", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter", ge=0)
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty (0-2)", ge=0.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, description="Presence penalty (0-2)", ge=0.0, le=2.0)
    
    @model_validator(mode='after')
    def validate_at_least_one_sampling_param(self) -> 'GenerationParams':
        """Validate at least one sampling parameter (temperature or top_p) has a value."""
        if self.temperature is None and self.top_p is None:
            # Set a default for temperature if both are None
            self.temperature = 0.7
        return self
    
    model_config = ConfigDict(
        validate_default=True,  # Validate default values
        extra="forbid"  # Prevent extra attributes
    )


class MessageRole(str, Enum):
    """Standard roles for conversation messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Message(BaseModel):
    """A message in a conversation with enhanced validation."""
    role: Union[MessageRole, str] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Optional name of the sender (for function calls)")
    
    @field_validator('content')
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v
    
    model_config = ConfigDict(
        extra="forbid"  # Prevent extra attributes
    )


class ClientConfig(BaseModel):
    """Base configuration for LLM clients."""
    timeout: Optional[float] = Field(30.0, description="Timeout for API calls in seconds")
    max_retries: Optional[int] = Field(3, description="Maximum number of retries for API calls")
    backoff_factor: Optional[float] = Field(0.5, description="Backoff factor for retries")
    
    model_config = ConfigDict(
        extra="allow"  # Allow extra attributes for provider-specific configs
    )


class LLMResponse(BaseModel):
    """Standardized response from LLM generation."""
    text: str = Field(..., description="Generated text")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing generation")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")
    model: str = Field(..., description="Model used for generation")
    provider: ProviderType = Field(..., description="Provider used for generation")
    
    model_config = ConfigDict(
        frozen=True  # Make instances immutable
    )