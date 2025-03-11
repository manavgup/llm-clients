"""
Type definitions and common interfaces for LLM clients.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    WATSONX = "watsonx"
    OLLAMA = "ollama"


class ModelInfo(BaseModel):
    """Information about an LLM model."""
    name: str = Field(..., description="Display name of the model")
    model_id: str = Field(..., description="Provider-specific model identifier")
    description: str = Field("", description="Description of the model")
    context_length: Optional[int] = Field(None, description="Maximum context length in tokens")
    provider: ProviderType = Field(..., description="Provider of this model")


class GenerationParams(BaseModel):
    """Common parameters for text generation across providers."""
    max_tokens: Optional[int] = Field(1000, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature (0-1)")
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter (0-1)")
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class Message(BaseModel):
    """A message in a conversation."""
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")