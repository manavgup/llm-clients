"""
Prompt formatters for different model families.
"""

import json
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class PromptFormatter:
    """Base class for model-specific prompt formatters."""
    
    @classmethod
    def format_prompt(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format a prompt for the model."""
        return prompt
    
    @classmethod
    def get_formatter_for_model(cls, model_id: str) -> 'PromptFormatter':
        """Get the appropriate formatter for a given model ID."""
        model_id_lower = model_id.lower()
        
        if "llama" in model_id_lower:
            return LlamaFormatter
        elif "granite" in model_id_lower:
            return GraniteFormatter
        elif "mistral" in model_id_lower or "mixtral" in model_id_lower:
            return MistralFormatter
        else:
            # Default formatter for models without special formatting
            return PromptFormatter


class LlamaFormatter(PromptFormatter):
    """Formatter for Meta's LLaMA models."""
    
    @classmethod
    def format_prompt(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format a prompt for LLaMA models.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        # For LLaMA 3.1 and higher
        default_system = "You are a helpful assistant that follows instructions precisely."
        system = system_prompt if system_prompt else default_system
        
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        return formatted_prompt


class GraniteFormatter(PromptFormatter):
    """Formatter for IBM Granite models."""
    
    @classmethod
    def format_prompt(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format a prompt for IBM Granite models.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        default_system = "You are a helpful assistant."
        system = system_prompt if system_prompt else default_system
        
        # Create a formatted prompt using Granite's expected format
        # Note: Removed the <|endoftext|> tokens as per official documentation
        formatted_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        return formatted_prompt


class MistralFormatter(PromptFormatter):
    """Formatter for Mistral models."""
    
    @classmethod
    def format_prompt(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format a prompt for Mistral models.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        # Format using the exact Mistral template
        default_system = "You are a helpful assistant that follows instructions precisely."
        system = system_prompt if system_prompt else default_system
        
        formatted_prompt = f"<s>[INST] {system}\n\n{prompt} [/INST]"
        
        return formatted_prompt