"""
Enhanced prompt formatters for different model families.
"""

import logging
from typing import Dict, Optional, Any, List, Type, ClassVar
from enum import Enum, auto

logger = logging.getLogger(__name__)


class FormatterType(Enum):
    """Types of prompt formatting styles."""
    DEFAULT = auto()
    LLAMA = auto()
    MISTRAL = auto()
    GRANITE = auto()
    FALCON = auto()
    SIMPLE = auto()


class PromptFormatter:
    """Base class for model-specific prompt formatters with enhanced typing."""
    
    formatter_type: ClassVar[FormatterType] = FormatterType.DEFAULT
    
    @classmethod
    def format_prompt(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format a prompt for the model.
        
        Args:
            prompt: The user prompt to format
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        return prompt
    
    @classmethod
    def format_messages(cls, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Format a list of messages for the model.
        
        Args:
            messages: List of messages with role and content keys
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        # Convert messages to a single prompt string with default implementation
        parts = []
        
        # Add system prompt if provided
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        # Add each message
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        
        return "\n\n".join(parts)
    
    @classmethod
    def get_formatter_for_model(cls, model_id: str) -> 'PromptFormatter':
        """
        Get the appropriate formatter for a given model ID.
        
        Args:
            model_id: The model ID to find a formatter for
            
        Returns:
            PromptFormatter: The appropriate formatter instance
        """
        model_id_lower = model_id.lower()
        
        # Map model ID patterns to formatters
        if "llama" in model_id_lower:
            return LlamaFormatter
        elif "granite" in model_id_lower:
            return GraniteFormatter
        elif "mistral" in model_id_lower or "mixtral" in model_id_lower:
            return MistralFormatter
        elif "falcon" in model_id_lower:
            return FalconFormatter
        elif "mpt" in model_id_lower:
            return SimpleFormatter
        else:
            # Default formatter for models without special formatting
            return PromptFormatter


class LlamaFormatter(PromptFormatter):
    """Formatter for Meta's LLaMA models."""
    
    formatter_type: ClassVar[FormatterType] = FormatterType.LLAMA
    
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
    
    @classmethod
    def format_messages(cls, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Format a list of messages for LLaMA models.
        
        Args:
            messages: List of messages with role and content keys
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        formatted_parts = []
        
        # Start with begin_of_text tag
        formatted_parts.append("<|begin_of_text|>")
        
        # Add system prompt if provided or use default
        default_system = "You are a helpful assistant that follows instructions precisely."
        system = system_prompt if system_prompt else default_system
        
        # Include system message
        formatted_parts.append(f"<|start_header_id|>system<|end_header_id|>{system}<|eot_id|>")
        
        # Process each message
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Skip system messages as we've handled them
            if role == "system":
                continue
                
            formatted_parts.append(f"<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>")
        
        # Add assistant header for the response
        formatted_parts.append("<|start_header_id|>assistant<|end_header_id|>")
        
        return "".join(formatted_parts)


class GraniteFormatter(PromptFormatter):
    """Formatter for IBM Granite models."""
    
    formatter_type: ClassVar[FormatterType] = FormatterType.GRANITE
    
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
        formatted_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        return formatted_prompt
    
    @classmethod
    def format_messages(cls, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Format a list of messages for Granite models.
        
        Args:
            messages: List of messages with role and content keys
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        formatted_parts = []
        
        # Add system prompt if provided or use default
        default_system = "You are a helpful assistant."
        system = system_prompt if system_prompt else default_system
        
        has_system_message = any(message.get("role") == "system" for message in messages)
        
        if not has_system_message:
            formatted_parts.append(f"<|system|>\n{system}")
        
        # Process each message
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Adapt role name to Granite format
            if role == "user":
                formatted_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|>\n{content}")
            elif role == "system" and not has_system_message:
                # Already handled system message if none in messages
                continue
            elif role == "system":
                formatted_parts.append(f"<|system|>\n{content}")
            else:
                # Unknown role, use best effort
                formatted_parts.append(f"<|{role}|>\n{content}")
        
        # Add assistant marker for response
        if not formatted_parts[-1].startswith("<|assistant|>"):
            formatted_parts.append("<|assistant|>\n")
        
        return "\n".join(formatted_parts)


class MistralFormatter(PromptFormatter):
    """Formatter for Mistral models."""
    
    formatter_type: ClassVar[FormatterType] = FormatterType.MISTRAL
    
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
        
        if system:
            formatted_prompt = f"<s>[INST] {system}\n\n{prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        return formatted_prompt
    
    @classmethod
    def format_messages(cls, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Format a list of messages for Mistral models.
        
        Args:
            messages: List of messages with role and content keys
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        formatted_parts = []
        current_inst = []
        
        # Prepare system instructions
        default_system = "You are a helpful assistant that follows instructions precisely."
        system = system_prompt if system_prompt else default_system
        
        has_system_message = any(message.get("role") == "system" for message in messages)
        
        # Construct the conversation with instruction tags
        for i, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                # Start a new instruction block
                if current_inst:
                    # Close previous instruction if it exists
                    formatted_parts.append(" [/INST] ")
                    current_inst = []
                
                # Start a new instruction
                if not formatted_parts:
                    # First message: include system prompt if no system message
                    if not has_system_message and system:
                        current_inst.append(f"<s>[INST] {system}\n\n{content}")
                    else:
                        current_inst.append(f"<s>[INST] {content}")
                else:
                    # Not the first message
                    current_inst.append(f"<s>[INST] {content}")
                
            elif role == "assistant":
                # Assistant response
                if current_inst:
                    # Close current instruction
                    formatted_parts.append("".join(current_inst) + " [/INST] ")
                    current_inst = []
                
                # Add assistant response
                formatted_parts.append(f"{content}")
                
            elif role == "system":
                # System message - add to next user message
                if i == 0:
                    system = content
            
        # If we have an unclosed instruction, close it
        if current_inst:
            formatted_parts.append("".join(current_inst) + " [/INST]")
        
        return "".join(formatted_parts)


class FalconFormatter(PromptFormatter):
    """Formatter for Falcon models."""
    
    formatter_type: ClassVar[FormatterType] = FormatterType.FALCON
    
    @classmethod
    def format_prompt(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format a prompt for Falcon models.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        # Falcon uses a simple prefix for system and user
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        parts.append(f"User: {prompt}")
        parts.append("Assistant: ")
        
        return "\n".join(parts)
    
    @classmethod
    def format_messages(cls, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Format a list of messages for Falcon models.
        
        Args:
            messages: List of messages with role and content keys
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        # Add explicit system prompt if provided
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        # Process messages
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                # Unknown role, use as-is
                parts.append(f"{role.capitalize()}: {content}")
        
        # Add final assistant prompt
        if not parts[-1].startswith("Assistant:"):
            parts.append("Assistant: ")
        
        return "\n".join(parts)


class SimpleFormatter(PromptFormatter):
    """Formatter for models that use simple role prefixes."""
    
    formatter_type: ClassVar[FormatterType] = FormatterType.SIMPLE
    
    @classmethod
    def format_prompt(cls, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format a prompt with simple role prefixes.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        
        return "\n".join(parts)
    
    @classmethod
    def format_messages(cls, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Format a list of messages with simple role prefixes.
        
        Args:
            messages: List of messages with role and content keys
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        # Add explicit system prompt if provided
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        
        # Process messages
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Add final assistant start
        if not messages or messages[-1].get("role", "") != "assistant":
            parts.append("<|im_start|>assistant\n")
        
        return "\n".join(parts)


# Registry of formatters by type
FORMATTER_REGISTRY: Dict[FormatterType, Type[PromptFormatter]] = {
    FormatterType.DEFAULT: PromptFormatter,
    FormatterType.LLAMA: LlamaFormatter,
    FormatterType.MISTRAL: MistralFormatter,
    FormatterType.GRANITE: GraniteFormatter,
    FormatterType.FALCON: FalconFormatter,
    FormatterType.SIMPLE: SimpleFormatter,
}


def get_formatter_by_type(formatter_type: FormatterType) -> Type[PromptFormatter]:
    """
    Get formatter class by formatter type.
    
    Args:
        formatter_type: The formatter type to get
        
    Returns:
        Type[PromptFormatter]: The formatter class
    """
    return FORMATTER_REGISTRY.get(formatter_type, PromptFormatter)