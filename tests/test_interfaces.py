"""
Tests for the interfaces module.
"""
import unittest
from typing import Dict, Optional, List

import pytest
from pydantic import ValidationError

from llm_clients.interfaces import (
    ProviderType, 
    ModelInfo, 
    GenerationParams, 
    Message, 
    MessageRole,
    ClientConfig
)


class TestProviderType(unittest.TestCase):
    """Test cases for ProviderType enum."""
    
    def test_provider_type_values(self):
        """Test ProviderType enum values."""
        self.assertEqual(ProviderType.ANTHROPIC.value, "anthropic")
        self.assertEqual(ProviderType.OPENAI.value, "openai")
        self.assertEqual(ProviderType.WATSONX.value, "watsonx")
        self.assertEqual(ProviderType.OLLAMA.value, "ollama")
    
    def test_provider_type_from_string(self):
        """Test converting strings to ProviderType."""
        self.assertEqual(ProviderType("anthropic"), ProviderType.ANTHROPIC)
        self.assertEqual(ProviderType("openai"), ProviderType.OPENAI)
        self.assertEqual(ProviderType("watsonx"), ProviderType.WATSONX)
        self.assertEqual(ProviderType("ollama"), ProviderType.OLLAMA)
        
        # Test with uppercase
        self.assertEqual(ProviderType("ANTHROPIC".lower()), ProviderType.ANTHROPIC)


class TestModelInfo(unittest.TestCase):
    """Test cases for ModelInfo class."""
    
    def test_model_info_validation(self):
        """Test ModelInfo validation."""
        # Test invalid context_length
        with pytest.raises(ValidationError) as exc_info:
            ModelInfo(
                name="Invalid Model",
                model_id="invalid-model",
                provider=ProviderType.OPENAI,
                context_length=-100
            )
        
        # Check for any context_length error, not specific message
        errors = exc_info.value.errors()
        context_length_error = False
        for error in errors:
            if "context_length" in str(error["loc"]):
                context_length_error = True
                break
        self.assertTrue(context_length_error, "No error for context_length found")
        
    def test_model_info_immutability(self):
        """Test that ModelInfo instances are immutable."""
        # Create a model instance
        model = ModelInfo(
            name="Immutable Model",
            model_id="immutable-model",
            provider=ProviderType.OPENAI
        )
        
        # Attempting to modify frozen model should raise TypeError or AttributeError
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            model.name = "Modified Name"


class TestGenerationParams(unittest.TestCase):
    """Test cases for GenerationParams class."""
    
    def test_generation_params_defaults(self):
        """Test GenerationParams with default values."""
        params = GenerationParams()
        
        self.assertEqual(params.max_tokens, 1000)
        self.assertEqual(params.temperature, 0.7)
        self.assertIsNone(params.top_p)
        self.assertIsNone(params.top_k)
        self.assertIsNone(params.stop_sequences)
        self.assertIsNone(params.seed)
        self.assertIsNone(params.frequency_penalty)
        self.assertIsNone(params.presence_penalty)
    
    def test_generation_params_custom(self):
        """Test GenerationParams with custom values."""
        params = GenerationParams(
            max_tokens=2000,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP", "END"],
            seed=42,
            frequency_penalty=0.8,
            presence_penalty=0.2
        )
        
        self.assertEqual(params.max_tokens, 2000)
        self.assertEqual(params.temperature, 0.5)
        self.assertEqual(params.top_p, 0.9)
        self.assertEqual(params.top_k, 40)
        self.assertEqual(params.stop_sequences, ["STOP", "END"])
        self.assertEqual(params.seed, 42)
        self.assertEqual(params.frequency_penalty, 0.8)
        self.assertEqual(params.presence_penalty, 0.2)
    
    def test_generation_params_validation(self):
        """Test GenerationParams validation."""
        # Test invalid temperature (too high)
        with pytest.raises(ValidationError) as exc_info:
            GenerationParams(temperature=1.5)
        
        # Verify error for temperature
        errors = exc_info.value.errors()
        self.assertTrue(any(["temperature" in err["loc"] for err in errors]))
        
        # Test invalid temperature (too low)
        with pytest.raises(ValidationError) as exc_info:
            GenerationParams(temperature=-0.5)
        
        # Verify error for negative temperature
        errors = exc_info.value.errors()
        self.assertTrue(any(["temperature" in err["loc"] for err in errors]))
        
        # Test invalid max_tokens
        with pytest.raises(ValidationError) as exc_info:
            GenerationParams(max_tokens=0)
        
        # Verify error for max_tokens
        errors = exc_info.value.errors()
        self.assertTrue(any(["max_tokens" in err["loc"] for err in errors]))
        
        # Test invalid top_p (too high)
        with pytest.raises(ValidationError) as exc_info:
            GenerationParams(top_p=1.5)
        
        # Verify error for top_p
        errors = exc_info.value.errors()
        self.assertTrue(any(["top_p" in err["loc"] for err in errors]))
    
    def test_sampling_param_default(self):
        """Test that at least one sampling parameter is set."""
        # When both temperature and top_p are None
        params = GenerationParams(temperature=None, top_p=None)
        
        # Validator should set a default for temperature
        self.assertEqual(params.temperature, 0.7)


class TestMessage(unittest.TestCase):
    """Test cases for Message class."""
    
    def test_message_creation(self):
        """Test creating Message instances."""
        # Test with enum
        msg = Message(role=MessageRole.USER, content="Hello")
        self.assertEqual(msg.role, MessageRole.USER)
        self.assertEqual(msg.content, "Hello")
        self.assertIsNone(msg.name)
        
        # Test with string
        msg = Message(role="user", content="Hello again")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello again")
        
        # Test with name
        msg = Message(role=MessageRole.FUNCTION, content="Result", name="calculator")
        self.assertEqual(msg.role, MessageRole.FUNCTION)
        self.assertEqual(msg.content, "Result")
        self.assertEqual(msg.name, "calculator")
    
    def test_message_validation(self):
        """Test Message validation."""
        # Test empty content
        with pytest.raises(ValidationError) as exc_info:
            Message(role=MessageRole.USER, content="   ")
        
        assert "Message content cannot be empty" in str(exc_info.value)


class TestClientConfig(unittest.TestCase):
    """Test cases for ClientConfig class."""
    
    def test_client_config_defaults(self):
        """Test ClientConfig with default values."""
        config = ClientConfig()
        
        self.assertEqual(config.timeout, 30.0)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.backoff_factor, 0.5)
    
    def test_client_config_custom(self):
        """Test ClientConfig with custom values."""
        config = ClientConfig(
            timeout=60.0,
            max_retries=5,
            backoff_factor=1.0
        )
        
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.backoff_factor, 1.0)
    
    def test_client_config_extra_fields(self):
        """Test ClientConfig with extra fields."""
        config = ClientConfig(
            timeout=45.0,
            custom_field="custom_value"
        )
        
        self.assertEqual(config.timeout, 45.0)
        self.assertEqual(config.custom_field, "custom_value")


if __name__ == "__main__":
    unittest.main()