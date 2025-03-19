"""
Tests for the base LLMClient class.
"""
import unittest
from unittest.mock import patch, MagicMock
from typing import ClassVar, Dict, Generator, Optional, List

import pytest
from pydantic import BaseModel

from llm_clients.llm_client import LLMClient
from llm_clients.interfaces import ProviderType, ModelInfo, GenerationParams


# Minimal concrete implementation for testing
class TestConfig(BaseModel):
    test_param: str = "default"


class TestClient(LLMClient[TestConfig]):
    """Minimal implementation of LLMClient for testing."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.OPENAI
    
    def __init__(self, model_id: str, config: Optional[TestConfig] = None):
        self.initialized = False
        self.validated = False
        self.config = config or TestConfig()
        super().__init__(model_id)
    
    def _validate_environment(self) -> None:
        """Test environment validation."""
        self.validated = True
    
    def _init_client(self) -> None:
        """Test client initialization."""
        self.initialized = True
    
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """Get test models."""
        return {
            "test-model": ModelInfo(
                name="Test Model",
                model_id="test-model",
                description="Test model for testing",
                provider=ProviderType.OPENAI
            )
        }
    
    def _generate_impl(self, prompt: str, params: GenerationParams) -> str:
        """Test generation implementation."""
        return f"Response to: {prompt}"
    
    def _generate_stream_impl(self, prompt: str, params: GenerationParams) -> Generator[str, None, None]:
        """Test streaming implementation."""
        for word in prompt.split():
            yield word + " "


class TestBaseClient(unittest.TestCase):
    """Test cases for the base LLMClient class."""
    
    def test_initialization(self):
        """Test the initialization process."""
        client = TestClient("test-model")
        
        # Check basic initialization
        self.assertEqual(client.model_id, "test-model")
        self.assertTrue(client.validated)
        self.assertTrue(client.initialized)
        
        # Check with custom config
        config = TestConfig(test_param="custom")
        client = TestClient("test-model", config)
        self.assertEqual(client.config.test_param, "custom")
    
    def test_generate(self):
        """Test the generate method."""
        client = TestClient("test-model")
        
        # Test with default params
        result = client.generate("Hello world")
        self.assertEqual(result, "Response to: Hello world")
        
        # Test with custom params
        params = GenerationParams(temperature=0.5, max_tokens=100)
        result = client.generate("Custom params", params)
        self.assertEqual(result, "Response to: Custom params")
    
    def test_generate_with_error(self):
        """Test error handling in generate."""
        client = TestClient("test-model")
        
        # Override implementation to raise an error
        with patch.object(TestClient, '_generate_impl', side_effect=ValueError("Test error")):
            with self.assertRaises(ValueError):
                client.generate("Error test")
    
    def test_generate_stream(self):
        """Test the generate_stream method."""
        client = TestClient("test-model")
        
        # Test streaming
        chunks = list(client.generate_stream("Hello streaming world"))
        self.assertEqual(chunks, ["Hello ", "streaming ", "world "])
    
    def test_context_manager(self):
        """Test context manager functionality."""
        # Create a client with mock close method
        client = TestClient("test-model")
        client.close = MagicMock()
        
        # Use as context manager
        with client as ctx:
            self.assertEqual(ctx, client)
        
        # Check close was called
        client.close.assert_called_once()
    
    def test_get_embeddings(self):
        """Test the get_embeddings method raises NotImplementedError."""
        client = TestClient("test-model")
        
        with self.assertRaises(NotImplementedError):
            client.get_embeddings(["Test"])
    
    def test_message_to_prompt_conversion(self):
        """Test conversion of messages to prompt."""
        from llm_clients.interfaces import Message, MessageRole
        
        client = TestClient("test-model")
        
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a test assistant."),
            Message(role=MessageRole.USER, content="Hello?"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        
        # Test the conversion
        prompt = client._messages_to_prompt(messages)
        
        # Check prompt format - the output format has changed in the new implementation
        self.assertIn("You are a test assistant.", prompt)
        self.assertIn("Hello?", prompt)
        self.assertIn("Hi there!", prompt)
    
    def test_generate_with_messages(self):
        """Test generate_with_messages method."""
        from llm_clients.interfaces import Message, MessageRole
        
        client = TestClient("test-model")
        
        # Patch _messages_to_prompt to verify it's called
        with patch.object(client, '_messages_to_prompt', return_value="Converted prompt") as mock_convert:
            messages = [
                Message(role=MessageRole.USER, content="Test message")
            ]
            
            result = client.generate_with_messages(messages)
            
            # Verify conversion was called
            mock_convert.assert_called_once_with(messages)
            
            # Verify generate was called with the converted prompt
            self.assertEqual(result, "Response to: Converted prompt")


if __name__ == "__main__":
    unittest.main()