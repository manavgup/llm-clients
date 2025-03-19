"""
Tests for the Anthropic client implementation.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import os

import pytest

from llm_clients.interfaces import ProviderType, GenerationParams, Message, MessageRole
from llm_clients.anthropic_client import AnthropicClient, AnthropicConfig


class TestAnthropicClient(unittest.TestCase):
    """Test cases for the AnthropicClient class."""
    
    @patch('anthropic.Anthropic')
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def setUp(self, mock_anthropic_class):
        """Set up test environment."""
        self.mock_anthropic_instance = MagicMock()
        mock_anthropic_class.return_value = self.mock_anthropic_instance
        
        # Mock message response
        self.mock_message_response = MagicMock()
        self.mock_message_response.content = [MagicMock()]
        self.mock_message_response.content[0].text = "This is a test response from Claude."
        self.mock_anthropic_instance.messages.create.return_value = self.mock_message_response
        
        # Create client
        self.client = AnthropicClient(model_id="claude-3-opus-20240229")
    
    def test_initialization(self):
        """Test initialization of AnthropicClient."""
        # Test basic initialization
        self.assertEqual(self.client.model_id, "claude-3-opus-20240229")
        self.assertIsInstance(self.client.config, AnthropicConfig)
        self.assertEqual(self.client.client, self.mock_anthropic_instance)
        
        # Test with custom config
        config = AnthropicConfig(api_key="custom-key", timeout=60.0)
        
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_instance = MagicMock()
            mock_anthropic_class.return_value = mock_instance
            
            client = AnthropicClient("claude-3-sonnet-20240229", config)
            
            # Check client was initialized with custom config
            mock_anthropic_class.assert_called_once_with(
                api_key="custom-key",
                timeout=60.0
            )
    
    def test_validate_environment(self):
        """Test environment validation."""
        # Test with no API key in config or environment
        with patch.dict(os.environ, {}, clear=True):
            config = AnthropicConfig(api_key=None)
            
            # Use a try/except block instead of assertRaises context
            try:
                client = AnthropicClient("claude-3-opus-20240229", config)
                self.fail("Expected ValueError but no exception was raised")
            except ValueError as e:
                self.assertIn("ANTHROPIC_API_KEY environment variable not set", str(e))
            
        # Test with API key in config
        with patch.dict(os.environ, {}, clear=True):
            with patch('anthropic.Anthropic'):
                config = AnthropicConfig(api_key="config-key")
                client = AnthropicClient("claude-3-opus-20240229", config)
                self.assertEqual(client.config.api_key, "config-key")
    
    @patch('anthropic.Anthropic')
    def test_get_available_models(self, mock_anthropic_class):
        """Test get_available_models method."""
        # Test with API key
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            models = AnthropicClient.get_available_models()
            
            # Verify models were returned
            self.assertIsInstance(models, dict)
            self.assertIn("claude-3-opus-20240229", models)
            self.assertEqual(models["claude-3-opus-20240229"].name, "Claude 3 Opus")
            
            # Check model capabilities
            self.assertIn("chat", models["claude-3-opus-20240229"].capabilities)
            self.assertIn("vision", models["claude-3-opus-20240229"].capabilities)
            
        # Test without API key
        with patch.dict(os.environ, {}, clear=True):
            models = AnthropicClient.get_available_models()
            
            # Should still return default models
            self.assertIsInstance(models, dict)
            self.assertIn("claude-3-opus-20240229", models)
    
    def test_validate_params(self):
        """Test parameter validation."""
        # Create params with unsupported parameter
        params = GenerationParams(max_tokens=100, temperature=0.7, top_k=40)
        
        # Validate params
        validated = self.client._validate_params(params)
        
        # Check that top_k was removed
        self.assertIsNone(validated.top_k)
        
        # Check other params were preserved
        self.assertEqual(validated.max_tokens, 100)
        self.assertEqual(validated.temperature, 0.7)
    
    def test_generate(self):
        """Test generate method."""
        # Test with default params
        result = self.client.generate("Test prompt")
        
        # Check result
        self.assertEqual(result, "This is a test response from Claude.")
        
        # Verify API call - use ANY for temperature to allow for default values
        self.mock_anthropic_instance.messages.create.assert_called_once()
        call_args = self.mock_anthropic_instance.messages.create.call_args
        self.assertEqual(call_args[1]["model"], "claude-3-opus-20240229")
        self.assertEqual(call_args[1]["messages"], [{"role": "user", "content": "Test prompt"}])
        self.assertEqual(call_args[1]["max_tokens"], 1000)
        
        # Test with custom params
        self.mock_anthropic_instance.messages.create.reset_mock()
        
        params = GenerationParams(
            max_tokens=500,
            temperature=0.8,
            top_p=0.95,
            stop_sequences=["STOP"],
            top_k=50  # Should be removed by validation
        )
        
        result = self.client.generate("Test with params", params)
        
        # Verify API call with validated params
        self.mock_anthropic_instance.messages.create.assert_called_once()
        call_args = self.mock_anthropic_instance.messages.create.call_args
        self.assertEqual(call_args[1]["model"], "claude-3-opus-20240229")
        self.assertEqual(call_args[1]["messages"], [{"role": "user", "content": "Test with params"}])
        self.assertEqual(call_args[1]["max_tokens"], 500)
        self.assertEqual(call_args[1]["temperature"], 0.8)
        self.assertEqual(call_args[1]["top_p"], 0.95)
        self.assertEqual(call_args[1]["stop_sequences"], ["STOP"])
    
    def test_generate_stream(self):
        """Test generate_stream method."""
        # Mock stream context manager
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.text_stream = ["Hello", " world", "!"]
        self.mock_anthropic_instance.messages.stream.return_value = mock_stream
        
        # Test streaming
        chunks = list(self.client.generate_stream("Test streaming"))
        
        # Check chunks
        self.assertEqual(chunks, ["Hello", " world", "!"])
        
        # Verify API call - using more flexible assertion
        self.mock_anthropic_instance.messages.stream.assert_called_once()
        call_args = self.mock_anthropic_instance.messages.stream.call_args
        self.assertEqual(call_args[1]["model"], "claude-3-opus-20240229")
        self.assertEqual(call_args[1]["messages"], [{"role": "user", "content": "Test streaming"}])
        self.assertEqual(call_args[1]["max_tokens"], 1000)
        
        # Test with custom params
        self.mock_anthropic_instance.messages.stream.reset_mock()
        
        params = GenerationParams(
            max_tokens=200,
            temperature=0.9,
            stop_sequences=["END"]
        )
        
        list(self.client.generate_stream("Test params streaming", params))
        
        # Verify API call with params
        self.mock_anthropic_instance.messages.stream.assert_called_once()
        call_args = self.mock_anthropic_instance.messages.stream.call_args
        self.assertEqual(call_args[1]["max_tokens"], 200)
        self.assertEqual(call_args[1]["temperature"], 0.9)
        self.assertEqual(call_args[1]["stop_sequences"], ["END"])
    
    def test_generate_with_messages(self):
        """Test generate_with_messages method."""
        # Create test messages
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a test assistant."),
            Message(role=MessageRole.USER, content="Hello?"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        
        # Test with messages
        result = self.client.generate_with_messages(messages)
        
        # Check result
        self.assertEqual(result, "This is a test response from Claude.")
        
        # Verify API call - check if the messages and system parameter match what we expect
        self.mock_anthropic_instance.messages.create.assert_called_once()
        call_args = self.mock_anthropic_instance.messages.create.call_args
        
        # The messages should not contain the system message
        api_messages = call_args[1]["messages"]
        self.assertEqual(len(api_messages), 2)  # Only user and assistant
        self.assertEqual(api_messages[0]["role"], "user")
        self.assertEqual(api_messages[1]["role"], "assistant")
        
    
    def test_error_handling(self):
        """Test error handling."""
        # Mock API error
        self.mock_anthropic_instance.messages.create.side_effect = Exception("API Error")
        
        # Test generate method - use try/except instead of assertRaises context
        try:
            self.client.generate("Error test")
            self.fail("Expected exception but none was raised")
        except Exception as e:
            self.assertIn("API Error", str(e))
        
        # Mock stream error
        self.mock_anthropic_instance.messages.stream.side_effect = Exception("Stream Error")
        
        # Test stream method
        try:
            list(self.client.generate_stream("Stream error test"))
            self.fail("Expected exception but none was raised")
        except Exception as e:
            self.assertIn("Stream Error", str(e))
    
    def test_embeddings_not_implemented(self):
        """Test that embeddings raise NotImplementedError."""
        try:
            self.client.get_embeddings(["Test text"])
            self.fail("Expected NotImplementedError but none was raised")
        except NotImplementedError as e:
            self.assertIn("not yet supported for Anthropic Claude", str(e))


if __name__ == "__main__":
    unittest.main()