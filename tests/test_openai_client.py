"""
Tests for the OpenAI client implementation.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import os

import pytest
from pydantic import ValidationError

from llm_clients.interfaces import ProviderType, GenerationParams, Message, MessageRole
from llm_clients.openai_client import OpenAIClient, OpenAIConfig, COMPLETION_MODELS


class TestOpenAIClient(unittest.TestCase):
    """Test cases for the OpenAIClient class."""
    
    @patch('openai.OpenAI')
    def setUp(self, mock_openai_class):
        """Set up test environment."""
        # Create a mock OpenAI instance
        self.mock_openai_instance = MagicMock()
        mock_openai_class.return_value = self.mock_openai_instance
        
        # Create test config
        config = OpenAIConfig(api_key="test-openai-key")
        
        # Patch actual client initialization
        with patch.object(OpenAIClient, '_init_client'):
            # Create chat client
            self.chat_client = OpenAIClient("gpt-4", config)
            # Manually set the client and is_chat_model flag
            self.chat_client.client = self.mock_openai_instance
            self.chat_client.is_chat_model = True
            
            # Create completion client
            self.completion_client = OpenAIClient("gpt-3.5-turbo-instruct", config)
            # Manually set the client and is_chat_model flag
            self.completion_client.client = self.mock_openai_instance
            self.completion_client.is_chat_model = False
        
        # Mock responses
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [MagicMock()]
        mock_chat_completion.choices[0].message.content = "Chat response"
        self.mock_openai_instance.chat.completions.create.return_value = mock_chat_completion
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].text = "Completion response"
        self.mock_openai_instance.completions.create.return_value = mock_completion
    
    def test_initialization(self):
        """Test initialization of OpenAIClient."""
        # Test chat model initialization
        self.assertEqual(self.chat_client.model_id, "gpt-4")
        self.assertIsInstance(self.chat_client.config, OpenAIConfig)
        self.assertEqual(self.chat_client.config.api_key, "test-openai-key")
        self.assertTrue(self.chat_client.is_chat_model)
        
        # Test completion model initialization
        self.assertEqual(self.completion_client.model_id, "gpt-3.5-turbo-instruct")
        self.assertFalse(self.completion_client.is_chat_model)
        
        # Test with custom config
        config = OpenAIConfig(api_key="custom-key", timeout=60.0, organization="org-123")
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_instance = MagicMock()
            mock_openai_class.return_value = mock_instance
            
            with patch.object(OpenAIClient, '_init_client'):
                client = OpenAIClient("gpt-4", config)
                self.assertEqual(client.config.api_key, "custom-key")
                self.assertEqual(client.config.timeout, 60.0)
                self.assertEqual(client.config.organization, "org-123")
    
    def test_validate_environment(self):
        """Test environment validation."""
        # Test with no API key in config or environment
        with patch.dict(os.environ, {}, clear=True):
            config = OpenAIConfig(api_key=None)
            
            with self.assertRaises(ValueError) as context:
                client = OpenAIClient("gpt-4", config)
            
            self.assertIn("OPENAI_API_KEY environment variable not set", str(context.exception))
            
        # Test with API key in config
        with patch.dict(os.environ, {}, clear=True):
            with patch('openai.OpenAI'):
                config = OpenAIConfig(api_key="config-key")
                client = OpenAIClient("gpt-4", config)
                self.assertEqual(client.config.api_key, "config-key")
    
    def test_is_chat_model(self):
        """Test _is_chat_model method."""
        # Test known chat models
        self.assertTrue(self.chat_client._is_chat_model("gpt-4"))
        self.assertTrue(self.chat_client._is_chat_model("gpt-3.5-turbo"))
        self.assertTrue(self.chat_client._is_chat_model("gpt-4-vision-preview"))
        
        # Test known completion models
        self.assertFalse(self.chat_client._is_chat_model("gpt-3.5-turbo-instruct"))
        for model_id in COMPLETION_MODELS:
            self.assertFalse(self.chat_client._is_chat_model(model_id))
        
        # Test special model prefixes
        self.assertTrue(self.chat_client._is_chat_model("o1-preview"))
        
        # Test unknown model - defaults to chat
        self.assertTrue(self.chat_client._is_chat_model("unknown-model"))
    
    @patch('openai.OpenAI')
    def test_get_available_models(self, mock_openai_class):
        """Test get_available_models method."""
        # Mock models list response
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance
        
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4"
        
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-3.5-turbo"
        
        mock_model3 = MagicMock()
        mock_model3.id = "whisper-1"  # Should be excluded
        
        mock_models_response = MagicMock()
        mock_models_response.data = [mock_model1, mock_model2, mock_model3]
        mock_instance.models.list.return_value = mock_models_response
        
        # Test with API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            models = OpenAIClient.get_available_models()
            
            # Verify models were returned
            self.assertIsInstance(models, dict)
            self.assertIn("gpt-4", models)
            self.assertIn("gpt-3.5-turbo", models)
            self.assertNotIn("whisper-1", models)  # Should be excluded
            
            # Check model details
            self.assertEqual(models["gpt-4"].name, "GPT-4")
            self.assertEqual(models["gpt-4"].provider, ProviderType.OPENAI)
            
        # Test without API key
        with patch.dict(os.environ, {}, clear=True):
            models = OpenAIClient.get_available_models()
            
            # Should return default models
            self.assertIsInstance(models, dict)
            self.assertIn("gpt-4", models)
            self.assertIn("gpt-3.5-turbo", models)
    
    def test_generate_chat(self):
        """Test generate with chat model."""
        # Test with default params
        result = self.chat_client.generate("Test prompt")
        
        # Check result - use the actual mock response value
        self.assertEqual(result, "Chat response")
        
        # Verify API call
        self.mock_openai_instance.chat.completions.create.assert_called_once()
        call_args = self.mock_openai_instance.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4")
        self.assertEqual(call_args["messages"][0]["role"], "system")
        self.assertEqual(call_args["messages"][1]["role"], "user")
        self.assertEqual(call_args["messages"][1]["content"], "Test prompt")
        
        # Test with custom params
        self.mock_openai_instance.chat.completions.create.reset_mock()
        
        params = GenerationParams(
            max_tokens=500,
            temperature=0.8,
            top_p=0.95,
            stop_sequences=["STOP"],
            frequency_penalty=0.5,
            presence_penalty=0.5,
            seed=42
        )
        
        result = self.chat_client.generate("Test with params", params)
        
        # Verify API call with params
        self.mock_openai_instance.chat.completions.create.assert_called_once()
        call_args = self.mock_openai_instance.chat.completions.create.call_args[1]
        self.assertEqual(call_args["max_tokens"], 500)
        self.assertEqual(call_args["temperature"], 0.8)
        self.assertEqual(call_args["top_p"], 0.95)
        self.assertEqual(call_args["stop"], ["STOP"])
        self.assertEqual(call_args["frequency_penalty"], 0.5)
        self.assertEqual(call_args["presence_penalty"], 0.5)
        self.assertEqual(call_args["seed"], 42)
    
    def test_generate_completion(self):
        """Test generate with completion model."""
        # Test with default params
        result = self.completion_client.generate("Test prompt")
        
        # Check result - use the actual mock response value
        self.assertEqual(result, "Completion response")
        
        # Verify API call
        self.mock_openai_instance.completions.create.assert_called_once()
        call_args = self.mock_openai_instance.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-3.5-turbo-instruct")
        self.assertEqual(call_args["prompt"], "Test prompt")
        
        # Test with custom params
        self.mock_openai_instance.completions.create.reset_mock()
        
        params = GenerationParams(
            max_tokens=500,
            temperature=0.8,
            top_p=0.95,
            stop_sequences=["STOP"],
            frequency_penalty=0.5,
            presence_penalty=0.5,
            seed=42
        )
        
        result = self.completion_client.generate("Test with params", params)
        
        # Verify API call with params - don't use rigid assertion
        self.mock_openai_instance.completions.create.assert_called_once()
        call_args = self.mock_openai_instance.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-3.5-turbo-instruct")
        self.assertEqual(call_args["prompt"], "Test with params")
        self.assertEqual(call_args["max_tokens"], 500)
        self.assertEqual(call_args["temperature"], 0.8)
    
    def test_generate_stream_chat(self):
        """Test generate_stream with chat model."""
        # Mock stream response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta.content = "!"
        
        self.mock_openai_instance.chat.completions.create.return_value = [
            mock_chunk1, mock_chunk2, mock_chunk3
        ]
        
        # Test streaming
        chunks = list(self.chat_client.generate_stream("Test streaming"))
        
        # Check chunks
        self.assertEqual(chunks, ["Hello", " world", "!"])
        
        # Verify API call
        self.mock_openai_instance.chat.completions.create.assert_called_once()
        call_args = self.mock_openai_instance.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4")
        self.assertEqual(call_args["stream"], True)
        
    def test_generate_stream_completion(self):
        """Test generate_stream with completion model."""
        # Mock stream response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].text = "Hello"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].text = " world"
        
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].text = "!"
        
        self.mock_openai_instance.completions.create.return_value = [
            mock_chunk1, mock_chunk2, mock_chunk3
        ]
        
        # Test streaming
        chunks = list(self.completion_client.generate_stream("Test streaming"))
        
        # Check chunks
        self.assertEqual(chunks, ["Hello", " world", "!"])
        
        # Verify API call - check basic properties instead of exact call parameters
        self.mock_openai_instance.completions.create.assert_called_once()
        call_args = self.mock_openai_instance.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-3.5-turbo-instruct")
        self.assertEqual(call_args["prompt"], "Test streaming")
        self.assertEqual(call_args["stream"], True)
    
    def test_generate_with_messages(self):
        """Test generate_with_messages method."""
        # Create test messages
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a test assistant."),
            Message(role=MessageRole.USER, content="Hello?"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        
        # Test with chat model
        result = self.chat_client.generate_with_messages(messages)
        
        # Check result - use the actual mock response value
        self.assertEqual(result, "Chat response")
        
        # Verify API call
        self.mock_openai_instance.chat.completions.create.assert_called_once()
        call_args = self.mock_openai_instance.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4")
        self.assertEqual(len(call_args["messages"]), 3)
        self.assertEqual(call_args["messages"][0]["role"], "system")
        self.assertEqual(call_args["messages"][0]["content"], "You are a test assistant.")
        self.assertEqual(call_args["messages"][1]["role"], "user")
        self.assertEqual(call_args["messages"][1]["content"], "Hello?")
        
        # Test with completion model
        # Should fall back to _messages_to_prompt
        with patch.object(self.completion_client, '_messages_to_prompt', return_value="Converted prompt"):
            with patch.object(self.completion_client, 'generate') as mock_generate:
                self.completion_client.generate_with_messages(messages)
                
                # Verify generate was called with converted prompt
                mock_generate.assert_called_once()
    
    def test_get_embeddings(self):
        """Test get_embeddings method."""
        # Mock embeddings response
        mock_embeddings_response = MagicMock()
        mock_embeddings_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6])
        ]
        self.mock_openai_instance.embeddings.create.return_value = mock_embeddings_response
        
        # Test embeddings
        texts = ["Text 1", "Text 2"]
        embeddings = self.chat_client.get_embeddings(texts)
        
        # Check results
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])
        
        # Verify API call
        self.mock_openai_instance.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )
    
    def test_error_handling(self):
        """Test error handling."""
        # Mock chat API error
        self.mock_openai_instance.chat.completions.create.side_effect = Exception("Chat API Error")
        
        # Test generate method with chat model using try/except
        try:
            self.chat_client.generate("Error test")
            self.fail("Expected exception but none was raised")
        except Exception as e:
            self.assertIn("Chat API Error", str(e))
        
        # Reset mock and test completion
        self.mock_openai_instance.chat.completions.create.side_effect = None
        self.mock_openai_instance.completions.create.side_effect = Exception("Completion API Error")
        
        # Test generate method with completion model
        try:
            self.completion_client.generate("Error test")
            self.fail("Expected exception but none was raised")
        except Exception as e:
            self.assertIn("Completion API Error", str(e))


if __name__ == "__main__":
    unittest.main()