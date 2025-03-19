"""
Tests for the Ollama client implementation.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import json

import pytest
import requests
from pydantic import ValidationError

from llm_clients.interfaces import ProviderType, GenerationParams, Message, MessageRole
from llm_clients.ollama_client import OllamaClient, OllamaConfig
from llm_clients.prompt_formatters import PromptFormatter


class TestOllamaClient(unittest.TestCase):
    """Test cases for the OllamaClient class."""
    
    @patch('requests.get')
    def setUp(self, mock_requests_get):
        """Set up test environment."""
        # Mock server version check
        mock_version_response = MagicMock()
        mock_version_response.json.return_value = {"version": "0.1.0"}
        mock_version_response.status_code = 200
        mock_requests_get.return_value = mock_version_response
        
        # Create client
        self.client = OllamaClient(model_id="llama3")
        
        # Setup formatter mock
        self.formatter_patcher = patch.object(self.client, 'formatter')
        self.mock_formatter = self.formatter_patcher.start()
        self.mock_formatter.format_prompt.side_effect = lambda x: f"Formatted: {x}"
    
    def tearDown(self):
        """Clean up after tests."""
        self.formatter_patcher.stop()
    
    def test_initialization(self):
        """Test initialization of OllamaClient."""
        # Test basic initialization
        self.assertEqual(self.client.model_id, "llama3")
        self.assertIsInstance(self.client.config, OllamaConfig)
        self.assertEqual(self.client.config.base_url, "http://localhost:11434")
        
        # Test with custom config
        config = OllamaConfig(
            base_url="http://custom-server:11434",
            request_timeout=120.0,
            num_gpu=2,
            use_formatter=False
        )
        
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"version": "0.1.0"}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            client = OllamaClient("mistral", config)
            
            self.assertEqual(client.model_id, "mistral")
            self.assertEqual(client.config.base_url, "http://custom-server:11434")
            self.assertEqual(client.config.request_timeout, 120.0)
            self.assertEqual(client.config.num_gpu, 2)
            self.assertFalse(client.config.use_formatter)
    
    def test_validate_environment(self):
        """Test environment validation."""
        # Test with environment variable
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env-server:11434"}, clear=True):
            # Create config with empty base_url
            config = OllamaConfig(base_url="")
            
            # Create a new client with patched _init_client to avoid actual initialization
            with patch.object(OllamaClient, '_init_client'):
                # Create a client with the config that has empty base_url
                client = OllamaClient("llama3", config)
                
                # Now manually call _validate_environment to test it's working correctly
                client._validate_environment()
                
                # Check if the environment variable was read
                self.assertEqual(client.config.base_url, "http://env-server:11434")
    
    @patch('requests.get')
    def test_get_available_models(self, mock_get):
        """Test get_available_models method."""
        # Mock tags response
        mock_tags_response = MagicMock()
        mock_tags_response.json.return_value = {
            "models": [
                {
                    "name": "llama3",
                    "details": {
                        "family": "llama",
                        "parameter_size": "8B"
                    }
                },
                {
                    "name": "mistral",
                    "details": {
                        "family": "mistral",
                        "parameter_size": "7B"
                    }
                },
                {
                    "name": "codellama:latest",
                    "details": {
                        "family": "codellama",
                        "parameter_size": "13B"
                    }
                }
            ]
        }
        mock_tags_response.status_code = 200
        mock_get.return_value = mock_tags_response
        
        # Test with default URL
        models = OllamaClient.get_available_models()
        
        # Verify API call
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=10.0
        )
        
        # Verify models
        self.assertIsInstance(models, dict)
        self.assertIn("llama3", models)
        self.assertIn("mistral", models)
        self.assertIn("codellama", models)
        
        # Don't check exact context length values as they might change
        self.assertIsNotNone(models["llama3"].context_length)
        
        self.assertEqual(models["llama3"].name, "Llama3")
        self.assertEqual(models["llama3"].provider, ProviderType.OLLAMA)
        self.assertIn("chat", models["llama3"].capabilities)
        
        self.assertEqual(models["codellama"].name, "Codellama")
        self.assertIn("code", models["codellama"].capabilities)
    
    @patch('requests.post')
    def test_generate(self, mock_post):
        """Test generate method."""
        # Mock generate response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "This is a test response from Ollama."}
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Test with default params
        result = self.client.generate("Test prompt")
        
        # Check result
        self.assertEqual(result, "This is a test response from Ollama.")
        
        # Verify prompt was formatted
        self.mock_formatter.format_prompt.assert_called_once_with("Test prompt")
        
        # Verify API call - use a more flexible check
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check the URL
        self.assertEqual(call_args[0][0], "http://localhost:11434/api/generate")
        
        # Check key parameters
        self.assertEqual(call_args[1]["json"]["model"], "llama3")
        self.assertEqual(call_args[1]["json"]["prompt"], "Formatted: Test prompt")
        self.assertEqual(call_args[1]["json"]["stream"], False)
    
    @patch('requests.post')
    def test_generate_stream(self, mock_post):
        """Test generate_stream method."""
        # Mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Create mock iter_lines that yields JSON strings
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "Hello"}).encode(),
            json.dumps({"response": " World"}).encode(),
            json.dumps({"response": "!"}).encode()
        ]
        
        mock_post.return_value = mock_response
        
        # Test streaming
        chunks = list(self.client.generate_stream("Test streaming"))
        
        # Check chunks
        self.assertEqual(chunks, ["Hello", " World", "!"])
        
        # Verify prompt was formatted
        self.mock_formatter.format_prompt.assert_called_once_with("Test streaming")
        
        # Verify API call - use more flexible check
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check the URL
        self.assertEqual(call_args[0][0], "http://localhost:11434/api/generate")
        
        # Check key parameters
        self.assertEqual(call_args[1]["json"]["model"], "llama3")
        self.assertEqual(call_args[1]["json"]["prompt"], "Formatted: Test streaming")
        self.assertEqual(call_args[1]["json"]["stream"], True)
    
    @patch('requests.post')
    def test_generate_with_messages(self, mock_post):
        """Test generate_with_messages method."""
        # Mock chat response
        mock_chat_response = MagicMock()
        mock_chat_response.json.return_value = {
            "message": {"content": "This is a response to the chat messages."}
        }
        mock_chat_response.status_code = 200
        
        # First call for chat endpoint
        mock_post.return_value = mock_chat_response
        
        # Create test messages
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a test assistant."),
            Message(role=MessageRole.USER, content="Hello?"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        
        # Test with messages
        result = self.client.generate_with_messages(messages)
        
        # Check result
        self.assertEqual(result, "This is a response to the chat messages.")
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check the URL
        self.assertEqual(call_args[0][0], "http://localhost:11434/api/chat")
        
        # Check that messages were properly converted
        api_messages = call_args[1]["json"]["messages"]
        self.assertEqual(len(api_messages), 3)
        self.assertEqual(api_messages[0]["role"], "system")
        self.assertEqual(api_messages[0]["content"], "You are a test assistant.")
        self.assertEqual(api_messages[1]["role"], "user")
        self.assertEqual(api_messages[1]["content"], "Hello?")
    
    @patch('requests.post')
    def test_get_embeddings(self, mock_post):
        """Test get_embeddings method."""
        # Mock embeddings responses
        mock_response1 = MagicMock()
        # Match the actual API response format with "embeddings" key (plural)
        mock_response1.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response1.status_code = 200
        
        mock_response2 = MagicMock()
        mock_response2.json.return_value = {"embeddings": [[0.4, 0.5, 0.6]]}
        mock_response2.status_code = 200
        
        mock_post.side_effect = [mock_response1, mock_response2]
        
        # Test embeddings
        texts = ["Text 1", "Text 2"]
        
        # Use an embedding-specific model
        with patch.object(self.client, 'model_id', "granite-embedding"):
            embeddings = self.client.get_embeddings(texts)
        
        # Check results
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])
        
        # Verify API calls
        self.assertEqual(mock_post.call_count, 2)
        
        # First call - check correct endpoint and parameters
        args1, kwargs1 = mock_post.call_args_list[0]
        self.assertEqual(args1[0], "http://localhost:11434/api/embed")  # Correct endpoint
        self.assertEqual(kwargs1["json"]["model"], "granite-embedding")
        self.assertEqual(kwargs1["json"]["input"], "Text 1")  # Using 'input' parameter
        
        # Second call
        args2, kwargs2 = mock_post.call_args_list[1]
        self.assertEqual(args2[0], "http://localhost:11434/api/embed")
        self.assertEqual(kwargs2["json"]["model"], "granite-embedding")
        self.assertEqual(kwargs2["json"]["input"], "Text 2")
        
        # Test with custom model
        mock_post.reset_mock()
        mock_post.side_effect = [MagicMock(json=lambda: {"embeddings": [[0.1, 0.2, 0.3]]})]
        
        self.client.get_embeddings(["Text"], model_id="mxbai-embed-large")
        
        # Check model ID was used
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "mxbai-embed-large")
        self.assertEqual(kwargs["json"]["input"], "Text")
    
    @patch('requests.post')
    def test_error_handling(self, mock_post):
        """Test error handling."""
        # Setup the mock to return an error
        mock_post.side_effect = requests.RequestException("Connection error")
        
        # Test generate method
        try:
            self.client.generate("Error test")
            self.fail("Expected exception but none was raised")
        except Exception as e:
            self.assertIn("Connection error", str(e))
        
        # Test stream method
        try:
            list(self.client.generate_stream("Stream error test"))
            self.fail("Expected exception but none was raised")
        except Exception as e:
            self.assertIn("Connection error", str(e))


if __name__ == "__main__":
    unittest.main()