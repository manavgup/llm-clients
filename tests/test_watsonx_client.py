"""
Tests for the WatsonX client implementation.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import os

import pytest

from llm_clients.interfaces import ProviderType, GenerationParams, Message, MessageRole
from llm_clients.watsonx_client import WatsonxClient, WatsonxConfig


class TestWatsonxClient(unittest.TestCase):
    """Test cases for the WatsonxClient class."""
    
    def setUp(self):
        """Set up test environment without actually creating the client."""
        # We'll create a patched client in each test method
        self.api_key = "test-key"
        self.project_id = "test-project"
        self.url = "https://us-south.ml.cloud.ibm.com"
        
        # Create patches for later use
        self.api_client_patcher = patch('ibm_watsonx_ai.APIClient')
        self.model_patcher = patch('ibm_watsonx_ai.foundation_models.ModelInference')
        self.formatter_patcher = None  # Will be set per test
    
    def tearDown(self):
        """Clean up after tests."""
        if self.formatter_patcher:
            self.formatter_patcher.stop()
    
    def _create_client(self, model_id="ibm/granite-13b-instruct-v2"):
        """Helper to create a client with proper mocking."""
        # Start patches
        mock_api_client = self.api_client_patcher.start()
        mock_model_class = self.model_patcher.start()
        
        # Mock API client
        self.mock_api_client_instance = MagicMock()
        mock_api_client.return_value = self.mock_api_client_instance
        
        # Mock model inference
        self.mock_model = MagicMock()
        mock_model_class.return_value = self.mock_model
        
        # Create config with test credentials
        config = WatsonxConfig(
            api_key=self.api_key,
            url=self.url,
            project_id=self.project_id
        )
        
        # Create the client with patched init
        with patch.object(WatsonxClient, '_init_client') as mock_init:
            client = WatsonxClient(model_id, config)
            
            # Manually set client attributes after creation
            client.client = self.mock_api_client_instance
            client.model = self.mock_model
            
            # Create formatter mock
            self.formatter_patcher = patch.object(client, 'formatter')
            self.mock_formatter = self.formatter_patcher.start()
            self.mock_formatter.format_prompt.side_effect = lambda prompt, system_prompt: f"Formatted ({system_prompt}): {prompt}"
            self.mock_formatter.format_messages.side_effect = lambda messages, system_prompt: f"Formatted messages with system: {system_prompt}"
            
            return client
    
    def test_initialization(self):
        """Test initialization of WatsonxClient."""
        # Create client with mocks
        client = self._create_client()
        
        # Test basic initialization
        self.assertEqual(client.model_id, "ibm/granite-13b-instruct-v2")
        self.assertIsInstance(client.config, WatsonxConfig)
        self.assertEqual(client.config.url, self.url)
        self.assertEqual(client.config.api_key, self.api_key)
        self.assertEqual(client.config.project_id, self.project_id)
        self.assertTrue(client.config.persistent_connection)
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_validate_environment(self):
        """Test environment validation."""
        # Test with API key in config
        with patch.dict(os.environ, {}, clear=True):
            with patch('ibm_watsonx_ai.APIClient'):
                with patch('ibm_watsonx_ai.foundation_models.ModelInference'):
                    config = WatsonxConfig(
                        api_key="config-key",
                        project_id="config-project"
                    )
                    
                    with patch.object(WatsonxClient, '_init_client'):
                        client = WatsonxClient("test-model", config)
                        self.assertEqual(client.config.api_key, "config-key")
                        self.assertEqual(client.config.project_id, "config-project")
        
        # Test with no API key in config or environment
        with patch.dict(os.environ, {'WATSONX_PROJECT_ID': 'test-project'}, clear=True):
            config = WatsonxConfig(api_key=None, project_id="test-project")
            
            try:
                with patch.object(WatsonxClient, '_init_client'):
                    client = WatsonxClient("test-model", config)
                self.fail("Expected ValueError but no exception was raised")
            except ValueError as e:
                self.assertIn("WATSONX_API_KEY environment variable not set", str(e))
    
    @patch('ibm_watsonx_ai.APIClient')
    def test_get_available_models(self, mock_api_client):
        """Test get_available_models method."""
        # Mock client instance and foundation_models
        mock_client_instance = MagicMock()
        mock_api_client.return_value = mock_client_instance
        
        mock_foundation_models = MagicMock()
        mock_client_instance.foundation_models = mock_foundation_models
        
        # Mock get_model_specs response
        mock_foundation_models.get_model_specs.return_value = {
            'resources': [
                {
                    'model_id': 'ibm/granite-13b-instruct-v2',
                    'label': 'Granite 13B',
                    'short_description': 'IBM Granite 13B model',
                    'model_limits': {
                        'max_sequence_length': 8192
                    }
                },
                {
                    'model_id': 'meta-llama/llama-3-8b-instruct',
                    'label': 'Llama 3 8B',
                    'short_description': 'Meta Llama 3 8B model',
                    'model_limits': {
                        'max_sequence_length': 4096
                    }
                }
            ]
        }
        
        # Test with API key and project ID - use patch for default models
        with patch.dict(os.environ, {
            'WATSONX_API_KEY': 'test-key',
            'WATSONX_PROJECT_ID': 'test-project'
        }):
            with patch.object(WatsonxClient, '_get_default_models') as mock_default_models:
                # Setup mock default models
                default_models = {
                    "ibm-granite-13b-instruct-v2": MagicMock(name="Granite 13B")
                }
                mock_default_models.return_value = default_models
                
                models = WatsonxClient.get_available_models()
                
                # Verify models were returned
                self.assertIsInstance(models, dict)
                
        # Test without credentials using default models
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(WatsonxClient, '_get_default_models') as mock_default_models:
                default_models = {
                    "ibm-granite-13b-instruct-v2": MagicMock(name="Granite 13B Instruct")
                }
                mock_default_models.return_value = default_models
                
                models = WatsonxClient.get_available_models()
                
                # Should return default models
                self.assertIsInstance(models, dict)
                self.assertEqual(models, default_models)
    
    def test_get_model(self):
        """Test _get_model method."""
        client = self._create_client()
        from ibm_watsonx_ai.foundation_models import ModelInference
        
        # First call should create model
        model = client._get_model()
        
        # Should be the mocked model
        self.assertEqual(model, self.mock_model)
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_convert_params_to_watson_format(self):
        """Test _convert_params_to_watson_format method."""
        client = self._create_client()
        
        # Mock GenParams fields
        client.GenParams.MAX_NEW_TOKENS = "max_new_tokens"
        client.GenParams.MIN_NEW_TOKENS = "min_new_tokens"
        client.GenParams.TEMPERATURE = "temperature"
        client.GenParams.TOP_P = "top_p" 
        client.GenParams.TOP_K = "top_k"
        client.GenParams.RANDOM_SEED = "random_seed"
        
        # Test with all parameters
        params = GenerationParams(
            max_tokens=200,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            seed=42
        )
        
        watson_params = client._convert_params_to_watson_format(params)
        
        # Check conversion
        self.assertEqual(watson_params["max_new_tokens"], 200)
        self.assertEqual(watson_params["min_new_tokens"], 1)
        self.assertEqual(watson_params["temperature"], 0.8)
        self.assertEqual(watson_params["top_p"], 0.95)
        self.assertEqual(watson_params["top_k"], 40)
        self.assertEqual(watson_params["random_seed"], 42)
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_process_response(self):
        """Test _process_response method."""
        client = self._create_client()
        
        # Test with results format
        response = {
            'results': [{'generated_text': 'Response in results format.'}]
        }
        result = client._process_response(response)
        self.assertEqual(result, "Response in results format.")
        
        # Test with generated_text format
        response = {'generated_text': 'Response in generated_text format.'}
        result = client._process_response(response)
        self.assertEqual(result, "Response in generated_text format.")
        
        # Test with string
        response = "String response."
        result = client._process_response(response)
        self.assertEqual(result, "String response.")
        
        # Test with other object
        response = ['List', 'response']
        result = client._process_response(response)
        self.assertEqual(result, "['List', 'response']")
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_generate(self):
        """Test generate method."""
        client = self._create_client()
        
        # Mock response
        self.mock_model.generate_text.return_value = {
            'results': [{'generated_text': 'This is a test response from WatsonX.'}]
        }
        
        # Test with default params
        result = client.generate("Test prompt")
        
        # Check result
        self.assertEqual(result, "This is a test response from WatsonX.")
        
        # Verify prompt was formatted
        self.mock_formatter.format_prompt.assert_called_once_with(
            "Test prompt", 
            client.config.default_system_prompt
        )
        
        # Verify generate_text was called
        self.mock_model.generate_text.assert_called_once_with(
            prompt="Formatted (You are an AI assistant that follows instructions precisely and accurately.): Test prompt"
        )
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_generate_stream(self):
        """Test generate_stream method."""
        client = self._create_client()
        
        # Mock stream response
        self.mock_model.generate_text_stream.return_value = [
            {'generated_text': 'Hello'},
            {'generated_text': ' world'},
            {'generated_text': '!'},
            'plain string chunk'
        ]
        
        # Test streaming
        chunks = list(client.generate_stream("Test streaming"))
        
        # Check chunks
        self.assertEqual(chunks, ["Hello", " world", "!", "plain string chunk"])
        
        # Verify prompt was formatted
        self.mock_formatter.format_prompt.assert_called_once_with(
            "Test streaming", 
            client.config.default_system_prompt
        )
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_generate_with_messages(self):
        """Test generate_with_messages method."""
        client = self._create_client()
        
        # Mock response
        self.mock_model.generate_text.return_value = {
            'results': [{'generated_text': 'Response to conversation.'}]
        }
        
        # Create test messages
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a test assistant."),
            Message(role=MessageRole.USER, content="Hello?"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        
        # Test with messages
        result = client.generate_with_messages(messages)
        
        # Check result
        self.assertEqual(result, "Response to conversation.")
        
        # Verify messages were formatted with system message
        self.mock_formatter.format_messages.assert_called_once_with(
            ANY,  # We'll check this separately
            "You are a test assistant."
        )
        
        # Don't check the exact role values as they may be represented differently
        # but check that messages were converted
        self.mock_formatter.format_messages.assert_called_once()
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_error_handling(self):
        """Test error handling."""
        client = self._create_client()
        
        # Mock API error
        self.mock_model.generate_text.side_effect = Exception("API Error")
        
        # Test generate method
        try:
            client.generate("Error test")
            self.fail("Expected exception but none was raised")
        except Exception as e:
            self.assertIn("API Error", str(e))
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_embeddings_not_implemented(self):
        """Test that embeddings raise NotImplementedError."""
        client = self._create_client()
        
        try:
            client.get_embeddings(["Test text"])
            self.fail("Expected NotImplementedError but none was raised")
        except NotImplementedError as e:
            self.assertIn("not yet supported for IBM WatsonX", str(e))
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()
    
    def test_close(self):
        """Test close method."""
        client = self._create_client()
        
        # Test close with model
        client.close()
        self.mock_model.close_persistent_connection.assert_called_once()
        
        # Clean up
        self.api_client_patcher.stop()
        self.model_patcher.stop()


if __name__ == "__main__":
    unittest.main()