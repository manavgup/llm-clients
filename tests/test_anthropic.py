import unittest
from unittest.mock import patch, MagicMock
from llm_clients.anthropic import AnthropicClient


class TestAnthropicClient(unittest.TestCase):
    @patch('anthropic.Anthropic')
    def setUp(self, mock_anthropic_class):
        self.mock_anthropic = MagicMock()
        mock_anthropic_class.return_value = self.mock_anthropic
        self.client = AnthropicClient(model_id="claude-3-opus-20240229")

    def test_init(self):
        """Test the initialization of the AnthropicClient."""
        self.assertEqual(self.client.model_id, "claude-3-opus-20240229")

    @patch('anthropic.Anthropic')
    def test_init_with_params(self, mock_anthropic_class):
        """Test initialization with custom parameters."""
        mock_anthropic = MagicMock()
        mock_anthropic_class.return_value = mock_anthropic
        
        # Based on your implementation, AnthropicClient only takes model_id parameter
        client = AnthropicClient(model_id="claude-3-sonnet-20240229")
        
        self.assertEqual(client.model_id, "claude-3-sonnet-20240229")
        # Your implementation gets API key from environment, not constructor parameter
        mock_anthropic_class.assert_called_once()

    def test_generate(self):
        """Test the generate method of AnthropicClient."""
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "This is a test response from Claude."
        self.mock_anthropic.messages.create.return_value = mock_response
        
        # Call the generate method
        result = self.client.generate("Test prompt")
        
        # Assert the result and method calls
        self.assertEqual(result, "This is a test response from Claude.")
        
        # Only check that it was called once with the correct model and message
        # Don't check the exact parameters since defaults might change
        args, kwargs = self.mock_anthropic.messages.create.call_args
        self.assertEqual(kwargs["model"], self.client.model_id)
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Test prompt"}])

    def test_generate_with_params(self):
        """Test generating with custom parameters."""
        # Setup the mock
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response with custom parameters"
        self.mock_anthropic.messages.create.return_value = mock_response
        
        # Create params object based on the actual implementation
        from llm_clients.interfaces import GenerationParams
        params = GenerationParams(
            max_tokens=2000,
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["STOP"]
        )
        
        # Call generate with params
        result = self.client.generate("Test prompt", params)
        
        # Assert
        self.assertEqual(result, "Response with custom parameters")
        self.mock_anthropic.messages.create.assert_called_once_with(
            model=self.client.model_id,
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=2000,
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["STOP"]
        )

    def test_handle_error(self):
        """Test error handling in the generate method."""
        # Setup the mock to raise an exception
        self.mock_anthropic.messages.create.side_effect = Exception("API Error")
        
        # Call generate and expect an exception
        with self.assertRaises(Exception) as context:
            self.client.generate("Test prompt")
        
        self.assertTrue("API Error" in str(context.exception))

    def test_get_available_models(self):
        """Test getting available models."""
        # Call the method
        models = AnthropicClient.get_available_models()
        
        # Verify we get a dictionary of models
        self.assertIsInstance(models, dict)
        self.assertGreater(len(models), 0)
        
        # Check for a key model
        self.assertIn("claude-3-opus", models)
        
        # Check model info structure
        model_info = models["claude-3-opus"]
        self.assertEqual(model_info.model_id, "claude-3-opus-20240229")
        self.assertEqual(model_info.provider, "anthropic")


if __name__ == '__main__':
    unittest.main()