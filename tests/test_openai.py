import unittest
from unittest.mock import patch, MagicMock
from llm_clients.openai import OpenAIClient, COMPLETION_MODELS


class TestOpenAIClient(unittest.TestCase):
    @patch('openai.OpenAI')
    def setUp(self, mock_openai_class):
        """Set up test case with mocked OpenAI client."""
        self.mock_openai = MagicMock()
        mock_openai_class.return_value = self.mock_openai
        
        # Mock chat completions and completions
        self.mock_chat_completion = MagicMock()
        self.mock_chat_completion.choices = [MagicMock()]
        self.mock_chat_completion.choices[0].message.content = "Chat response"
        self.mock_openai.chat.completions.create.return_value = self.mock_chat_completion
        
        self.mock_completion = MagicMock()
        self.mock_completion.choices = [MagicMock()]
        self.mock_completion.choices[0].text = "Completion response"
        self.mock_openai.completions.create.return_value = self.mock_completion
        
        # Create client with environment variable patch
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            self.client = OpenAIClient(model_id="gpt-4")

    def test_init(self):
        """Test initialization of OpenAIClient."""
        self.assertEqual(self.client.model_id, "gpt-4")
        self.assertTrue(self.client.is_chat_model)
        
    def test_is_chat_model(self):
        """Test the _is_chat_model method."""
        # Chat models
        self.assertTrue(self.client._is_chat_model("gpt-4"))
        self.assertTrue(self.client._is_chat_model("gpt-3.5-turbo"))
        
        # Completion models
        self.assertFalse(self.client._is_chat_model("gpt-3.5-turbo-instruct"))
        for model in COMPLETION_MODELS:
            self.assertFalse(self.client._is_chat_model(model))
            
    def test_generate_chat(self):
        """Test generate with a chat model."""
        result = self.client.generate("Test prompt")
        
        self.assertEqual(result, "Chat response")
        
        # Verify key aspects of the API call without being too strict about exact parameters
        args, kwargs = self.mock_openai.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], self.client.model_id)
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Test prompt"}])
        
    @patch('openai.OpenAI')
    def test_generate_completion(self, mock_openai_class):
        """Test generate with a completion model."""
        mock_openai = MagicMock()
        mock_openai_class.return_value = mock_openai
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].text = "Completion response"
        mock_openai.completions.create.return_value = mock_completion
        
        # Create client with a completion model
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = OpenAIClient(model_id="gpt-3.5-turbo-instruct")
        
        result = client.generate("Test prompt")
        
        self.assertEqual(result, "Completion response")
        
        # Verify key aspects of the API call without being too strict about exact parameters
        args, kwargs = mock_openai.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-3.5-turbo-instruct")
        self.assertEqual(kwargs["prompt"], "Test prompt")
        
    def test_generate_with_params(self):
        """Test generate with custom parameters."""
        from llm_clients.interfaces import GenerationParams
        params = GenerationParams(
            max_tokens=2000,
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["STOP"],
            seed=42
        )
        
        result = self.client.generate("Test prompt", params)
        
        self.assertEqual(result, "Chat response")
        self.mock_openai.chat.completions.create.assert_called_once_with(
            model=self.client.model_id,
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=2000,
            temperature=0.5,
            top_p=0.9,
            stop=["STOP"],
            seed=42
        )
        
    def test_handle_error(self):
        """Test error handling in generate."""
        self.mock_openai.chat.completions.create.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception) as context:
            self.client.generate("Test prompt")
            
        self.assertTrue("API Error" in str(context.exception))
        
    def test_get_available_models(self):
        """Test get_available_models method."""
        # Call the method
        models = OpenAIClient.get_available_models()
        
        # Verify we get a dictionary of models
        self.assertIsInstance(models, dict)
        self.assertGreater(len(models), 0)
        
        # Check for key models
        self.assertIn("gpt-4-turbo", models)
        self.assertIn("gpt-3.5-turbo", models)
        
        # Check model info structure
        gpt4_info = models["gpt-4-turbo"]
        self.assertEqual(gpt4_info.model_id, "gpt-4-turbo")
        self.assertEqual(gpt4_info.provider, "openai")
        self.assertEqual(gpt4_info.context_length, 128000)
        
    def test_generate_stream(self):
        """Test streaming generation."""
        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta.content = "!"
        
        self.mock_openai.chat.completions.create.return_value = [
            mock_chunk1, mock_chunk2, mock_chunk3
        ]
        
        # Call generate_stream and collect results
        chunks = []
        for chunk in self.client.generate_stream("Test prompt"):
            chunks.append(chunk)
            
        # Verify results
        self.assertEqual(chunks, ["Hello", " world", "!"])
        
        # Verify key aspects of the API call, especially stream=True
        args, kwargs = self.mock_openai.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], self.client.model_id)
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Test prompt"}])
        self.assertTrue(kwargs["stream"])


if __name__ == '__main__':
    unittest.main()