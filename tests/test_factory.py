"""
Tests for the factory module.
"""
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Optional, Type, ClassVar

import pytest

from llm_clients.factory import (
    register_client,
    get_client,
    list_available_models,
    get_available_models,
    update_default_model,
    CLIENT_REGISTRY,
    DEFAULT_MODELS,
    _resolve_provider_type,
    _get_client_class,
    _import_client_module
)
from llm_clients.interfaces import ProviderType, ModelInfo, ClientConfig
from llm_clients.llm_client import LLMClient


# Mock client classes for testing
class MockClientConfig(ClientConfig):
    """Mock config for test client."""
    test_option: Optional[str] = None


class MockClient(LLMClient[MockClientConfig]):
    """Mock client for testing."""
    
    provider_type: ClassVar[ProviderType] = ProviderType.OPENAI
    
    def __init__(self, model_id: str, config: Optional[MockClientConfig] = None):
        """Initialize mock client."""
        self.config = config or MockClientConfig()
        super().__init__(model_id)
    
    def _validate_environment(self) -> None:
        """Mock environment validation."""
        pass
    
    def _init_client(self) -> None:
        """Mock client initialization."""
        pass
    
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """Get mock models."""
        return {
            "mock-model": ModelInfo(
                name="Mock Model",
                model_id="mock-model",
                description="Mock model for testing",
                provider=ProviderType.OPENAI
            )
        }
    
    def _generate_impl(self, prompt: str, params: any) -> str:
        """Mock generation implementation."""
        return f"Mock response to: {prompt}"
    
    def _generate_stream_impl(self, prompt: str, params: any):
        """Mock streaming implementation."""
        yield "Mock stream"


class TestFactory(unittest.TestCase):
    """Test cases for the factory module."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear registry before each test
        CLIENT_REGISTRY.clear()
        
        # Register mock client
        register_client(MockClient)
    
    def test_register_client(self):
        """Test registering a client."""
        # Verify the client was registered
        self.assertIn(ProviderType.OPENAI, CLIENT_REGISTRY)
        self.assertEqual(CLIENT_REGISTRY[ProviderType.OPENAI], MockClient)
        
        # Test registering invalid client (missing provider_type)
        class InvalidClient:
            pass
        
        with self.assertRaises(ValueError):
            register_client(InvalidClient)
    
    def test_get_client(self):
        """Test getting a client."""
        # Test with string provider
        client = get_client("openai", "test-model")
        self.assertIsInstance(client, MockClient)
        self.assertEqual(client.model_id, "test-model")
        
        # Test with enum provider
        client = get_client(ProviderType.OPENAI, "test-model")
        self.assertIsInstance(client, MockClient)
        
        # Test with default model
        DEFAULT_MODELS[ProviderType.OPENAI] = "default-model"
        client = get_client("openai")
        self.assertEqual(client.model_id, "default-model")
        
        # Test with custom config
        config = MockClientConfig(test_option="test-value")
        client = get_client("openai", "test-model", config)
        self.assertEqual(client.config.test_option, "test-value")
        
        # Test with unknown provider
        with self.assertRaises(ValueError):
            get_client("unknown-provider")
    
    def test_list_available_models(self):
        """Test listing available models."""
        # Mock default models for testing
        DEFAULT_MODELS[ProviderType.OPENAI] = "mock-model"
        
        # Get the model list
        model_list = list_available_models()
        
        # Verify it contains expected content
        self.assertIsInstance(model_list, str)
        self.assertIn("Available Models:", model_list)
        self.assertIn("OPENAI", model_list)
        self.assertIn("Mock Model", model_list)
        self.assertIn("(default)", model_list)
    
    def test_get_available_models(self):
        """Test getting available models for a provider."""
        # Test with string provider
        models = get_available_models("openai")
        self.assertIsInstance(models, dict)
        self.assertIn("mock-model", models)
        self.assertEqual(models["mock-model"].name, "Mock Model")
        
        # Test with enum provider
        models = get_available_models(ProviderType.OPENAI)
        self.assertIn("mock-model", models)
        
        # Test with unknown provider
        with self.assertRaises(ValueError):
            get_available_models("unknown-provider")
    
    def test_update_default_model(self):
        """Test updating the default model."""
        # Update default model
        update_default_model("openai", "new-default-model")
        self.assertEqual(DEFAULT_MODELS[ProviderType.OPENAI], "new-default-model")
        
        # Test with enum provider
        update_default_model(ProviderType.OPENAI, "newer-default-model")
        self.assertEqual(DEFAULT_MODELS[ProviderType.OPENAI], "newer-default-model")
        
        # Test with unknown provider
        with self.assertRaises(ValueError):
            update_default_model("unknown-provider", "model")
    
    def test_resolve_provider_type(self):
        """Test resolving provider type from string."""
        # Test valid strings
        self.assertEqual(_resolve_provider_type("openai"), ProviderType.OPENAI)
        self.assertEqual(_resolve_provider_type("anthropic"), ProviderType.ANTHROPIC)
        self.assertEqual(_resolve_provider_type("OPENAI".lower()), ProviderType.OPENAI)
        
        # Test enum values
        self.assertEqual(_resolve_provider_type(ProviderType.OPENAI), ProviderType.OPENAI)
        
        # Test invalid string
        with self.assertRaises(ValueError):
            _resolve_provider_type("invalid-provider")
    
    def test_get_client_class(self):
        """Test getting client class."""
        # Test registered provider
        client_class = _get_client_class(ProviderType.OPENAI)
        self.assertEqual(client_class, MockClient)
        
        # Test unregistered provider
        CLIENT_REGISTRY.clear()  # Clear registry
        
        # Mock import function to avoid actual imports
        with patch("llm_clients.factory._import_client_module") as mock_import:
            # Set up mock to register a client
            def side_effect(provider_type):
                CLIENT_REGISTRY[provider_type] = MockClient
            
            mock_import.side_effect = side_effect
            
            # Get client class for unregistered provider
            client_class = _get_client_class(ProviderType.OPENAI)
            
            # Verify import was attempted
            mock_import.assert_called_once_with(ProviderType.OPENAI)
            
            # Verify correct class was returned
            self.assertEqual(client_class, MockClient)
        
        # Test import failure
        CLIENT_REGISTRY.clear()  # Clear registry
        
        with patch("llm_clients.factory._import_client_module", side_effect=ImportError("Test error")):
            with self.assertRaises(ValueError):
                _get_client_class(ProviderType.OPENAI)
    
    def test_import_client_module(self):
        """Test importing client module."""
        # Mock importlib.import_module to avoid actual imports
        with patch("importlib.import_module") as mock_import:
            # Test successful import
            _import_client_module(ProviderType.OPENAI)
            mock_import.assert_called_once_with(".openai_client", package="llm_clients")
            
            # Test failed import
            mock_import.reset_mock()
            mock_import.side_effect = ImportError("Test error")
            
            with self.assertRaises(ImportError):
                _import_client_module(ProviderType.OPENAI)
            
            # Test unknown provider
            with self.assertRaises(ImportError) as exc_info:
                # Use a valid ProviderType but remove it from the module_map
                with patch("llm_clients.factory.module_map", {}):
                    _import_client_module(ProviderType.OPENAI)
                
                self.assertIn("No module mapping", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()