#!/usr/bin/env python3
"""
Example usage of the LLM clients library.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from the llm_clients library
from llm_clients import (
    get_client, 
    list_available_models,
    get_available_models,
    ProviderType,
    GenerationParams
)

def test_client(provider: str, model_id: str = None):
    """Test a specific LLM provider with basic prompts."""
    print(f"\n=== Testing {provider.upper()} Client ===")
    
    try:
        # Create client
        client = get_client(provider, model_id)
        
        # Print info about the model
        print(f"Using model: {client.model_id}")
        
        # Basic generation
        prompt = "Explain quantum computing in simple terms"
        print(f"\nPrompt: {prompt}")
        
        # Set generation parameters
        params = GenerationParams(
            max_tokens=200,
            temperature=0.7
        )
        
        # Generate response
        response = client.generate(prompt, params)
        
        print(f"\nResponse:\n{response}")
        
        # Close connection if needed
        client.close()
        
        return True
    except Exception as e:
        print(f"Error testing {provider}: {str(e)}")
        return False

def test_streaming(provider: str, model_id: str = None):
    """Test streaming generation with a specific provider."""
    print(f"\n=== Testing {provider.upper()} Streaming ===")
    
    try:
        # Create client
        client = get_client(provider, model_id)
        
        # Print info about the model
        print(f"Using model: {client.model_id}")
        
        # Streaming generation
        prompt = "Tell me a short story about artificial intelligence"
        print(f"\nPrompt: {prompt}")
        
        # Set generation parameters
        params = GenerationParams(
            max_tokens=300,
            temperature=0.8
        )
        
        # Stream response
        print("\nStreaming response:")
        
        for chunk in client.generate_stream(prompt, params):
            print(chunk, end="", flush=True)
        
        print("\n")
        
        # Close connection if needed
        client.close()
        
        return True
    except Exception as e:
        print(f"Error testing {provider} streaming: {str(e)}")
        return False

def list_models():
    """List available models for all providers."""
    print("\n=== Available Models ===")
    print(list_available_models())

def list_provider_models(provider: str):
    """List available models for a specific provider."""
    print(f"\n=== Available Models for {provider.upper()} ===")
    try:
        models = get_available_models(provider)
        print(f"Found {len(models)} models:")
        
        for key, model in models.items():
            print(f"  - {key}: {model.name} ({model.description})")
            
    except Exception as e:
        print(f"Error listing models for {provider}: {str(e)}")

def main():
    """Run example client tests."""
    # List all available models
    list_models()
    
    # Test each provider
    print("\n=== Testing Providers ===")
    
    # Test Anthropic Claude
    test_client("anthropic", "claude-3-sonnet")
    
    # Test OpenAI GPT
    test_client("openai", "gpt-3.5-turbo")
    
    # Test IBM WatsonX
    test_client("watsonx", "ibm/granite-13b-instruct-v2")
    
    # Test Ollama
    test_client("ollama", "llama3")
    
    # Test streaming with one provider
    test_streaming("anthropic", "claude-3-haiku")

if __name__ == "__main__":
    main()