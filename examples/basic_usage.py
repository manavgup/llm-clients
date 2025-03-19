#!/usr/bin/env python3
"""
Updated example usage of the LLM clients library showcasing newer features.
"""
import os
import logging
from typing import List, Optional
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
    update_default_model,
    ProviderType,
    GenerationParams,
    Message,
    MessageRole,
    AnthropicConfig,
    OpenAIConfig,
    WatsonxConfig,
    OllamaConfig
)


def test_basic_client(provider: str, model_id: str = None):
    """Test a specific LLM provider with basic prompts."""
    print(f"\n=== Testing {provider.upper()} Client (Basic Generation) ===")
    try:
        # Create client
        client = get_client(provider, model_id)
        
        # Print info about the model
        print(f"Using model: {client.model_id}")
        
        # Basic generation
        prompt = "Explain quantum computing in simple terms, in less than 100 words."
        print(f"\nPrompt: {prompt}")
        
        # Set generation parameters
        params = GenerationParams(
            max_tokens=150,
            temperature=0.7,
            top_p=0.95,
            # Some providers support top_k
            top_k=40 if provider in ["ollama", "watsonx"] else None,
            # You can set stop sequences
            stop_sequences=["\n\n", "END"],
            # For reproducibility
            seed=42
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
        prompt = "Tell me a short story about artificial intelligence in the style of Isaac Asimov."
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


def test_chat_conversation(provider: str, model_id: str = None):
    """Test chat-style conversation with a specific provider."""
    print(f"\n=== Testing {provider.upper()} Chat Conversation ===")
    try:
        # Create client - optionally with provider-specific configuration
        config = None
        if provider == "anthropic":
            config = AnthropicConfig(
                default_system_prompt="You are an AI assistant that specializes in providing concise, accurate responses."
            )
        elif provider == "openai":
            config = OpenAIConfig(
                default_system_prompt="You are a helpful AI assistant with expertise in science and technology."
            )
        
        client = get_client(provider, model_id, config)
        
        # Print info about the model
        print(f"Using model: {client.model_id}")
        
        # Create a conversation
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant that answers questions concisely."),
            Message(role=MessageRole.USER, content="What are the primary types of machine learning?"),
            Message(role=MessageRole.ASSISTANT, content="The primary types of machine learning are supervised learning, unsupervised learning, and reinforcement learning."),
            Message(role=MessageRole.USER, content="Can you briefly explain the difference between supervised and unsupervised learning?")
        ]
        
        # Set generation parameters
        params = GenerationParams(
            max_tokens=150,
            temperature=0.7
        )
        
        # Generate response based on the conversation
        print("\nGenerating response to conversation...")
        response = client.generate_with_messages(messages, params)
        print(f"\nResponse:\n{response}")
        
        # Close connection if needed
        client.close()
        return True
    except Exception as e:
        print(f"Error testing {provider} chat: {str(e)}")
        return False


def test_custom_config(provider: str, model_id: str = None):
    """Test a provider with custom configuration."""
    print(f"\n=== Testing {provider.upper()} with Custom Config ===")
    try:
        # Create provider-specific config
        config = None
        
        if provider == "anthropic":
            config = AnthropicConfig(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                default_system_prompt="You are Claude, a helpful AI assistant created by Anthropic that prioritizes accuracy and clarity.",
                timeout=60.0
            )
        elif provider == "openai":
            config = OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_ORG_ID"),
                default_system_prompt="You are a GPT model developed by OpenAI that specializes in providing accurate information."
            )
        elif provider == "watsonx":
            config = WatsonxConfig(
                api_key=os.getenv("WATSONX_API_KEY"),
                project_id=os.getenv("WATSONX_PROJECT_ID"),
                default_system_prompt="You are an AI assistant created by IBM that excels at reasoning and problem-solving."
            )
        elif provider == "ollama":
            config = OllamaConfig(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                request_timeout=30.0,
                num_gpu=1,  # Use 1 GPU
                num_thread=4  # Use 4 CPU threads
            )
            
        # Create client with custom config
        client = get_client(provider, model_id, config)
        
        # Print info about the model
        print(f"Using model: {client.model_id} with custom configuration")
        
        # Generate a response with custom config
        prompt = "What are the most important recent advances in natural language processing?"
        print(f"\nPrompt: {prompt}")
        
        response = client.generate(prompt, GenerationParams(max_tokens=250))
        print(f"\nResponse:\n{response}")
        
        # Close connection if needed
        client.close()
        return True
    except Exception as e:
        print(f"Error testing {provider} with custom config: {str(e)}")
        return False


def test_embeddings(provider: str, model_id: str = None):
    """Test generating embeddings with supported providers."""
    if provider not in ["openai", "ollama"]:
        print(f"\n=== Skipping embeddings test for {provider} (not supported) ===")
        return False
        
    print(f"\n=== Testing {provider.upper()} Embeddings ===")
    try:
        # For Ollama, use a specific embedding model if none specified
        if provider == "ollama" and model_id is None:
            # Based on successful curl test, use the "granite-embedding" model
            embed_model = "granite-embedding"
            print(f"Using specific embedding model for Ollama: {embed_model}")
        else:
            embed_model = model_id
            
        # Create client
        client = get_client(provider, embed_model)
        
        # Print info about the model
        print(f"Using model: {client.model_id}")
        
        # Generate embeddings for multiple texts
        texts = [
            "Artificial intelligence is revolutionizing many fields.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing focuses on text and speech understanding."
        ]
        
        print(f"\nGenerating embeddings for {len(texts)} texts...")
        embeddings = client.get_embeddings(texts)
        
        # Display results
        print(f"\nGenerated {len(embeddings)} embeddings:")
        for i, embedding in enumerate(embeddings):
            print(f"Embedding {i+1}: Vector with {len(embedding)} dimensions")
            # Show a small sample of the vector
            print(f"  Sample: {embedding[:5]}...")
        
        # Close connection if needed
        client.close()
        return True
    except NotImplementedError:
        print(f"Embeddings not implemented for {provider}")
        return False
    except Exception as e:
        print(f"Error testing {provider} embeddings: {str(e)}")
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
            context_info = f", Context: {model.context_length} tokens" if model.context_length else ""
            capabilities = f", Capabilities: {', '.join(model.capabilities)}" if model.capabilities else ""
            print(f" - {key}: {model.name} ({model.description}{context_info}{capabilities})")
    except Exception as e:
        print(f"Error listing models for {provider}: {str(e)}")


def set_default_models():
    """Update default models for providers."""
    # You can change the default model for each provider
    update_default_model(ProviderType.ANTHROPIC, "claude-3-haiku-20240307")
    update_default_model(ProviderType.OPENAI, "gpt-3.5-turbo")
    # Make sure these models are available locally
    update_default_model(ProviderType.OLLAMA, "llama3:8b")
    
    print("\n=== Updated Default Models ===")
    for provider in ProviderType:
        # Create a client with no model_id to use the default
        try:
            client = get_client(provider)
            print(f"{provider.value.capitalize()}: {client.model_id}")
            client.close()
        except Exception as e:
            print(f"{provider.value.capitalize()}: Error - {str(e)}")


def main():
    """Run comprehensive examples of the LLM clients library."""
    # Set custom default models (optional)
    set_default_models()
    
    # List models
    list_models()
    
    # List models for specific providers
    for provider in [ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.WATSONX, ProviderType.OLLAMA]:
        list_provider_models(provider)
    
    # Test basic generation
    print("\n=== Testing Basic Generation ===")
    test_basic_client("anthropic", "claude-3-haiku-20240307")
    test_basic_client("openai", "gpt-3.5-turbo")
    # These might require local setup
    if os.getenv("WATSONX_API_KEY"):
        test_basic_client("watsonx", "ibm/granite-13b-instruct-v2")
    if os.getenv("OLLAMA_BASE_URL") or os.path.exists("/usr/local/bin/ollama"):
        test_basic_client("ollama", "llama3")
    
    # Test streaming
    print("\n=== Testing Streaming Generation ===")
    test_streaming("anthropic", "claude-3-haiku-20240307")
    test_streaming("openai", "gpt-3.5-turbo")
    
    # Test chat conversation
    print("\n=== Testing Chat Conversations ===")
    test_chat_conversation("anthropic", "claude-3-haiku-20240307")
    test_chat_conversation("openai", "gpt-3.5-turbo")
    
    # Test custom configurations
    print("\n=== Testing Custom Configurations ===")
    test_custom_config("anthropic", "claude-3-haiku-20240307")
    test_custom_config("openai", "gpt-3.5-turbo")
    
    # Test embeddings (only supported by some providers)
    print("\n=== Testing Embeddings ===")
    test_embeddings("openai", "gpt-3.5-turbo")
    test_embeddings("ollama", "granite-embedding")


if __name__ == "__main__":
    main()