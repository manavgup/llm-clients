from setuptools import setup, find_packages

setup(
    name="llm_clients",
    version="0.1.0",
    description="Standardized clients for various LLM providers",
    author="Manav Gupta",
    author_email="manavg@gmail.com",
    packages=find_packages(),
    install_requires=[
        "anthropic",  # Add the dependencies your clients need
        "openai",
        "ibm_watsonx_ai"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)