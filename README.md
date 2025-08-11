# LangGraph Agent Setup Guide

A comprehensive guide to setting up and running agents using LangGraph with various LLM providers.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Basic Agent Structure](#basic-agent-structure)
- [Quick Start](#quick-start)
- [Advanced Configuration](#advanced-configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- pip package manager
- API keys for your chosen LLM provider
- Basic understanding of Python and graph concepts

## Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv langgraph-env

# Activate virtual environment
# On Windows:
langgraph-env\Scripts\activate
# On macOS/Linux:
source langgraph-env/bin/activate
```

### 2. Install Core Dependencies

```bash
# Core LangGraph packages
pip install langgraph langchain-core langchain-community

# Choose your LLM provider (install one or more):
pip install langchain-openai      # For OpenAI GPT models
pip install langchain-anthropic   # For Claude models
pip install langchain-google-genai # For Google Gemini
pip install langchain-ollama      # For local Ollama models

# Additional useful packages
pip install python-dotenv         # For environment variables
pip install pydantic             # For data validation
pip install streamlit            # For web interface (optional)
```

### 3. Install All Dependencies at Once

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
langgraph>=0.1.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
python-dotenv>=1.0.0
pydantic>=2.0.0
streamlit>=1.28.0
```

## Environment Setup

### 1. Create Environment File

Create a `.env` file in your project root:

```bash
# .env file
# Choose your LLM provider and add the corresponding API key

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Other configurations
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
```

### 2. Load Environment Variables

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Access your API keys
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
```

## Basic Agent Structure

Here's the fundamental structure of a LangGraph agent:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# 1. Define State
class
