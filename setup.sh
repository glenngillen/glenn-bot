#!/bin/bash

echo "Setting up Glenn Bot..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create .env file from example
cp .env.example .env

# Create necessary directories
mkdir -p data/chroma data/conversations knowledge/frameworks

echo "Setup complete!"
echo ""
echo "To run Glenn Bot:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the bot: python run.py"
echo ""
echo "Make sure Ollama is running with required models:"
echo "- ollama pull llama3:8b"
echo "- ollama pull nomic-embed-text"