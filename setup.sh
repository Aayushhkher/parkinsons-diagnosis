#!/bin/bash

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p backend/static/explanations
touch backend/static/explanations/.gitkeep

# Run tests
pytest

echo "Setup completed successfully!" 