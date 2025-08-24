#!/bin/bash

# AI Chatbot Deployment Script

echo "🤖 Starting AI Chatbot Deployment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📊 Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
print('NLTK data downloaded successfully!')
"

# Initialize database
echo "🗄️ Initializing database..."
python -c "
from app import ChatbotEngine
chatbot = ChatbotEngine()
print('Database initialized successfully!')
"

echo "✅ Deployment completed successfully!"
echo ""
echo "🚀 To start the application:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Run the application: python app.py"
echo "   3. Open your browser to: http://localhost:5000"
echo ""
echo "🎉 Your AI Chatbot is ready to use!"
