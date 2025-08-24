@echo off
echo 🤖 Starting AI Chatbot Deployment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip.
    pause
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Download NLTK data
echo 📊 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('wordnet'); print('NLTK data downloaded successfully!')"

REM Initialize database
echo 🗄️ Initializing database...
python -c "from app import ChatbotEngine; chatbot = ChatbotEngine(); print('Database initialized successfully!')"

echo ✅ Deployment completed successfully!
echo.
echo 🚀 To start the application:
echo    1. Activate virtual environment: venv\Scripts\activate
echo    2. Run the application: python app.py
echo    3. Open your browser to: http://localhost:5000
echo.
echo 🎉 Your AI Chatbot is ready to use!
pause
