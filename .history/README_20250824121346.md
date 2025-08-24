# AI-Powered Chatbot

An intelligent chatbot application with Natural Language Processing capabilities, featuring a modern animated UI and comprehensive conversation logging.

## Features

### 🤖 AI Capabilities
- **Transformer Models**: Uses state-of-the-art transformer models for sentiment analysis and question answering
- **TF-IDF Similarity**: Intelligent FAQ matching using TF-IDF vectorization
- **NLTK Integration**: Advanced text processing with NLTK
- **Contextual Responses**: Generates appropriate responses based on context and sentiment

### 🎨 Modern UI
- **Animated Interface**: Smooth animations and transitions
- **Floating Pill Navigation**: Modern pill-shaped navigation bar
- **Glassmorphism Design**: Beautiful glass-effect styling with backdrop blur
- **Responsive Design**: Works perfectly on desktop and mobile devices
- **Real-time Analytics**: Live conversation analytics panel

### 📊 Analytics & Logging
- **Conversation Logging**: Complete conversation history stored in SQLite
- **Sentiment Analysis**: Real-time sentiment analysis of user messages
- **Confidence Scoring**: Confidence metrics for bot responses
- **Usage Analytics**: Track conversation patterns and user engagement

## Technologies Used

- **Backend**: Python, Flask, SQLite
- **AI/ML**: Transformers, NLTK, Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Database**: SQLite for conversation logging
- **Deployment**: Gunicorn ready for production

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy .env file and modify as needed
   cp .env.example .env
   ```

5. **Initialize the application**
   ```bash
   python app.py
   ```

## Usage

### Starting the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### API Endpoints

#### POST /api/chat
Send a message to the chatbot
```json
{
  "message": "Hello, how are you?",
  "session_id": "optional_session_id"
}
```

Response:
```json
{
  "response": "Hello! I'm doing great, thank you for asking...",
  "sentiment": "POSITIVE",
  "confidence": 0.85,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### GET /api/history/{session_id}
Get conversation history for a session

#### GET /api/analytics
Get conversation analytics and statistics

## Features in Detail

### AI Engine
The chatbot uses multiple AI techniques:

1. **Sentiment Analysis**: Analyzes the emotional tone of user messages
2. **FAQ Matching**: Uses TF-IDF vectorization to find the best matching FAQ
3. **Contextual Responses**: Generates appropriate responses based on context
4. **Confidence Scoring**: Provides confidence metrics for response quality

### Modern UI Features

1. **Floating Particles**: Animated background particles
2. **Glassmorphism**: Modern glass-effect design
3. **Smooth Animations**: All interactions are smoothly animated
4. **Typing Indicators**: Real-time typing indicators
5. **Analytics Panel**: Slide-out analytics panel
6. **Responsive Design**: Mobile-first responsive design

### Database Schema

#### Conversations Table
- `id`: Primary key
- `session_id`: Session identifier
- `user_message`: User's message
- `bot_response`: Bot's response
- `sentiment`: Sentiment score
- `confidence`: Confidence score
- `timestamp`: Message timestamp

#### FAQ Table
- `id`: Primary key
- `question`: FAQ question
- `answer`: FAQ answer
- `category`: Question category
- `keywords`: Associated keywords

## Customization

### Adding New FAQs
You can add new FAQ entries by inserting them into the database:

```python
# Add to the sample_faqs list in app.py
{
    'question': 'Your question here?',
    'answer': 'Your detailed answer here.',
    'category': 'category_name',
    'keywords': 'relevant, keywords, here'
}
```

### Customizing the UI
The UI can be customized by modifying the CSS in `templates/index.html`:

- Colors: Modify the gradient colors in the CSS variables
- Animations: Adjust animation durations and effects
- Layout: Modify the grid and flexbox layouts

### Adding New AI Models
You can integrate additional transformer models:

```python
# In the setup_transformers method
self.custom_model = pipeline(
    "your-task",
    model="your-model-name"
)
```

## Deployment

### Production Setup

1. **Environment Variables**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

2. **Using Gunicorn**
   ```bash
   gunicorn --bind 0.0.0.0:5000 app:app
   ```

3. **Docker Deployment**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
   ```

### Cloud Deployment Options

- **Heroku**: Ready for Heroku deployment
- **AWS**: Can be deployed on EC2 or Elastic Beanstalk
- **Google Cloud**: Compatible with App Engine
- **Azure**: Can be deployed on Azure App Service

## Performance Optimization

1. **Model Caching**: Transformer models are cached after first load
2. **Database Indexing**: Proper indexing on frequently queried columns
3. **Response Caching**: FAQ responses can be cached for better performance
4. **Async Support**: Can be enhanced with async/await for better concurrency

## Security Features

1. **Input Validation**: All user inputs are validated and sanitized
2. **SQL Injection Protection**: Uses parameterized queries
3. **Rate Limiting**: Can be enhanced with Flask-Limiter
4. **CORS Configuration**: Properly configured CORS headers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue in the repository.

## Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output
- [ ] File upload support
- [ ] Integration with external APIs
- [ ] Advanced analytics dashboard
- [ ] User authentication
- [ ] Custom training data upload
- [ ] Webhook integrations
