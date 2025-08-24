import os
import sqlite3
import nltk
import json
import logging
import re
import random
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ChatbotEngine:
    def __init__(self):
        self.db_path = 'chatbot.db'
        self.initialize_database()
        self.setup_nltk()
        self.load_knowledge_base()
        self.load_faq_data()
        self.setup_transformers()
        self.conversation_context = {}  # Store conversation context per session
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
    
    def load_knowledge_base(self):
        """Load comprehensive knowledge base from JSON file"""
        try:
            with open('knowledge_base.json', 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            
            # Create a comprehensive dataset for similarity matching
            self.knowledge_texts = []
            self.knowledge_answers = []
            self.knowledge_categories = []
            
            for item in self.knowledge_base:
                for question in item['questions']:
                    self.knowledge_texts.append(question + ' ' + item['keywords'])
                    self.knowledge_answers.append(item['answer'])
                    self.knowledge_categories.append(item['category'])
            
            # Create TF-IDF vectors for knowledge base
            if self.knowledge_texts:
                self.knowledge_vectorizer = TfidfVectorizer(
                    stop_words='english', 
                    max_features=5000,
                    ngram_range=(1, 3),
                    lowercase=True
                )
                self.knowledge_vectors = self.knowledge_vectorizer.fit_transform(self.knowledge_texts)
            
            logger.info(f"Loaded knowledge base with {len(self.knowledge_texts)} entries")
            
        except FileNotFoundError:
            logger.warning("Knowledge base file not found, using basic responses")
            self.knowledge_base = []
            self.knowledge_texts = []
            self.knowledge_answers = []
            self.knowledge_categories = []
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.knowledge_base = []
    
    def setup_transformers(self):
        """Initialize transformer models"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Sentiment analysis pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                
                logger.info("Transformer models loaded successfully")
            else:
                logger.warning("Transformers not available, using fallback methods")
                self.sentiment_analyzer = None
        except Exception as e:
            logger.error(f"Error loading transformer models: {e}")
            self.sentiment_analyzer = None
    
    def initialize_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                sentiment REAL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create FAQ table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                category TEXT,
                keywords TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def load_faq_data(self):
        """Load FAQ data and create TF-IDF vectors"""
        conn = sqlite3.connect(self.db_path)
        
        # Sample FAQ data
        sample_faqs = [
            {
                'question': 'What are your business hours?',
                'answer': 'We are open Monday to Friday from 9 AM to 6 PM, and Saturday from 10 AM to 4 PM.',
                'category': 'general',
                'keywords': 'hours, time, open, closed, schedule'
            },
            {
                'question': 'How can I reset my password?',
                'answer': 'You can reset your password by clicking on "Forgot Password" on the login page and following the instructions sent to your email.',
                'category': 'account',
                'keywords': 'password, reset, forgot, login, account'
            },
            {
                'question': 'What payment methods do you accept?',
                'answer': 'We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and bank transfers.',
                'category': 'payment',
                'keywords': 'payment, credit card, paypal, visa, mastercard'
            },
            {
                'question': 'How can I contact customer support?',
                'answer': 'You can reach our customer support team via email at support@company.com or call us at 1-800-123-4567.',
                'category': 'support',
                'keywords': 'contact, support, email, phone, help'
            },
            {
                'question': 'What is your return policy?',
                'answer': 'We offer a 30-day return policy for unused items in original packaging. Please contact support to initiate a return.',
                'category': 'policy',
                'keywords': 'return, refund, policy, exchange, unused'
            }
        ]
        
        # Insert sample data if table is empty
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM faq')
        count = cursor.fetchone()[0]
        
        if count == 0:
            for faq in sample_faqs:
                cursor.execute('''
                    INSERT INTO faq (question, answer, category, keywords)
                    VALUES (?, ?, ?, ?)
                ''', (faq['question'], faq['answer'], faq['category'], faq['keywords']))
            conn.commit()
        
        # Load FAQ data for similarity matching
        df = pd.read_sql_query('SELECT * FROM faq', conn)
        self.faq_data = df
        
        # Create TF-IDF vectors
        if not df.empty:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            self.faq_vectors = self.vectorizer.fit_transform(df['question'] + ' ' + df['keywords'])
        
        conn.close()
        logger.info(f"Loaded {len(self.faq_data)} FAQ entries")
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of user message"""
        if self.sentiment_analyzer:
            try:
                results = self.sentiment_analyzer(text)
                # Get the sentiment with highest score
                best_sentiment = max(results[0], key=lambda x: x['score'])
                return best_sentiment['label'], best_sentiment['score']
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
        
        # Fallback sentiment analysis
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                return 'POSITIVE', abs(compound)
            elif compound <= -0.05:
                return 'NEGATIVE', abs(compound)
            else:
                return 'NEUTRAL', abs(compound)
        except:
            return 'NEUTRAL', 0.5
    
    def find_best_knowledge_match(self, user_message):
        """Find the best matching answer from knowledge base"""
        if not self.knowledge_texts:
            return None, 0.0, None
        
        try:
            # Clean and preprocess user message
            cleaned_message = self.preprocess_text(user_message)
            
            # Vectorize user message
            user_vector = self.knowledge_vectorizer.transform([cleaned_message])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(user_vector, self.knowledge_vectors).flatten()
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score > 0.2:  # Lower threshold for more responses
                return (self.knowledge_answers[best_idx], 
                       best_score, 
                       self.knowledge_categories[best_idx])
            
        except Exception as e:
            logger.error(f"Error in knowledge base matching: {e}")
        
        return None, 0.0, None
    
    def preprocess_text(self, text):
        """Preprocess text for better matching"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text
    
    def extract_intent(self, user_message):
        """Extract user intent from message"""
        message_lower = user_message.lower()
        
        # Question patterns
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(word in message_lower for word in question_words):
            return 'question'
        
        # Request patterns
        request_words = ['tell me', 'explain', 'describe', 'show me', 'give me']
        if any(phrase in message_lower for phrase in request_words):
            return 'request'
        
        # Greeting patterns
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in message_lower for greeting in greetings):
            return 'greeting'
        
        # Gratitude patterns
        thanks = ['thank', 'thanks', 'appreciate']
        if any(thank in message_lower for thank in thanks):
            return 'gratitude'
        
        # Goodbye patterns
        goodbyes = ['bye', 'goodbye', 'see you', 'farewell', 'later']
        if any(goodbye in message_lower for goodbye in goodbyes):
            return 'goodbye'
        
        return 'general'
    
    def generate_response(self, user_message, session_id):
        """Generate chatbot response"""
        # Analyze sentiment
        sentiment_label, sentiment_score = self.analyze_sentiment(user_message)
        
        # Find FAQ match
        best_faq, confidence = self.find_best_faq_match(user_message)
        
        if best_faq is not None and confidence > 0.5:
            response = best_faq['answer']
        elif best_faq is not None and confidence > 0.3:
            response = f"I think you're asking about: {best_faq['question']}. {best_faq['answer']} \n\nIs this what you were looking for?"
        else:
            # Generate contextual response based on sentiment and keywords
            response = self.generate_fallback_response(user_message, sentiment_label)
        
        # Log conversation
        self.log_conversation(session_id, user_message, response, sentiment_score, confidence)
        
        return {
            'response': response,
            'sentiment': sentiment_label,
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_fallback_response(self, user_message, sentiment):
        """Generate fallback response when no FAQ match is found"""
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in user_message.lower() for greeting in greetings):
            return "Hello! I'm here to help you. You can ask me about our business hours, account issues, payments, or contact information. What would you like to know?"
        
        # Check for gratitude
        thanks = ['thank', 'thanks', 'appreciate']
        if any(thank in user_message.lower() for thank in thanks):
            return "You're welcome! Is there anything else I can help you with?"
        
        # Check for goodbye
        goodbyes = ['bye', 'goodbye', 'see you', 'farewell']
        if any(goodbye in user_message.lower() for goodbye in goodbyes):
            return "Goodbye! Have a great day! Feel free to come back if you have any more questions."
        
        # Sentiment-based responses
        if sentiment == 'NEGATIVE':
            return "I understand you might be frustrated. Let me try to help you. Could you please provide more details about your specific issue? You can also contact our support team directly for immediate assistance."
        elif sentiment == 'POSITIVE':
            return "That's great to hear! I'm here to help with any questions you might have about our services, policies, or general information."
        else:
            return "I'd be happy to help! I can assist you with questions about business hours, account issues, payments, returns, or contact information. What specific topic would you like to know more about?"
    
    def log_conversation(self, session_id, user_message, bot_response, sentiment_score, confidence):
        """Log conversation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (session_id, user_message, bot_response, sentiment, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, user_message, bot_response, sentiment_score, confidence))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    def get_conversation_history(self, session_id, limit=10):
        """Get conversation history for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_message, bot_response, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'user_message': row[0],
                    'bot_response': row[1],
                    'timestamp': row[2]
                })
            
            conn.close()
            return history[::-1]  # Reverse to get chronological order
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

# Initialize chatbot engine
chatbot = ChatbotEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Generate response
        response_data = chatbot.generate_response(user_message, session_id)
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/history/<session_id>')
def get_history(session_id):
    try:
        history = chatbot.get_conversation_history(session_id)
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analytics')
def get_analytics():
    try:
        conn = sqlite3.connect(chatbot.db_path)
        
        # Get conversation stats
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(sentiment), AVG(confidence) FROM conversations')
        avg_sentiment, avg_confidence = cursor.fetchone()
        
        # Get recent activity
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM conversations
            WHERE timestamp >= date('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''')
        
        daily_activity = []
        for row in cursor.fetchall():
            daily_activity.append({
                'date': row[0],
                'count': row[1]
            })
        
        conn.close()
        
        return jsonify({
            'total_conversations': total_conversations,
            'average_sentiment': avg_sentiment or 0,
            'average_confidence': avg_confidence or 0,
            'daily_activity': daily_activity
        })
    
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
