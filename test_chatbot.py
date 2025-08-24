import unittest
import json
import tempfile
import os
from app import app, ChatbotEngine

class ChatbotTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create temporary database for testing
        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        
    def tearDown(self):
        """Clean up after tests"""
        os.close(self.db_fd)
        os.unlink(app.config['DATABASE'])
    
    def test_index_page(self):
        """Test the main page loads correctly"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI ChatBot', response.data)
    
    def test_chat_endpoint_valid_message(self):
        """Test chat endpoint with valid message"""
        response = self.app.post('/api/chat',
                                data=json.dumps({
                                    'message': 'Hello, how are you?',
                                    'session_id': 'test_session'
                                }),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)
        self.assertIn('sentiment', data)
        self.assertIn('confidence', data)
        self.assertIn('timestamp', data)
    
    def test_chat_endpoint_empty_message(self):
        """Test chat endpoint with empty message"""
        response = self.app.post('/api/chat',
                                data=json.dumps({
                                    'message': '',
                                    'session_id': 'test_session'
                                }),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_chat_endpoint_faq_matching(self):
        """Test FAQ matching functionality"""
        response = self.app.post('/api/chat',
                                data=json.dumps({
                                    'message': 'What are your business hours?',
                                    'session_id': 'test_session'
                                }),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('Monday to Friday', data['response'])
    
    def test_analytics_endpoint(self):
        """Test analytics endpoint"""
        response = self.app.get('/api/analytics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('total_conversations', data)
        self.assertIn('average_sentiment', data)
        self.assertIn('average_confidence', data)
    
    def test_history_endpoint(self):
        """Test conversation history endpoint"""
        # First send a message
        self.app.post('/api/chat',
                     data=json.dumps({
                         'message': 'Test message',
                         'session_id': 'test_session_history'
                     }),
                     content_type='application/json')
        
        # Then get history
        response = self.app.get('/api/history/test_session_history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('history', data)

class ChatbotEngineTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.chatbot = ChatbotEngine()
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        # Test positive sentiment
        sentiment, score = self.chatbot.analyze_sentiment("I love this service!")
        self.assertIn(sentiment.upper(), ['POSITIVE', 'LABEL_2'])
        self.assertGreater(score, 0)
        
        # Test negative sentiment
        sentiment, score = self.chatbot.analyze_sentiment("This is terrible!")
        self.assertIn(sentiment.upper(), ['NEGATIVE', 'LABEL_0'])
        self.assertGreater(score, 0)
    
    def test_faq_matching(self):
        """Test FAQ matching functionality"""
        # Test exact match
        faq, confidence = self.chatbot.find_best_faq_match("What are your business hours?")
        self.assertIsNotNone(faq)
        self.assertGreater(confidence, 0.5)
        
        # Test similar match
        faq, confidence = self.chatbot.find_best_faq_match("When are you open?")
        self.assertIsNotNone(faq)
        
        # Test no match
        faq, confidence = self.chatbot.find_best_faq_match("Random unrelated question about quantum physics")
        self.assertLess(confidence, 0.3)
    
    def test_response_generation(self):
        """Test response generation"""
        response_data = self.chatbot.generate_response("Hello", "test_session")
        
        self.assertIn('response', response_data)
        self.assertIn('sentiment', response_data)
        self.assertIn('confidence', response_data)
        self.assertIn('timestamp', response_data)
        self.assertIsInstance(response_data['response'], str)
        self.assertGreater(len(response_data['response']), 0)
    
    def test_fallback_responses(self):
        """Test fallback response generation"""
        # Test greeting
        response_data = self.chatbot.generate_response("Hello", "test_session")
        self.assertIn('hello', response_data['response'].lower())
        
        # Test thanks
        response_data = self.chatbot.generate_response("Thank you", "test_session")
        self.assertIn('welcome', response_data['response'].lower())
        
        # Test goodbye
        response_data = self.chatbot.generate_response("Goodbye", "test_session")
        self.assertIn('goodbye', response_data['response'].lower())

def run_performance_test():
    """Run performance tests"""
    import time
    
    print("🚀 Running Performance Tests...")
    chatbot = ChatbotEngine()
    
    # Test response time
    start_time = time.time()
    for i in range(10):
        chatbot.generate_response(f"Test message {i}", f"session_{i}")
    end_time = time.time()
    
    avg_response_time = (end_time - start_time) / 10
    print(f"⏱️ Average response time: {avg_response_time:.3f} seconds")
    
    if avg_response_time < 1.0:
        print("✅ Performance test passed!")
    else:
        print("⚠️ Performance test warning: Response time is slow")

def run_integration_test():
    """Run integration tests"""
    print("🔗 Running Integration Tests...")
    
    try:
        # Test database connection
        chatbot = ChatbotEngine()
        print("✅ Database connection successful")
        
        # Test model loading
        if chatbot.sentiment_analyzer:
            print("✅ Sentiment analyzer loaded")
        else:
            print("⚠️ Sentiment analyzer not loaded (using fallback)")
        
        if chatbot.qa_pipeline:
            print("✅ QA pipeline loaded")
        else:
            print("⚠️ QA pipeline not loaded (using fallback)")
        
        # Test FAQ data
        if not chatbot.faq_data.empty:
            print(f"✅ FAQ data loaded: {len(chatbot.faq_data)} entries")
        else:
            print("❌ FAQ data not loaded")
        
        print("✅ Integration tests completed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

if __name__ == '__main__':
    print("🧪 Starting Chatbot Tests...")
    print("=" * 50)
    
    # Run unit tests
    print("📋 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 50)
    
    # Run performance tests
    run_performance_test()
    
    print("\n" + "=" * 50)
    
    # Run integration tests
    run_integration_test()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
