from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from google import genai
from config import Config
from database import Database
from ml_engine import MLDifficultyEngine
import json
import time
import os
from dotenv import load_dotenv
import random

app = Flask(__name__)
app.config.from_object(Config)

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load and configure Gemini API
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

print("\n" + "="*60)
print("GEMINI API CONFIGURATION")
print("="*60)

if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found!")
    raise ValueError("GEMINI_API_KEY not found!")
else:
    print(f"✓ API Key: {api_key[:15]}...{api_key[-10:]}")

try:
    client = genai.Client(api_key=api_key)
    print("✓ Google GenAI client ready")
    print("="*60 + "\n")
except Exception as e:
    print(f"❌ Error: {e}")
    raise

db = Database(app.config['DATABASE_PATH'])

# Initialize ML Engine
print("Initializing ML Engine...")
ml_engine = MLDifficultyEngine()
print("✓ ML Engine ready\n")

# Question cache to avoid rate limits
question_cache = {}

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    user_data = db.get_user_by_id(int(user_id))
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['email'])
    return None

class SimpleDifficultyEngine:
    """Improved fallback formula-based engine"""
    def __init__(self):
        self.difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
        self.difficulty_names = ['easy', 'medium', 'hard']
        self.alpha = 1.0
    
    def calculate_next_difficulty(self, current_difficulty, is_correct, response_time, session_stats):
        """
        Improved formula: Df = Dp + α × (Rc - Ra) + speed_bonus
        """
        Dp = self.difficulty_map[current_difficulty]
        Rc = 1 if is_correct else 0
        
        total_answered = session_stats.get('total_answered', 1)
        correct_answers = session_stats.get('correct_answers', 0)
        
        Ra = correct_answers / total_answered if total_answered > 0 else 0.5
        
        Df = Dp + self.alpha * (Rc - Ra)
        
        avg_time = session_stats.get('avg_response_time', 30)
        if is_correct and response_time < avg_time * 0.7:
            Df += 0.5
            speed_bonus = True
        else:
            speed_bonus = False
        
        if total_answered >= 3 and Ra > 0.9 and Dp < 2:
            Df = min(Dp + 1, 2)
            print(f"  🚀 High accuracy detected! Increasing difficulty.")
        
        if total_answered >= 3 and Ra < 0.4 and Dp > 0:
            Df = max(Dp - 1, 0)
            print(f"  💡 Struggling detected. Decreasing difficulty.")
        
        Df = max(0, min(2, Df))
        difficulty_index = round(Df)
        next_difficulty = self.difficulty_names[difficulty_index]
        
        print(f"\n[Fallback Formula]")
        print(f"  Current: {current_difficulty} (Dp={Dp})")
        print(f"  Answer: {'✓ Correct' if is_correct else '✗ Wrong'} (Rc={Rc})")
        print(f"  Response time: {response_time:.1f}s (avg: {avg_time:.1f}s)")
        if speed_bonus:
            print(f"  ⚡ Speed bonus applied!")
        print(f"  Average accuracy: {Ra:.1%} (Ra={Ra:.2f})")
        print(f"  Formula: Df = {Dp} + {self.alpha} × ({Rc} - {Ra:.2f}) = {Df:.2f}")
        print(f"  Next: {next_difficulty}\n")
        
        return next_difficulty

simple_engine = SimpleDifficultyEngine()

def generate_fallback_question(topic, difficulty):
    """Generate simple offline questions when API is unavailable"""
    fallback_questions = {
        'easy': [
            {'question': f'What is the most basic concept in {topic}?', 'options': {'A': 'Foundation', 'B': 'Advanced theory', 'C': 'Complex formula', 'D': 'Research'}, 'correct_answer': 'A'},
            {'question': f'Which is related to {topic}?', 'options': {'A': 'Study', 'B': 'Nothing', 'C': 'Unrelated topic', 'D': 'Random fact'}, 'correct_answer': 'A'},
            {'question': f'What best describes {topic}?', 'options': {'A': 'Important subject', 'B': 'Irrelevant', 'C': 'Useless', 'D': 'Unknown'}, 'correct_answer': 'A'},
        ],
        'medium': [
            {'question': f'What is an intermediate concept in {topic}?', 'options': {'A': 'Applied knowledge', 'B': 'Basic info', 'C': 'Expert level', 'D': 'None'}, 'correct_answer': 'A'},
            {'question': f'How would you apply {topic}?', 'options': {'A': 'Practical use', 'B': 'Never use it', 'C': 'Forget it', 'D': 'Ignore it'}, 'correct_answer': 'A'},
        ],
        'hard': [
            {'question': f'What is an advanced concept in {topic}?', 'options': {'A': 'Expert theory', 'B': 'Basics', 'C': 'Simple fact', 'D': 'Common knowledge'}, 'correct_answer': 'A'},
            {'question': f'What research exists about {topic}?', 'options': {'A': 'Extensive studies', 'B': 'No research', 'C': 'Unknown', 'D': 'Basic info'}, 'correct_answer': 'A'},
        ]
    }
    
    questions = fallback_questions.get(difficulty, fallback_questions['easy'])
    return random.choice(questions)

def generate_question_safe(topic, content, difficulty, previous_questions, question_num):
    """Generate questions with smart caching to avoid rate limits"""
    
    print(f"\n[Q{question_num}] Generating {difficulty} question about {topic}...")
    
    # Create cache key
    cache_key = f"{topic}_{difficulty}"
    
    # Initialize cache for this key if not exists
    if cache_key not in question_cache:
        question_cache[cache_key] = []
    
    # Check if we have cached questions
    cached_questions = question_cache[cache_key]
    
    # If we have unused cached questions, use them
    if len(cached_questions) > 0:
        question = cached_questions.pop(0)
        print(f"  ✓ Using cached question (API call saved!)\n")
        return question
    
    # Need to generate new questions
    print(f"  📡 Calling API to generate 5 questions at once...")
    
    # Build avoid list
    avoid_list = ""
    if previous_questions and len(previous_questions) > 0:
        avoid_list = "\n\nDO NOT repeat these questions:\n"
        for i, q in enumerate(previous_questions[-5:], 1):
            avoid_list += f"- {q[:70]}\n"
    
    has_content = content and not content.startswith("General knowledge about")
    
    if has_content:
        prompt = f"""Generate 5 DIFFERENT {difficulty} difficulty multiple choice questions about {topic}.

Content: {content[:300]}
{avoid_list}

Return ONLY a JSON array with 5 unique questions:
[
  {{"question":"Question 1?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"A"}},
  {{"question":"Question 2?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"B"}},
  {{"question":"Question 3?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"C"}},
  {{"question":"Question 4?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"D"}},
  {{"question":"Question 5?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"A"}}
]

All questions must be completely different from each other."""
    else:
        prompt = f"""Generate 5 DIFFERENT {difficulty} difficulty multiple choice questions about {topic}.
{avoid_list}

Return ONLY a JSON array with 5 unique questions:
[
  {{"question":"Question 1?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"A"}},
  {{"question":"Question 2?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"B"}},
  {{"question":"Question 3?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"C"}},
  {{"question":"Question 4?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"D"}},
  {{"question":"Question 5?","options":{{"A":"opt1","B":"opt2","C":"opt3","D":"opt4"}},"correct_answer":"A"}}
]

All questions must be completely different from each other."""

    model_name = 'gemini-2.0-flash-exp'
    
    for attempt in range(2):
        try:
            print(f"  Attempt {attempt + 1}/2...")
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            text = response.text.strip()
            text = text.replace('``````', '').strip()
            
            start = text.find('[')
            end = text.rfind(']')
            
            if start >= 0 and end > start:
                json_str = text[start:end+1]
                questions = json.loads(json_str)
                
                if isinstance(questions, list) and len(questions) > 0:
                    valid_questions = []
                    
                    for q in questions:
                        if 'question' in q and 'options' in q and 'correct_answer' in q:
                            if len(q['options']) == 4:
                                valid_questions.append(q)
                    
                    if len(valid_questions) > 0:
                        for q in valid_questions[1:]:
                            question_cache[cache_key].append(q)
                        
                        first_q = valid_questions[0]
                        print(f"  ✓ Generated {len(valid_questions)} questions, using first one")
                        print(f"  ✓ Cached {len(valid_questions)-1} questions for next use\n")
                        return first_q
                        
        except Exception as e:
            error_msg = str(e)
            
            if '429' in error_msg or 'rate' in error_msg.lower():
                if attempt < 1:
                    wait_time = 65
                    print(f"  ⚠️ Rate limit hit! Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"\n  ❌ Rate limit exceeded. API quota exhausted.")
                    print(f"  💡 Using fallback question to continue quiz.\n")
                    return generate_fallback_question(topic, difficulty)
            elif '503' in error_msg:
                print(f"  ⚠️ Model busy, waiting 10s...")
                time.sleep(10)
            else:
                print(f"  ✗ Error: {error_msg[:70]}")
                time.sleep(2)
    
    print(f"  ✗ API failed. Using fallback question.\n")
    return generate_fallback_question(topic, difficulty)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        data = request.json
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        user_data = db.get_user_by_email(email)
        
        if user_data and bcrypt.check_password_hash(user_data['password_hash'], password):
            user = User(user_data['id'], user_data['username'], user_data['email'])
            login_user(user, remember=data.get('remember', False))
            db.update_last_login(user_data['id'])
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        data = request.json
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        
        if db.get_user_by_email(email):
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user_id = db.create_user(username, email, password_hash)
        
        if user_id:
            user = User(user_id, username, email)
            login_user(user)
            return jsonify({'success': True, 'message': 'Account created successfully'})
        else:
            return jsonify({'success': False, 'message': 'Username or email already exists'}), 400
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=current_user.username)

@app.route('/api/start-quiz', methods=['POST'])
@login_required
def start_quiz():
    try:
        data = request.json
        topic = data.get('topic', '').strip()
        content = data.get('content', '').strip()
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        if not content:
            content = f"General knowledge about {topic}"
        
        session_id = db.create_session(current_user.id, topic, content)
        
        session['quiz_session_id'] = session_id
        session['topic'] = topic
        session['content'] = content
        session['current_question'] = 0
        session['difficulty'] = 'easy'
        session['previous_questions'] = []
        
        return jsonify({'session_id': session_id, 'message': 'Quiz started'})
        
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-question', methods=['POST'])
@login_required
def get_question():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        topic = session.get('topic')
        content = session.get('content')
        difficulty = session.get('difficulty', 'easy')
        current_q = session.get('current_question', 0)
        previous_questions = session.get('previous_questions', [])
        
        if current_q >= 10:
            return jsonify({'completed': True})
        
        if not topic:
            return jsonify({'error': 'Session expired'}), 400
        
        question_data = generate_question_safe(topic, content, difficulty, previous_questions, current_q + 1)
        
        if not question_data:
            return jsonify({'error': 'Unable to generate question. Please try again.'}), 500
        
        question_id = db.save_question(
            session_id, current_q + 1, question_data['question'],
            question_data['options'], question_data['correct_answer'], difficulty
        )
        
        session['current_question'] = current_q + 1
        previous_questions.append(question_data['question'])
        session['previous_questions'] = previous_questions
        session['current_question_id'] = question_id
        session['question_start_time'] = time.time()
        
        return jsonify({
            'question_number': current_q + 1,
            'question': question_data['question'],
            'options': question_data['options'],
            'difficulty': difficulty,
            'total_questions': 10
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit-answer', methods=['POST'])
@login_required
def submit_answer():
    try:
        data = request.json
        user_answer = data.get('answer')
        question_id = session.get('current_question_id')
        session_id = session.get('quiz_session_id')
        
        if not user_answer or not question_id:
            return jsonify({'error': 'Invalid request'}), 400
        
        start_time = session.get('question_start_time', time.time())
        response_time = time.time() - start_time
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT correct_answer, difficulty FROM quiz_questions WHERE id = ?', (question_id,))
        result = cursor.fetchone()
        conn.close()
        
        correct_answer = result['correct_answer']
        current_difficulty = result['difficulty']
        is_correct = (user_answer.upper() == correct_answer.upper())
        
        db.update_answer(question_id, user_answer, is_correct, response_time)
        db.update_performance_metrics(session_id)
        
        stats = db.get_session_features(session_id)
        
        features = ml_engine.extract_features(
            session_stats=stats,
            current_difficulty=current_difficulty,
            is_correct=is_correct,
            response_time=response_time
        )
        
        next_difficulty = None
        ml_used = False
        
        try:
            training_data = db.get_ml_training_data(user_id=current_user.id, limit=100)
            
            print(f"\n{'='*60}")
            print(f"DIFFICULTY PREDICTION")
            print(f"{'='*60}")
            print(f"Training samples available: {len(training_data)}")
            
            if ml_engine.has_enough_data(training_data):
                if not os.path.exists(ml_engine.model_path):
                    print("\n🔄 Training initial ML model...")
                    success = ml_engine.train_model(training_data)
                    
                    if success:
                        next_difficulty = ml_engine.predict_next_difficulty(features)
                        ml_used = True
                        print(f"✓ Using ML prediction: {next_difficulty}")
                    else:
                        next_difficulty = simple_engine.calculate_next_difficulty(
                            current_difficulty=current_difficulty,
                            is_correct=is_correct,
                            response_time=response_time,
                            session_stats=stats
                        )
                        ml_used = False
                else:
                    next_difficulty = ml_engine.predict_next_difficulty(features)
                    ml_used = True
                    print(f"✓ Using ML prediction: {next_difficulty}")
            
            else:
                print(f"\n⚠️  Not enough training data ({len(training_data)}/{ml_engine.min_training_samples})")
                next_difficulty = simple_engine.calculate_next_difficulty(
                    current_difficulty=current_difficulty,
                    is_correct=is_correct,
                    response_time=response_time,
                    session_stats=stats
                )
                ml_used = False
                print(f"✓ Using fallback formula: {next_difficulty}")
        
        except Exception as ml_error:
            print(f"\n❌ ML prediction error: {ml_error}")
            print("Falling back to simple formula engine...")
            
            next_difficulty = simple_engine.calculate_next_difficulty(
                current_difficulty=current_difficulty,
                is_correct=is_correct,
                response_time=response_time,
                session_stats=stats
            )
            ml_used = False
        
        print(f"{'='*60}\n")
        
        session['difficulty'] = next_difficulty
        
        metrics = db.get_performance_metrics(session_id)
        
        return jsonify({
            'is_correct': is_correct,
            'correct_answer': correct_answer,
            'next_difficulty': next_difficulty,
            'response_time': round(response_time, 2),
            'ml_prediction': ml_used,
            'metrics': {
                'total_score': metrics.get('total_score', 0),
                'average_score': round(metrics.get('average_score', 0), 0),
                'total_time': round(metrics.get('total_time', 0) / 60, 0),
                'questions_answered': stats.get('total_answered', 0)
            }
        })
    
    except Exception as e:
        print(f"\n❌ ERROR in submit_answer: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-metrics', methods=['POST'])
@login_required
def get_metrics():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        metrics = db.get_performance_metrics(session_id)
        stats = db.get_session_stats(session_id)
        
        return jsonify({
            'total_score': metrics.get('total_score', 0),
            'average_score': round(metrics.get('average_score', 0), 0),
            'study_time': round(metrics.get('total_time', 0) / 60, 0),
            'flashcard_sets': 1,
            'questions_answered': stats.get('total_answered', 0)
        })
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml-stats', methods=['GET'])
@login_required
def get_ml_stats():
    try:
        accuracy_data = ml_engine.get_model_accuracy()
        model_exists = os.path.exists(ml_engine.model_path)
        training_data = db.get_ml_training_data(current_user.id, limit=100)
        all_training_data = db.get_ml_training_data(limit=1000)
        
        response_data = {
            'model_exists': model_exists,
            'model_path': ml_engine.model_path if model_exists else None,
            'train_accuracy': round(accuracy_data['train_accuracy'], 4) if accuracy_data else 0,
            'test_accuracy': round(accuracy_data['test_accuracy'], 4) if accuracy_data else 0,
            'trained_on': accuracy_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if accuracy_data and accuracy_data['timestamp'] else None,
            'user_training_samples': len(training_data),
            'total_training_samples': len(all_training_data),
            'min_samples_needed': ml_engine.min_training_samples,
            'ready_for_ml': len(all_training_data) >= ml_engine.min_training_samples,
            'user_ready_for_ml': len(training_data) >= ml_engine.min_training_samples
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"ERROR in ml-stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain-ml', methods=['POST'])
@login_required
def retrain_ml_model():
    try:
        training_data = db.get_ml_training_data(limit=1000)
        
        if len(training_data) < ml_engine.min_training_samples:
            return jsonify({
                'success': False,
                'message': f'Not enough training data. Need {ml_engine.min_training_samples}, have {len(training_data)}'
            }), 400
        
        print("\n🔄 Manual retraining triggered by user...")
        success = ml_engine.train_model(training_data)
        
        if success:
            metrics = ml_engine.get_model_accuracy()
            return jsonify({
                'success': True,
                'message': 'Model retrained successfully!',
                'train_accuracy': metrics['train_accuracy'],
                'test_accuracy': metrics['test_accuracy'],
                'training_samples': len(training_data)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Model training failed'
            }), 500
    
    except Exception as e:
        print(f"ERROR in retrain-ml: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
