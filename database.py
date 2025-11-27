import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Quiz sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Quiz questions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                question_number INTEGER NOT NULL,
                question TEXT NOT NULL,
                options TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                user_answer TEXT,
                is_correct INTEGER,
                response_time REAL,
                difficulty TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES quiz_sessions(id)
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                total_score INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0,
                total_time REAL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES quiz_sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username, email, password_hash):
        """Create a new user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return user_id
        except sqlite3.IntegrityError:
            return None
    
    def get_user_by_email(self, email):
        """Get user by email"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        return user
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        return user
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET last_login = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def create_session(self, user_id, topic, content):
        """Create a new quiz session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quiz_sessions (user_id, topic, content)
            VALUES (?, ?, ?)
        ''', (user_id, topic, content))
        
        session_id = cursor.lastrowid
        
        # Initialize performance metrics
        cursor.execute('''
            INSERT INTO performance_metrics (session_id)
            VALUES (?)
        ''', (session_id,))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def save_question(self, session_id, question_number, question, options, correct_answer, difficulty):
        """Save a quiz question"""
        import json
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quiz_questions 
            (session_id, question_number, question, options, correct_answer, difficulty)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, question_number, question, json.dumps(options), correct_answer, difficulty))
        
        question_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return question_id
    
    def update_answer(self, question_id, user_answer, is_correct, response_time):
        """Update question with user's answer"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE quiz_questions
            SET user_answer = ?, is_correct = ?, response_time = ?
            WHERE id = ?
        ''', (user_answer, 1 if is_correct else 0, response_time, question_id))
        
        conn.commit()
        conn.close()
    
    def update_performance_metrics(self, session_id):
        """Update performance metrics for a session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get session statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_questions,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_count,
                SUM(response_time) as total_time
            FROM quiz_questions
            WHERE session_id = ? AND user_answer IS NOT NULL
        ''', (session_id,))
        
        result = cursor.fetchone()
        
        total_questions = result['total_questions'] or 0
        correct_count = result['correct_count'] or 0
        total_time = result['total_time'] or 0
        
        total_score = correct_count * 10
        average_score = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        cursor.execute('''
            UPDATE performance_metrics
            SET total_score = ?, average_score = ?, total_time = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        ''', (total_score, average_score, total_time, session_id))
        
        conn.commit()
        conn.close()
    
    def get_performance_metrics(self, session_id):
        """Get performance metrics for a session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM performance_metrics
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_score': result['total_score'],
                'average_score': result['average_score'],
                'total_time': result['total_time']
            }
        
        return {'total_score': 0, 'average_score': 0, 'total_time': 0}
    
    def get_session_stats(self, session_id):
        """Get statistics for a quiz session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_answered,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_answers
            FROM quiz_questions
            WHERE session_id = ? AND user_answer IS NOT NULL
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_answered': result['total_answered'] or 0,
            'correct_answers': result['correct_answers'] or 0
        }
    
    def get_ml_training_data(self, user_id=None, limit=1000):
        """
        Get training data for ML model
        Returns: List of (features, next_difficulty) tuples
        """
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if user_id:
            # Get data for specific user
            query = """
            SELECT 
                q1.difficulty as current_diff,
                q1.is_correct as is_correct,
                q1.response_time as response_time,
                q2.difficulty as next_diff,
                s.id as session_id
            FROM quiz_questions q1
            JOIN quiz_questions q2 ON q1.session_id = q2.session_id 
                AND q2.question_number = q1.question_number + 1
            JOIN quiz_sessions s ON q1.session_id = s.id
            WHERE s.user_id = ?
            ORDER BY q1.id DESC
            LIMIT ?
            """
            cursor.execute(query, (user_id, limit))
        else:
            # Get data across all users
            query = """
            SELECT 
                q1.difficulty as current_diff,
                q1.is_correct as is_correct,
                q1.response_time as response_time,
                q2.difficulty as next_diff,
                s.id as session_id
            FROM quiz_questions q1
            JOIN quiz_questions q2 ON q1.session_id = q2.session_id 
                AND q2.question_number = q1.question_number + 1
            JOIN quiz_sessions s ON q1.session_id = s.id
            ORDER BY q1.id DESC
            LIMIT ?
            """
            cursor.execute(query, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        training_data = []
        
        difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
        
        for row in results:
            # Create simplified features for each question
            features = [
                difficulty_map[row['current_diff']],
                1 if row['is_correct'] else 0,
                row['response_time'],
                0.5,  # Default accuracy
                0,    # Questions answered
                30,   # Avg response time
                0,    # Correct streak
                0     # Incorrect streak
            ]
            
            next_diff = difficulty_map[row['next_diff']]
            
            training_data.append((features, next_diff))
        
        return training_data
    
    def get_session_features(self, session_id):
        """Get current session statistics for ML feature extraction"""
        
        stats = self.get_session_stats(session_id)
        
        # Calculate additional features
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get average response time
        cursor.execute("""
            SELECT AVG(response_time) as avg_time
            FROM quiz_questions
            WHERE session_id = ? AND response_time IS NOT NULL
        """, (session_id,))
        
        result = cursor.fetchone()
        avg_time = result['avg_time'] if result['avg_time'] else 30
        
        # Calculate current streak
        cursor.execute("""
            SELECT is_correct
            FROM quiz_questions
            WHERE session_id = ?
            ORDER BY question_number DESC
            LIMIT 5
        """, (session_id,))
        
        recent = cursor.fetchall()
        conn.close()
        
        # Calculate streak
        streak = 0
        if recent:
            for q in recent:
                if q['is_correct'] == recent[0]['is_correct']:
                    streak += 1
                else:
                    break
        
        stats['avg_response_time'] = avg_time
        stats['current_streak'] = streak
        
        return stats
