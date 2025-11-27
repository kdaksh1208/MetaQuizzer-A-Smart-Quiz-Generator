"""
ML Model Training Script for MetaQuizzer
========================================
This script trains the ML model on existing quiz data.

Usage:
    python train_ml_model.py
    python train_ml_model.py --status

Requirements:
    - At least 10 completed quiz questions in database
    - database.py and ml_engine.py must be in same folder
"""

from database import Database
from ml_engine import MLDifficultyEngine
from config import Config
import os

def train_initial_model():
    """Train ML model on existing quiz data with accuracy reporting"""
    
    print("\n" + "="*60)
    print("ML MODEL TRAINING - MetaQuizzer")
    print("="*60 + "\n")
    
    # Initialize
    print("Initializing database and ML engine...")
    db = Database(Config.DATABASE_PATH)
    ml_engine = MLDifficultyEngine()
    print("✓ Initialized\n")
    
    # Get training data
    print("Fetching training data from database...")
    training_data = db.get_ml_training_data(limit=1000)
    
    print(f"Found {len(training_data)} training samples")
    
    # Check if we have enough data
    if len(training_data) < ml_engine.min_training_samples:
        print(f"\n⚠️  WARNING: Not enough training data!")
        print(f"Minimum required: {ml_engine.min_training_samples} samples")
        print(f"Currently have: {len(training_data)} samples")
        print(f"\nNeed {ml_engine.min_training_samples - len(training_data)} more samples.")
        print("\n💡 HOW TO FIX:")
        print("   1. Run your Flask app: python app.py")
        print("   2. Complete at least 2-3 full quizzes (10 questions each)")
        print("   3. Run this script again\n")
        return False
    
    # Train the model (includes accuracy calculation)
    print("\n" + "-"*60)
    print("TRAINING MODEL...")
    print("-"*60 + "\n")
    
    success = ml_engine.train_model(training_data)
    
    if success:
        # Get and display accuracy metrics
        metrics = ml_engine.get_model_accuracy()
        
        print("\n" + "="*60)
        print("✓ MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {ml_engine.model_path}")
        print(f"Training samples: {len(training_data)}")
        
        if metrics:
            print(f"\n📊 Final Model Accuracy:")
            print(f"   Training Accuracy: {metrics['train_accuracy']:.2%}")
            print(f"   Testing Accuracy:  {metrics['test_accuracy']:.2%}")
            
            # Accuracy interpretation
            test_acc = metrics['test_accuracy']
            if test_acc >= 0.85:
                print(f"   ✓ Excellent! Model is highly accurate.")
            elif test_acc >= 0.70:
                print(f"   ✓ Good! Model performs well.")
            elif test_acc >= 0.60:
                print(f"   ⚠️  Moderate. Consider getting more training data.")
            else:
                print(f"   ⚠️  Low accuracy. Need more diverse training data.")
        
        print(f"\n✓ You can now use ML predictions in your quiz app!")
        print("="*60 + "\n")
        return True
    else:
        print("\n" + "="*60)
        print("✗ MODEL TRAINING FAILED")
        print("="*60 + "\n")
        return False

def check_database_status():
    """Check how much data is available for training"""
    
    print("\n" + "="*60)
    print("DATABASE STATUS CHECK")
    print("="*60 + "\n")
    
    db = Database(Config.DATABASE_PATH)
    
    # Check if database exists
    if not os.path.exists(Config.DATABASE_PATH):
        print("❌ Database not found!")
        print(f"Expected location: {Config.DATABASE_PATH}")
        print("\nCreate database by running your app first:")
        print("   python app.py\n")
        return
    
    # Get statistics
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as count FROM users")
    user_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM quiz_sessions")
    session_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM quiz_questions")
    question_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM quiz_questions WHERE is_correct IS NOT NULL")
    answered_count = cursor.fetchone()['count']
    
    conn.close()
    
    print(f"📊 Database Statistics:")
    print(f"   Users: {user_count}")
    print(f"   Quiz sessions: {session_count}")
    print(f"   Questions generated: {question_count}")
    print(f"   Questions answered: {answered_count}")
    
    # Get training data count
    training_data = db.get_ml_training_data(limit=1000)
    min_samples = MLDifficultyEngine().min_training_samples
    
    print(f"\n📈 ML Training Data:")
    print(f"   Training samples available: {len(training_data)}")
    print(f"   Minimum required: {min_samples}")
    
    if len(training_data) >= min_samples:
        print(f"\n   ✓ Ready for ML training!")
    else:
        needed = min_samples - len(training_data)
        print(f"\n   ⚠️  Need {needed} more training samples")
        print(f"   (Complete ~{(needed // 9) + 1} more quizzes)")
    
    # Check if model already exists
    ml_engine = MLDifficultyEngine()
    if os.path.exists(ml_engine.model_path):
        print(f"\n🤖 Existing Model Found:")
        metrics = ml_engine.get_model_accuracy()
        if metrics:
            print(f"   Trained on: {metrics['timestamp']}")
            print(f"   Training Accuracy: {metrics['train_accuracy']:.2%}")
            print(f"   Testing Accuracy: {metrics['test_accuracy']:.2%}")
        else:
            print(f"   Location: {ml_engine.model_path}")
    else:
        print(f"\n🤖 No trained model found yet.")
    
    print("="*60 + "\n")

if __name__ == '__main__':
    import sys
    
    # Check if user wants database status only
    if len(sys.argv) > 1 and sys.argv[1] == '--status':
        check_database_status()
    else:
        # Check database first
        check_database_status()
        
        # Ask user to proceed
        print("Proceed with training? (y/n): ", end='')
        try:
            choice = input().strip().lower()
        except KeyboardInterrupt:
            print("\n\nTraining cancelled.\n")
            sys.exit(0)
        
        if choice == 'y':
            train_initial_model()
        else:
            print("\nTraining cancelled.\n")
