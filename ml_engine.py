"""
ML Difficulty Engine for MetaQuizzer
Uses scikit-learn RandomForest to predict optimal quiz difficulty
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import warnings

class MLDifficultyEngine:
    """
    Machine Learning-based adaptive difficulty engine using scikit-learn
    Trains on user performance history to predict optimal next difficulty
    """
    
    def __init__(self, model_path='models/difficulty_model.pkl'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
        self.difficulty_names = ['easy', 'medium', 'hard']
        
        # Minimum samples needed before using ML
        self.min_training_samples = 10
        
        # Load pre-trained model if exists
        self.load_model()
    
    def extract_features(self, session_stats, current_difficulty, is_correct, response_time):
        """
        Extract features from current question and user history
        
        Features:
        1. Current difficulty level (0, 1, 2)
        2. Is current answer correct (0 or 1)
        3. Response time (seconds)
        4. Overall accuracy (percentage)
        5. Questions answered so far
        6. Average response time
        7. Streak of correct answers
        8. Streak of incorrect answers
        """
        
        total_answered = session_stats.get('total_answered', 0)
        correct_answers = session_stats.get('correct_answers', 0)
        accuracy = correct_answers / total_answered if total_answered > 0 else 0.5
        
        # Calculate streaks
        correct_streak = session_stats.get('current_streak', 0) if is_correct else 0
        incorrect_streak = 0 if is_correct else session_stats.get('current_streak', 0)
        
        features = [
            self.difficulty_map[current_difficulty],  # Current difficulty
            1 if is_correct else 0,                   # Current correctness
            response_time,                             # Response time
            accuracy,                                  # Overall accuracy
            total_answered,                            # Questions answered
            session_stats.get('avg_response_time', 30), # Average response time
            correct_streak,                            # Correct streak
            incorrect_streak                           # Incorrect streak
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, training_data):
        """
        Train the ML model on historical data with accuracy evaluation
        
        training_data: List of tuples (features, next_difficulty)
        """
        
        if len(training_data) < self.min_training_samples:
            print(f"⚠️  Not enough data to train ML model ({len(training_data)}/{self.min_training_samples})")
            return False
        
        # Separate features and labels
        X = np.array([item[0] for item in training_data])
        y = np.array([item[1] for item in training_data])
        
        # Check for class diversity
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"\n⚠️  WARNING: Training data has only one difficulty class!")
            print(f"   All samples are: {self.difficulty_names[unique_classes[0]]}")
            print(f"   Need diverse data (easy, medium, hard) for ML to work properly.")
            print(f"\n   💡 TIP: Answer some questions correctly and some wrong")
            print(f"   to create varied difficulty progression.\n")
            print(f"   Using fallback formula until you have varied data.\n")
            return False
        
        # Split into train and test sets (80-20 split)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(unique_classes) > 1 else None
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Unique classes: {len(unique_classes)} {[self.difficulty_names[c] for c in unique_classes]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"\nTraining Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_predictions = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        # Evaluate on test set
        test_predictions = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Get feature importances
        feature_names = [
            'Current Difficulty',
            'Is Correct',
            'Response Time',
            'Overall Accuracy',
            'Questions Answered',
            'Avg Response Time',
            'Correct Streak',
            'Incorrect Streak'
        ]
        
        importances = self.model.feature_importances_
        
        # Display results
        print(f"\n{'='*60}")
        print(f"MODEL TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"\n📊 Accuracy Scores:")
        print(f"   Training Accuracy: {train_accuracy:.2%}")
        print(f"   Testing Accuracy:  {test_accuracy:.2%}")
        
        if abs(train_accuracy - test_accuracy) > 0.15:
            print(f"   ⚠️  Warning: Large gap suggests overfitting")
        elif test_accuracy > 0.75:
            print(f"   ✓ Excellent model performance!")
        elif test_accuracy > 0.60:
            print(f"   ✓ Good model performance")
        else:
            print(f"   ⚠️  Model needs more training data")
        
        # Feature importance
        print(f"\n🔍 Feature Importance:")
        feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        for i, (name, importance) in enumerate(feature_importance[:5], 1):
            print(f"   {i}. {name}: {importance:.3f}")
        
        # Confusion matrix with proper labels
        print(f"\n📈 Confusion Matrix:")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = confusion_matrix(y_test, test_predictions, labels=[0, 1, 2])
        
        print(f"   Predicted →")
        print(f"   Actual ↓    Easy  Medium  Hard")
        for i, row_label in enumerate(['Easy', 'Medium', 'Hard']):
            row_str = '  '.join([f"{val:4d}" for val in cm[i]])
            print(f"   {row_label:8} {row_str}")
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = classification_report(
                y_test, 
                test_predictions, 
                target_names=['Easy', 'Medium', 'Hard'],
                labels=[0, 1, 2],
                zero_division=0
            )
        print(report)
        
        print(f"{'='*60}\n")
        
        # Save model with metadata
        self.save_model_with_metrics(train_accuracy, test_accuracy, feature_importance)
        
        return True
    
    def predict_next_difficulty(self, features):
        """
        Predict optimal next difficulty using trained ML model
        
        Returns: 'easy', 'medium', or 'hard'
        """
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            
            # Get probabilities - handle single class case
            try:
                probabilities = self.model.predict_proba(features_scaled)[0]
                
                # Ensure we have probabilities for all 3 classes
                full_probs = [0.0, 0.0, 0.0]
                classes = self.model.classes_
                
                for i, cls in enumerate(classes):
                    full_probs[cls] = probabilities[i]
                
                print(f"\n[ML Prediction]")
                print(f"  Easy: {full_probs[0]:.1%}")
                print(f"  Medium: {full_probs[1]:.1%}")
                print(f"  Hard: {full_probs[2]:.1%}")
                print(f"  → Predicted: {self.difficulty_names[prediction]}\n")
                
            except Exception as prob_error:
                # If probability extraction fails, just show prediction
                print(f"\n[ML Prediction]")
                print(f"  → Predicted: {self.difficulty_names[prediction]}")
                print(f"  (Confidence unavailable - limited training data)\n")
            
            return self.difficulty_names[prediction]
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Fall back to formula
            return self.fallback_prediction(features)
    
    def save_model_with_metrics(self, train_accuracy, test_accuracy, feature_importance):
        """Save model with accuracy metrics"""
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'timestamp': datetime.now(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ ML model saved to {self.model_path}")
    
    def load_model(self):
        """Load pre-trained model with accuracy info"""
        
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                
                print(f"\n{'='*60}")
                print(f"ML MODEL LOADED")
                print(f"{'='*60}")
                print(f"Trained on: {model_data['timestamp']}")
                
                if 'train_accuracy' in model_data:
                    print(f"Training Accuracy: {model_data['train_accuracy']:.2%}")
                    print(f"Testing Accuracy: {model_data['test_accuracy']:.2%}")
                
                print(f"{'='*60}\n")
                
                return True
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                return False
        
        return False
    
    def get_model_accuracy(self):
        """Get current model accuracy"""
        
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                return {
                    'train_accuracy': model_data.get('train_accuracy', 0),
                    'test_accuracy': model_data.get('test_accuracy', 0),
                    'timestamp': model_data.get('timestamp', None)
                }
            except:
                return None
        
        return None
    
    def has_enough_data(self, training_data):
        """Check if we have enough data to use ML"""
        return len(training_data) >= self.min_training_samples
    
    def fallback_prediction(self, features):
        """
        Fallback to simple rule-based system if ML model not ready
        Uses the original formula: Df = Dp + α × (Rc - Ra)
        """
        
        Dp = features[0][0]  # Current difficulty
        Rc = features[0][1]  # Correctness
        Ra = features[0][3]  # Accuracy
        alpha = 0.5
        
        Df = Dp + alpha * (Rc - Ra)
        Df = max(0, min(2, Df))
        
        difficulty_index = round(Df)
        
        return self.difficulty_names[difficulty_index]
