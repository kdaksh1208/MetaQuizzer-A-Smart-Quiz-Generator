import os
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DATABASE_PATH = 'metaquizzer.db'
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Removed DIFFICULTY_LEVELS, TIME_THRESHOLDS, ACCURACY_THRESHOLD
    # Now using mathematical formula instead
