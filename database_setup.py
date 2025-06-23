# database_setup.py
import sqlite3
from datetime import datetime

def create_database():
    """Create the civic enforcement database with all required tables"""
    conn = sqlite3.connect('civic_enforcement.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            aadhaar_hash TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            education_level INTEGER,
            income_level INTEGER,
            city_type INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            consent_given BOOLEAN DEFAULT FALSE,
            consent_timestamp TIMESTAMP
        )
    ''')
    
    # Quiz results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question_1 INTEGER,
            question_2 INTEGER,
            question_3 INTEGER,
            question_4 INTEGER,
            question_5 INTEGER,
            total_score REAL,
            responsibility_level TEXT,
            taken_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Violations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT NOT NULL,
            confidence REAL,
            evidence_path TEXT,
            location TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            fine_amount REAL DEFAULT 0,
            paid BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Behavior predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS behavior_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            violation_probability REAL,
            risk_level TEXT,
            prediction_confidence REAL,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # System logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            details TEXT,
            ip_address TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert sample data for testing
    cursor.execute('''
        INSERT OR IGNORE INTO users (aadhaar_hash, email, phone, age, gender, 
                                   education_level, income_level, city_type, consent_given)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('sample_hash_123', 'test@example.com', '9876543210', 30, 'Male', 3, 2, 1, True))
    
    # Sample quiz data
    cursor.execute('''
        INSERT OR IGNORE INTO quiz_results (user_id, question_1, question_2, question_3, 
                                          question_4, question_5, total_score, responsibility_level)
        VALUES (1, 4, 3, 5, 4, 3, 3.8, 'High')
    ''')
    
    # Sample violation data
    sample_violations = [
        ('littering', 0.85, 'evidence_001.jpg', 'Park Area', 'confirmed', 500),
        ('footpath_violation', 0.72, 'evidence_002.jpg', 'Main Street', 'pending', 200),
        ('smoking_violation', 0.91, 'evidence_003.jpg', 'Bus Stop', 'confirmed', 1000)
    ]
    
    for violation in sample_violations:
        cursor.execute('''
            INSERT OR IGNORE INTO violations (type, confidence, evidence_path, 
                                            location, status, fine_amount)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', violation)
    
    conn.commit()
    conn.close()
    print("Database created successfully with sample data!")

def get_database_stats():
    """Get database statistics"""
    conn = sqlite3.connect('civic_enforcement.db')
    cursor = conn.cursor()
    
    stats = {}
    
    # Get table counts
    tables = ['users', 'quiz_results', 'violations', 'behavior_predictions', 'system_logs']
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {table}')
        stats[table] = cursor.fetchone()[0]
    
    # Get violation statistics
    cursor.execute('''
        SELECT type, COUNT(*) as count, AVG(confidence) as avg_confidence
        FROM violations 
        GROUP BY type
    ''')
    stats['violation_types'] = cursor.fetchall()
    
    conn.close()
    return stats

if __name__ == "__main__":
    create_database()
    stats = get_database_stats()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
