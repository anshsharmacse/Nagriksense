# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import sqlite3
import hashlib
import re
from datetime import datetime
from aadhaar_validator import AadhaarValidator, AadhaarCompliance
from behavior_prediction import BehaviorPredictor

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize components
aadhaar_validator = AadhaarValidator()
behavior_predictor = BehaviorPredictor()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def register():
    """User registration with Aadhaar validation"""
    if request.method == 'POST':
        try:
            # Get form data
            aadhaar = request.form.get('aadhaar', '').strip()
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            age = int(request.form.get('age', 0))
            gender = request.form.get('gender', '')
            education = int(request.form.get('education', 1))
            income = int(request.form.get('income', 1))
            city_type = int(request.form.get('city_type', 1))
            consent = request.form.get('consent') == 'on'
            
            # Validate inputs
            if not all([aadhaar, email, phone, age, gender]):
                return jsonify({'success': False, 'message': 'All fields are required'})
            
            if not consent:
                return jsonify({'success': False, 'message': 'Consent is required'})
            
            # Validate Aadhaar
            is_valid, message = aadhaar_validator.validate_aadhaar(aadhaar)
            if not is_valid:
                return jsonify({'success': False, 'message': message})
            
            # Simulate UIDAI verification
            verified, result = aadhaar_validator.simulate_uidai_verification(aadhaar, consent)
            if not verified:
                return jsonify({'success': False, 'message': result})
            
            # Hash Aadhaar for storage
            aadhaar_hash = hashlib.sha256(aadhaar.encode()).hexdigest()
            
            # Store in database
            conn = sqlite3.connect('civic_enforcement.db')
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO users (aadhaar_hash, email, phone, age, gender, 
                                     education_level, income_level, city_type, 
                                     consent_given, consent_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (aadhaar_hash, email, phone, age, gender, education, 
                     income, city_type, consent, datetime.now()))
                
                user_id = cursor.lastrowid
                conn.commit()
                
                # Log consent
                AadhaarCompliance.log_consent(aadhaar, request.remote_addr, consent)
                
                # Store user session
                session['user_id'] = user_id
                session['aadhaar_masked'] = aadhaar_validator.mask_aadhaar(aadhaar)
                
                return jsonify({
                    'success': True, 
                    'message': 'Registration successful!',
                    'redirect': url_for('quiz')
                })
                
            except sqlite3.IntegrityError:
                return jsonify({'success': False, 'message': 'User already registered'})
            
            finally:
                conn.close()
                
        except Exception as e:
            return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})
    
    return render_template('register.html', 
                         consent_text=AadhaarCompliance.get_consent_text())

@app.route('/validate_aadhaar', methods=['POST'])
@limiter.limit("20 per minute")
def validate_aadhaar_endpoint():
    """AJAX endpoint for real-time Aadhaar validation"""
    aadhaar = request.json.get('aadhaar', '')
    
    is_valid, message = aadhaar_validator.validate_aadhaar(aadhaar)
    
    return jsonify({
        'valid': is_valid,
        'message': message,
        'masked': aadhaar_validator.mask_aadhaar(aadhaar) if is_valid else ''
    })

@app.route('/quiz')
def quiz():
    """Civic sense quiz page"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    quiz_questions = [
        {
            'id': 1,
            'question': 'Do you always throw trash in designated dustbins?',
            'options': [
                ('5', 'Always'),
                ('4', 'Usually'),
                ('3', 'Sometimes'),
                ('2', 'Rarely'),
                ('1', 'Never')
            ]
        },
        {
            'id': 2,
            'question': 'Do you use footpaths for walking instead of roads?',
            'options': [
                ('5', 'Always'),
                ('4', 'Usually'),
                ('3', 'Sometimes'),
                ('2', 'Rarely'),
                ('1', 'Never')
            ]
        },
        {
            'id': 3,
            'question': 'Do you follow traffic rules and signals?',
            'options': [
                ('5', 'Always'),
                ('4', 'Usually'),
                ('3', 'Sometimes'),
                ('2', 'Rarely'),
                ('1', 'Never')
            ]
        },
        {
            'id': 4,
            'question': 'Do you help others follow civic norms?',
            'options': [
                ('5', 'Always'),
                ('4', 'Usually'),
                ('3', 'Sometimes'),
                ('2', 'Rarely'),
                ('1', 'Never')
            ]
        },
        {
            'id': 5,
            'question': 'Do you take care of public property?',
            'options': [
                ('5', 'Always'),
                ('4', 'Usually'),
                ('3', 'Sometimes'),
                ('2', 'Rarely'),
                ('1', 'Never')
            ]
        }
    ]
    
    return render_template('quiz.html', questions=quiz_questions)

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    """Process quiz submission"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please log in first'})
    
    try:
        user_id = session['user_id']
        
        # Get quiz answers
        answers = []
        for i in range(1, 6):
            answer = int(request.form.get(f'question_{i}', 0))
            answers.append(answer)
        
        # Calculate score
        total_score = sum(answers) / len(answers)
        
        # Determine responsibility level
        if total_score >= 4.5:
            responsibility_level = "Excellent"
        elif total_score >= 3.5:
            responsibility_level = "Good"
        elif total_score >= 2.5:
            responsibility_level = "Average"
        else:
            responsibility_level = "Needs Improvement"
        
        # Store quiz results
        conn = sqlite3.connect('civic_enforcement.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quiz_results (user_id, question_1, question_2, question_3, 
                                    question_4, question_5, total_score, responsibility_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, *answers, total_score, responsibility_level))
        
        conn.commit()
        
        # Get user data for behavior prediction
        cursor.execute('''
            SELECT age, gender, education_level, income_level, city_type
            FROM users WHERE id = ?
        ''', (user_id,))
        user_data = cursor.fetchone()
        
        conn.close()
        
        # Predict behavior
        if user_data:
            prediction_data = {
                'age': user_data[0],
                'gender': user_data[1],
                'education_level': user_data[2],
                'income_level': user_data[3],
                'city_type': user_data[4],
                'quiz_score': total_score,
                'previous_violations': 0,  # New user
                'social_influence': 3.0,   # Average
                'community_involvement': total_score,  # Based on quiz
                'awareness_score': total_score
            }
            
            behavior_result = behavior_predictor.predict_behavior(prediction_data)
            
            return jsonify({
                'success': True,
                'score': total_score,
                'level': responsibility_level,
                'prediction': behavior_result,
                'redirect': url_for('profile')
            })
        
        else:
            return jsonify({
                'success': True,
                'score': total_score,
                'level': responsibility_level,
                'redirect': url_for('profile')
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Quiz submission error: {str(e)}'})

@app.route('/profile')
def profile():
    """User profile and dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    user_id = session['user_id']
    
    conn = sqlite3.connect('civic_enforcement.db')
    cursor = conn.cursor()
    
    # Get user data
    cursor.execute('''
        SELECT u.email, u.phone, u.age, u.gender, u.created_at,
               qr.total_score, qr.responsibility_level, qr.taken_at
        FROM users u
        LEFT JOIN quiz_results qr ON u.id = qr.user_id
        WHERE u.id = ?
        ORDER BY qr.taken_at DESC
        LIMIT 1
    ''', (user_id,))
    
    user_data = cursor.fetchone()
    
    # Get violations
    cursor.execute('''
        SELECT type, confidence, timestamp, status, fine_amount
        FROM violations
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    
    violations = cursor.fetchall()
    
    # Get behavior predictions
    cursor.execute('''
        SELECT violation_probability, risk_level, prediction_confidence, predicted_at
        FROM behavior_predictions
        WHERE user_id = ?
        ORDER BY predicted_at DESC
        LIMIT 1
    ''', (user_id,))
    
    prediction = cursor.fetchone()
    
    conn.close()
    
    return render_template('profile.html', 
                         user_data=user_data,
                         violations=violations,
                         prediction=prediction,
                         aadhaar_masked=session.get('aadhaar_masked'))

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'success': False, 'message': 'Rate limit exceeded. Please try again later.'}), 429

if __name__ == '__main__':
    # Ensure database exists
    from database_setup import create_database
    create_database()
    
    # Load ML model
    behavior_predictor.load_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
