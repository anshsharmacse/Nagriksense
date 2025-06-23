# behavior_prediction.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sqlite3
from datetime import datetime

class BehaviorPredictor:
    def __init__(self, db_path='civic_enforcement.db'):
        """Initialize behavior prediction system"""
        self.db_path = db_path
        self.model = None
        self.feature_columns = [
            'age', 'gender_encoded', 'education_level', 'income_level',
            'city_type', 'quiz_score', 'previous_violations', 
            'social_influence', 'age_education_interaction',
            'quiz_violation_ratio', 'risk_score', 'community_involvement',
            'awareness_score'
        ]
        self.label_encoders = {}
        
    def generate_training_data(self, n_samples=5000):
        """Generate realistic training data based on Indian demographics"""
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(35, 12, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'education_level': np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                             p=[0.15, 0.25, 0.30, 0.20, 0.10]),
            'income_level': np.random.choice([1, 2, 3], n_samples, p=[0.40, 0.45, 0.15]),
            'city_type': np.random.choice([1, 2, 3], n_samples, p=[0.30, 0.50, 0.20]),
            'quiz_score': np.random.normal(3.2, 1.2, n_samples).clip(0, 5),
            'previous_violations': np.random.poisson(1.5, n_samples),
            'social_influence': np.random.uniform(1, 5, n_samples),
            'community_involvement': np.random.uniform(1, 5, n_samples),
            'awareness_score': np.random.uniform(1, 5, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Clip age to realistic range
        df['age'] = df['age'].clip(18, 100)
        
        # Feature engineering
        df['age_education_interaction'] = df['age'] * df['education_level']
        df['quiz_violation_ratio'] = df['quiz_score'] / (df['previous_violations'] + 1)
        df['risk_score'] = (df['previous_violations'] * 2 + 
                           (5 - df['quiz_score']) + 
                           (5 - df['social_influence'])) / 3
        
        # Create target variable based on realistic patterns
        violation_probability = (
            0.1 +  # Base probability
            0.15 * (df['previous_violations'] > 2).astype(int) +
            0.1 * (df['quiz_score'] < 2.5).astype(int) +
            0.05 * (df['age'] < 25).astype(int) +
            0.05 * (df['education_level'] < 3).astype(int) +
            0.05 * (df['income_level'] == 1).astype(int) -
            0.1 * (df['community_involvement'] > 3.5).astype(int) -
            0.05 * (df['awareness_score'] > 4).astype(int)
        ).clip(0, 0.8)
        
        df['will_violate'] = np.random.binomial(1, violation_probability)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        df_processed = df.copy()
        
        # Encode categorical variables
        if 'gender' in df_processed.columns:
            if 'gender' not in self.label_encoders:
                self.label_encoders['gender'] = LabelEncoder()
                df_processed['gender_encoded'] = self.label_encoders['gender'].fit_transform(df_processed['gender'])
            else:
                df_processed['gender_encoded'] = self.label_encoders['gender'].transform(df_processed['gender'])
        
        return df_processed[self.feature_columns]
    
    def train_model(self):
        """Train the behavior prediction model"""
        print("Generating training data...")
        df = self.generate_training_data()
        
        print("Preparing features...")
        X = self.prepare_features(df)
        y = df['will_violate']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Save model
        joblib.dump(self.model, 'behavior_prediction_model.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        
        return accuracy
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('behavior_prediction_model.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            return True
        except FileNotFoundError:
            print("Model not found. Please train the model first.")
            return False
    
    def predict_behavior(self, user_data):
        """Predict violation likelihood for a user"""
        if self.model is None:
            if not self.load_model():
                return None
        
        # Prepare user data
        df_user = pd.DataFrame([user_data])
        X_user = self.prepare_features(df_user)
        
        # Make prediction
        probability = self.model.predict_proba(X_user)[0]
        prediction = self.model.predict(X_user)[0]
        
        # Risk categorization
        risk_prob = probability[1]  # Probability of violation
        if risk_prob < 0.3:
            risk_level = "Low"
        elif risk_prob < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'violation_probability': risk_prob,
            'risk_level': risk_level,
            'prediction': bool(prediction),
            'confidence': max(probability)
        }
    
    def update_model_with_new_data(self, new_data):
        """Update model with new user data"""
        # This would implement online learning or periodic retraining
        pass

if __name__ == "__main__":
    # Train and test the behavior prediction model
    predictor = BehaviorPredictor()
    
    # Train model
    accuracy = predictor.train_model()
    print(f"Model trained with accuracy: {accuracy:.3f}")
    
    # Test prediction
    test_user = {
        'age': 28,
        'gender': 'Male',
        'education_level': 3,
        'income_level': 2,
        'city_type': 1,
        'quiz_score': 2.5,
        'previous_violations': 3,
        'social_influence': 2.8,
        'community_involvement': 2.0,
        'awareness_score': 2.5
    }
    
    result = predictor.predict_behavior(test_user)
    if result:
        print(f"Prediction for test user:")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Violation Probability: {result['violation_probability']:.3f}")
