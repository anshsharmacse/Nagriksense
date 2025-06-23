# violation_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import os

class ViolationDetector:
    def __init__(self, model_path='models/yolov8n.pt', db_path='civic_enforcement.db'):
        """Initialize the violation detection system"""
        self.model = YOLO(model_path)
        self.db_path = db_path
        self.violation_classes = {
            'bottle': 'littering',
            'cup': 'littering', 
            'can': 'littering',
            'cigarette': 'smoking_violation',
            'person': 'footpath_violation'
        }
        
    def detect_violations(self, image_path):
        """Detect civic violations in an image"""
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                return None, "Could not load image"
            
            # Run YOLO detection
            results = self.model(image)
            
            violations = []
            annotated_image = image.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection details
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        if confidence > 0.5 and class_name in self.violation_classes:
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Record violation
                            violation = {
                                'type': self.violation_classes[class_name],
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'timestamp': datetime.now().isoformat()
                            }
                            violations.append(violation)
                            
                            # Draw bounding box on image
                            cv2.rectangle(annotated_image, (int(x1), int(y1)), 
                                        (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(annotated_image, 
                                      f"{violation['type']}: {confidence:.2f}",
                                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 0, 255), 2)
            
            # Save evidence image if violations detected
            if violations:
                evidence_path = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(evidence_path, annotated_image)
                self._log_violations(violations, evidence_path)
                
            return violations, evidence_path if violations else None
            
        except Exception as e:
            return None, f"Detection error: {str(e)}"
    
    def process_video_stream(self, source=0):
        """Process real-time video stream for violation detection"""
        cap = cv2.VideoCapture(source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection every 10 frames to reduce computational load
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 == 0:
                # Save frame temporarily
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Detect violations
                violations, _ = self.detect_violations(temp_path)
                
                if violations:
                    print(f"Violations detected: {len(violations)}")
                    for violation in violations:
                        print(f"- {violation['type']}: {violation['confidence']:.2f}")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Display frame (for testing)
            cv2.imshow('Civic Violation Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _log_violations(self, violations, evidence_path):
        """Log violations to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for violation in violations:
                cursor.execute('''
                    INSERT INTO violations (type, confidence, evidence_path, timestamp, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (violation['type'], violation['confidence'], evidence_path, 
                     violation['timestamp'], 'pending'))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Database error: {e}")

# Fallback detection using traditional computer vision
class FallbackDetector:
    def __init__(self):
        """Initialize fallback detection methods"""
        self.cascade_classifiers = {}
        
    def detect_littering_basic(self, image):
        """Basic littering detection using color and shape analysis"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common litter items
        bottle_colors = [
            ([100, 50, 50], [130, 255, 255]),  # Blue bottles
            ([0, 50, 50], [10, 255, 255]),     # Red bottles
        ]
        
        detections = []
        for lower, upper in bottle_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'type': 'potential_littering',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.6
                    })
        
        return detections

if __name__ == "__main__":
    # Test the violation detector
    detector = ViolationDetector()
    
    # Test with image
    violations, evidence = detector.detect_violations("test_image.jpg")
    if violations:
        print(f"Found {len(violations)} violations")
        for v in violations:
            print(f"- {v['type']}: {v['confidence']:.2f}")
    
    # Test with webcam (uncomment to test)
    # detector.process_video_stream(0)
