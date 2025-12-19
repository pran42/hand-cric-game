import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from collections import deque
import os

class HandGestureDetector:
    def __init__(self, model_path='models/hand_cricket_model.h5'):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Try to load trained model
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.use_ml_model = False
        
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                self.scaler = joblib.load('models/scaler.pkl')
                self.label_encoder = joblib.load('models/label_encoder.pkl')
                self.use_ml_model = True
                print("✅ ML model loaded successfully!")
            else:
                print("⚠️  No trained model found. Using rule-based detection.")
        except Exception as e:
            print(f"❌ Error loading ML model: {e}. Using rule-based detection.")
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        self.confidence_threshold = 0.6
        
    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize landmarks"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks).reshape(1, -1)
    
    def custom_number_detection(self, hand_landmarks):
        """
        Custom hand number detection based on specific gestures:
        - 0: Fist closed (no fingers extended)
        - 1: Only index finger extended
        - 2: Index + Middle fingers extended
        - 3: Index + Middle + Ring fingers extended  
        - 4: Index + Middle + Ring + Pinky fingers extended
        - 5: All five fingers extended (Index + Middle + Ring + Pinky + Thumb)
        - 6: Only thumb extended
        """
        # Landmark indices
        WRIST = 0
        THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
        INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
        MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
        RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
        PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
        
        def is_finger_extended(tip, pip, mcp=None, is_thumb=False):
            """Check if finger is fully extended"""
            if is_thumb:
                # For thumb, check if it's extended away from hand (not folded)
                return tip.x < pip.x  # Thumb tip is to the left of PIP when extended
            else:
                # For other fingers, check if tip is above PIP joint (finger extended up)
                return tip.y < pip.y
        
        # Get landmarks
        landmarks = hand_landmarks.landmark
        
        # Check each finger individually
        thumb_extended = is_finger_extended(landmarks[THUMB_TIP], landmarks[THUMB_IP], landmarks[THUMB_MCP], is_thumb=True)
        index_extended = is_finger_extended(landmarks[INDEX_TIP], landmarks[INDEX_PIP], landmarks[INDEX_MCP])
        middle_extended = is_finger_extended(landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP], landmarks[MIDDLE_MCP])
        ring_extended = is_finger_extended(landmarks[RING_TIP], landmarks[RING_PIP], landmarks[RING_MCP])
        pinky_extended = is_finger_extended(landmarks[PINKY_TIP], landmarks[PINKY_PIP], landmarks[PINKY_MCP])
        
        # Debug finger states
        finger_states = {
            'thumb': thumb_extended,
            'index': index_extended,
            'middle': middle_extended,
            'ring': ring_extended,
            'pinky': pinky_extended
        }
        
        # Apply custom numbering rules
        if not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return '0'  # Fist closed
        
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
            return '1'  # Only index finger
        
        elif index_extended and middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
            return '2'  # Index + Middle
        
        elif index_extended and middle_extended and ring_extended and not pinky_extended and not thumb_extended:
            return '3'  # Index + Middle + Ring
        
        elif index_extended and middle_extended and ring_extended and pinky_extended and not thumb_extended:
            return '4'  # Index + Middle + Ring + Pinky
        
        elif index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended:
            return '5'  # All five fingers
        
        elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return '6'  # Only thumb
        
        else:
            # If no clear pattern, return 'none'
            return 'none'
    
    def predict_gesture(self, frame):
        """Predict hand number from frame using custom detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        predicted_number = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                if self.use_ml_model and self.model is not None:
                    # Use ML model prediction
                    try:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        landmarks_scaled = self.scaler.transform(landmarks)
                        prediction = self.model.predict(landmarks_scaled, verbose=0)
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        if confidence > self.confidence_threshold:
                            predicted_number = self.label_encoder.inverse_transform([predicted_class])[0]
                            
                            # Smooth predictions using buffer
                            self.prediction_buffer.append(predicted_number)
                            
                            # Get most frequent prediction from buffer
                            if len(self.prediction_buffer) >= 3:
                                from collections import Counter
                                most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
                                predicted_number = most_common
                    except Exception as e:
                        print(f"ML prediction error: {e}")
                        predicted_number = self.custom_number_detection(hand_landmarks)
                        confidence = 0.8
                else:
                    # Use custom rule-based detection
                    predicted_number = self.custom_number_detection(hand_landmarks)
                    confidence = 0.8
                
                # Get hand bounding box for better visualization
                h, w, c = frame.shape
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)
                
                return predicted_number, confidence, frame
        
        return None, 0, frame

# Standalone webcam hand number detector with custom gestures
def webcam_hand_number_detector():
    """Standalone function to test custom hand number detection with webcam"""
    detector = HandGestureDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("🎥 Webcam Hand Number Detection Started!")
    print("📝 CUSTOM GESTURE MAPPING:")
    print("   0: Fist closed (no fingers)")
    print("   1: Only index finger")
    print("   2: Index + Middle fingers") 
    print("   3: Index + Middle + Ring fingers")
    print("   4: Index + Middle + Ring + Pinky fingers")
    print("   5: All five fingers extended")
    print("   6: Only thumb extended")
    print("   Make sure your hand is clearly visible")
    print("   Press 'q' to quit")
    print("-" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand number
        predicted_number, confidence, processed_frame = detector.predict_gesture(frame)
        
        # Display prediction with custom colors
        if predicted_number and predicted_number != 'none':
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            
            # Show the detected number large in the center
            cv2.putText(processed_frame, f"NUMBER: {predicted_number}", 
                       (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            cv2.putText(processed_frame, f"Confidence: {confidence:.2%}", 
                       (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
            
            # Show gesture description
            gesture_descriptions = {
                '0': 'Fist Closed',
                '1': 'Index Finger Only',
                '2': 'Index + Middle',
                '3': 'Index + Middle + Ring',
                '4': 'Index + Middle + Ring + Pinky', 
                '5': 'All Five Fingers',
                '6': 'Thumb Only'
            }
            
            description = gesture_descriptions.get(predicted_number, 'Unknown Gesture')
            cv2.putText(processed_frame, description, 
                       (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        else:
            cv2.putText(processed_frame, "Show hand gesture to camera", 
                       (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, "See console for gesture mapping", 
                       (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Display instructions at bottom
        cv2.putText(processed_frame, "Press 'q' to quit", 
                   (20, processed_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Hand Number Detection - Custom Gestures', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam detection closed.")

# Training data collection for custom gestures
def collect_custom_gesture_data():
    """Collect training data specifically for the custom gesture mapping"""
    import csv
    from tqdm import tqdm
    
    detector = HandGestureDetector()
    cap = cv2.VideoCapture(0)
    
    print("📊 Custom Gesture Data Collection")
    print("Collecting data for your specific number system...")
    
    gestures = ['0', '1', '2', '3', '4', '5', '6']
    samples_per_gesture = 50
    
    for gesture in gestures:
        print(f"\n🔄 Collecting data for gesture: {gesture}")
        print(f"   Please show: {get_gesture_description(gesture)}")
        input("   Press Enter when ready...")
        
        samples_collected = 0
        while samples_collected < samples_per_gesture:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = detector.extract_landmarks(results.multi_hand_landmarks[0])
                # Save to CSV (you would implement this)
                samples_collected += 1
                print(f"   Collected {samples_collected}/{samples_per_gesture}")
            
            cv2.putText(frame, f"Gesture {gesture}: {get_gesture_description(gesture)}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{samples_per_gesture}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

def get_gesture_description(number):
    """Get description for each custom gesture"""
    descriptions = {
        '0': 'Fist closed (no fingers extended)',
        '1': 'Only index finger extended', 
        '2': 'Index + Middle fingers extended',
        '3': 'Index + Middle + Ring fingers extended',
        '4': 'Index + Middle + Ring + Pinky fingers extended',
        '5': 'All five fingers extended',
        '6': 'Only thumb extended'
    }
    return descriptions.get(number, 'Unknown gesture')

if __name__ == "__main__":
    webcam_hand_number_detector()