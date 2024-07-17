import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Hand Lib
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start Webcam
cap = cv2.VideoCapture(0)

# Tip and base
def get_gesture(landmarks):
    # Thumb 
    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]
    # Index finger 
    index_tip = landmarks[8]
    index_base = landmarks[5]
    # Middle finger 
    middle_tip = landmarks[12]
    middle_base = landmarks[9]
    # Ring finger 
    ring_tip = landmarks[16]
    # Pinky 
    pinky_tip = landmarks[20]

    # Check if finger is extended
    def is_finger_extended(tip, base):
        return tip.y < base.y

    # Thumbs up: thumb tip is above thumb base and other fingers are curled
    if thumb_tip.y < thumb_base.y and not any(is_finger_extended(tip, base) for tip, base in 
                                              [(index_tip, index_base), (middle_tip, middle_base)]):
        return "Yes"
        print("yes")
    
    # Thumbs down: thumb tip is below thumb base and other fingers are curled
    elif thumb_tip.y > thumb_base.y and not any(is_finger_extended(tip, base) for tip, base in 
                                                [(index_tip, index_base), (middle_tip, middle_base)]):
        return "No"
        print("no")
    
    # Okay: distance between thumb tip and index tip is small, other fingers extended
    elif (np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])) < 0.1 and
          is_finger_extended(middle_tip, middle_base) and 
          is_finger_extended(ring_tip, landmarks[13]) and 
          is_finger_extended(pinky_tip, landmarks[17])):
        return "Okay"
        print("Okay")
    
    # Peace: index and middle fingers are extended, others are curled
    elif (is_finger_extended(index_tip, index_base) and 
          is_finger_extended(middle_tip, middle_base) and 
          not is_finger_extended(ring_tip, landmarks[13]) and 
          not is_finger_extended(pinky_tip, landmarks[17])):
        return "Peace"
        print("Peace")
    
    else:
        return "Unknown"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Empty Frame")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing hands and text
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            gesture = get_gesture(landmarks)
            cv2.rectangle(image, (40,50), (400,110),(20, 20, 20), -1)
            cv2.putText(image, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Sign Language', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
