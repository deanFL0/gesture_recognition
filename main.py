import os
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("gesture_classifier.h5")
gesture_labels = ["faster", "like", "next", "previous", "resume", "stop"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils   
mp_drawing_style = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)  

sequence = []
sentence = []
predictions = []
THRESHOLD = 0.8

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't access webcam!")
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Hand detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_style.get_default_hand_landmarks_style(), mp_drawing_style.get_default_hand_connections_style())

            landmark = [[lm.x, lm.y,lm.z] for lm in hand_landmarks.landmark]
            base_palm = landmark[0]
            normalized_landmarks = [[lm[0] - base_palm[0],
                                    lm[1] - base_palm[1],
                                    lm[2] - base_palm[2]] for lm in landmark]
            normalized_landmarks = np.array(normalized_landmarks).flatten()
            sequence.append(normalized_landmarks)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(gesture_labels[np.argmax(res)])
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > THRESHOLD:
                        if len(sentence) > 0:
                            if gesture_labels[np.argmax(res)] != sentence[-1]:
                                sentence.append(gesture_labels[np.argmax(res)])
                        else:
                            sentence.append(gesture_labels[np.argmax(res)])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]

            cv2.rectangle(frame, (0,0), (640,40), (245,117,16),-1)
            cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()