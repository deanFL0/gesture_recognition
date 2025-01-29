import os
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("gesture_classifier.h5")

# Load class mapping from TXT file
class_mapping = {}
with open("class_mapping.txt", "r") as file:
    lines = file.readlines()[1:]  # Skip header
    for line in lines:
        class_name, encoded_value = line.strip().split(": ")
        class_mapping[int(encoded_value)] = class_name

# Initilize mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils   
mp_drawing_style = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6)

sequence = []
sentence = []
predictions = []
THRESHOLD = 0.8 # Change threshold as needed

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't access webcam!")
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    landmarks = []

    # Face 
    face_landmarks = results.face_landmarks.landmark if results.face_landmarks else []
    landmarks.extend([[lm.x, lm.y, lm.z] for lm in face_landmarks])
    landmarks.extend([[0, 0, 0]] * (468 - len(face_landmarks))) 

    # Left hand
    left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
    landmarks.extend([[lm.x, lm.y, lm.z] for lm in left_hand_landmarks])
    landmarks.extend([[0, 0, 0]] * (21 - len(left_hand_landmarks)))

    # Right hand
    right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
    landmarks.extend([[lm.x, lm.y, lm.z] for lm in right_hand_landmarks])
    landmarks.extend([[0, 0, 0]] * (21 - len(right_hand_landmarks)))

    # Pose
    pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
    landmarks.extend([[lm.x, lm.y, lm.z] for lm in pose_landmarks])
    landmarks.extend([[0, 0, 0]] * (33 - len(pose_landmarks)))

    # Make sure landmark recorded correctly
    assert len(landmarks) == (468 + 21 + 21 + 33), "Total landmark not compitable"

    # Add landmark to sequence
    landmarks = np.array(landmarks).flatten()
    sequence.append(landmarks)
    sequence = sequence[-30:] 

    # Predict gesture
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        predicted_class = np.argmax(res)

        if predicted_class in class_mapping:
            predicted_label = class_mapping[predicted_class]
            print(predicted_label)
            predictions.append(predicted_class)

            # Show result if above the threshold
            if np.unique(predictions[-10:])[0] == predicted_class:
                if res[predicted_class] > THRESHOLD:
                    if len(sentence) > 0:
                        if predicted_label != sentence[-1]:
                            sentence.append(predicted_label)
                    else:
                        sentence.append(predicted_label)

            if len(sentence) > 5:
                sentence = sentence[-5:]

    # Draw landmark
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, 
            mp_holistic.FACEMESH_CONTOURS, 
            mp_drawing_style.get_default_face_mesh_tesselation_style(), 
            mp_drawing_style.get_default_face_mesh_contours_style(),
            mp_drawing_style.get_default_face_mesh_iris_connections_style()
        )
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Show prediction
    cv2.rectangle(frame, (0,0), (640,40), (245,117,16),-1)
    cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
