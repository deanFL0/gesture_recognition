import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="gesture_classification_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

gesture_labels = ["faster", "like", "next", "pause", "previous", "resume"]

CONFIDENCE_THRESHOLD = 0.8

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils   
mp_drawing_style = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.7) 

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't access webcam!")
        break

    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmark, 
                                     mp_hands.HAND_CONNECTIONS,
                                     mp_drawing_style.get_default_hand_landmarks_style(), 
                                     mp_drawing_style.get_default_hand_connections_style())

            landmarks = []
            for lm in hand_landmark.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            input_data = np.array(landmarks).flatten().astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            predicted_class_index = np.argmax(output_data)
            predicted_class_confidence = output_data[0][predicted_class_index]

            if predicted_class_confidence >= CONFIDENCE_THRESHOLD:
                predicted_label = gesture_labels[predicted_class_index]
                cv2.putText(frame, f"Gesture: {predicted_label} ({predicted_class_confidence:.2f})", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
            