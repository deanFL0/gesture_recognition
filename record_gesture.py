import cv2
import os
import mediapipe as mp
import csv

# Functino to save gesture data to csv
def save_data(data, label):
    save_dir = "gesture_data"
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{save_dir}/{label}_{len(os.listdir(save_dir))}.csv"
    with open(file_name,mode='w',newline='') as file:
        writer = csv.writer(file)
        header = [f"{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']]
        writer.writerow(header)
        for frame in data:
            row = [coord for landmark in frame for coord in landmark]
            writer.writerow(row)
    print(f"Gesture {label} saved to {file_name}")

while True:
    print("Enter gesture name and hit 'ENTER' or enter 'exit' to close")
    label = input("Gesture name: ")
    if label.lower() == 'exit':
        print("Application ended")
        break
    
    recording = False
    sequence = []

    # Initialize mediapipe and webcam
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils   
    mp_drawing_style = mp.solutions.drawing_styles
    hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.7, max_num_hands=1,)  

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
                landmarks = [[lm.x, lm.y,lm.z] for lm in hand_landmarks.landmark]
                
                base_palm = landmarks[0]
                normalized_landmarks = [[lm[0] - base_palm[0],
                                        lm[1] - base_palm[1],
                                        lm[2] - base_palm[2]] for lm in landmarks]
            if recording:
                sequence.append(normalized_landmarks)
                if len(sequence) == 30:
                    save_data(sequence, label)
                    sequence = []
                    break
                    
        
        cv2.putText(frame, "Click 'r' to start recording", (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Record Gesture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()