import cv2
import os
import mediapipe as mp
import csv
import time 

# Functin to save gesture data to csv
def save_data(data, label):
    save_dir = "gesture_data"
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{save_dir}/{label}_{len(os.listdir(save_dir))}.csv"
    with open(file_name,mode='w',newline='') as file:
        writer = csv.writer(file)
        # Landmark header (468 + 21 + 21 + 33 = 543 landmark)
        header = [f"{i}_{coord}" for i in range(543) for coord in ['x', 'y', 'z']]
        writer.writerow(header)
        for frame in data:
            row = [coord for landmark in frame for coord in landmark]
            writer.writerow(row)
    print(f"Gesture {label} saved to {file_name}")

# Set Max Sequence to record
max_sequence = 30
# Set wait tim (second)
wait_time = 3

while True:
    print("Enter gesture name and hit 'ENTER' or enter 'exit' to close")
    label = input("Gesture name: ")
    if label.lower() == 'exit':
        print("Application ended")
        break
    
    recording = False
    sequence = []

    # Initialize mediapipe and webcam
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils   
    mp_drawing_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't access webcam!")
                break

            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            landmarks = []

            # Face landmark
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing_style.get_default_face_mesh_tesselation_style(),
                    mp_drawing_style.get_default_face_mesh_contours_style(),
                    mp_drawing_style.get_default_face_mesh_iris_connections_style()
                )
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
            else:
                landmarks.extend([0,0,0] * 468) # Placeholder
            
            #Left hand landmark
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            else:
                landmarks.extend([[0, 0, 0]] * 21)  # Placeholder

            # Right hand
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            else:
                landmarks.extend([[0, 0, 0]] * 21)  # Placeholder

            # Pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS
                )
                landmarks.extend([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            else:
                landmarks.extend([[0, 0, 0]] * 33)  # Placeholder

            # Save data if recording
            if recording:
                sequence.append(landmarks)
                if len(sequence) == max_sequence:
                    save_data(sequence, label)
                    sequence = []
                    recording = False
                    break

            cv2.putText(frame, "Click 'r' to start recording", (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Record Gesture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                for i in range(wait_time, 0, -1):
                    ret, frame = cap.read()
                    cv2.putText(frame, f"Starting in {i}", (100, 250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.imshow("Record Gesture", frame)
                    cv2.waitKey(1000)  
                recording = not recording
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cap.release()
    cv2.destroyAllWindows()