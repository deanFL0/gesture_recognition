# Gesture Recognition using Mediapipe & TensorFlow
Camera-based hand and body gesture recognition using MediaPipe Holistic and TensorFlow models.

## Installation
Clone the repo.

```
git clone https://github.com/deanFL0/gesture_recognition.git
```

With python 3.12+ and pip install the requirement.

```
pip install requirement.txt
```

## How to run
### Recording gesture for train data

* Run the python file with command below.

```
python run record_gesture.py
```

* Wait for prompt and input the name of the gesture that will be recorded.
* Wait for the webcam application to open and then hit 'r' to start 3 second countdown, the recording will begin after the countdown.
* To change length of landmark sequence to be recorded and wait time edit line 23 and 25 in 'record_gesture.py'
```
# Set Max Sequence to record
max_sequence = 30
# Set wait tim (second)
wait_time = 3
```

### Run the gesture recognation application
* Run the python file with command below.
```
python run main.py
```
* Press 'esc' to exit.
  
### Retrain classifier model
* Open 'model_train.ipynb'
* Change number of output class of the model as needed.
```
model = keras.Sequential([
    Input(shape=(30, 1629)),
    LSTM(64, return_sequences=True, activation='relu'),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax') # Change the number of output as needed
])
```
* Run the notebook.
