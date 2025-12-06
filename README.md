# Real-Time Emotion Detection using FER-2013

This project trains a **Convolutional Neural Network (CNN)** on the **FER-2013** facial expression dataset, and uses the trained model for **real-time emotion detection** from a webcam feed.

## 1. Dataset

We use the **FER-2013** dataset (48×48 grayscale face images, 7 emotion classes).

- Download it from Kaggle: search for **"FER-2013 facial expression dataset"**.
- After downloading, extract `fer2013.csv`.
- Place `fer2013.csv` in the **project root folder**, like:

  ```text
  Emotion-Detection-FER2013/
      fer2013.csv
      train_emotion_cnn.py
      realtime_emotion.py

## 2. Project Structure
Emotion-Detection-FER2013/
├── fer2013.csv                # (user-provided, ignored by git)
├── train_emotion_cnn.py       # script to train the CNN model
├── realtime_emotion.py        # real-time webcam emotion detection
├── emotion_detection_model.h5 # saved trained model (optional, usually ignored)
├── requirements.txt
└── README.md

## 3. Setup Instructions
  3.1 Create and activate virtual environment (optional but recommended)
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
  3.2 Install dependencies
    pip install -r requirements.txt
  Or manually:
    pip install numpy pandas matplotlib tensorflow opencv-python

## 4. Training the Model
Make sure fer2013.csv is in the project root.
Run the training script (name may differ based on your file):
  python train_emotion_cnn.py

What this script does:
- Loads fer2013.csv
- Converts pixel strings to 48×48×1 grayscale images
- Normalizes pixel values to [0, 1]
- Splits data into train/test sets
- Builds a CNN with Conv2D, MaxPooling, Dropout, Dense, Softmax
- Trains for a fixed number of epochs
- Plots training/validation accuracy and loss
- Saves the trained model as emotion_detection_model.h5

## 5. Evaluating the Model
Inside train_emotion_cnn.py, we evaluate the model on a held-out test split:

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%")

This prints the final test accuracy in percentage.

## 6. Real-Time Emotion Detection
We use OpenCV + Haarcascade + the trained CNN model:
- python realtime_emotion.py
- realtime_emotion.py:
- Opens webcam (cv2.VideoCapture(0))
- Detects faces using haarcascade_frontalface_default.xml
- Crops each face, converts to grayscale, resizes to 48×48
- Normalizes and reshapes input to (1, 48, 48, 1)
- Loads emotion_detection_model.h5 and predicts the emotion
- Draws bounding box + emotion label (Angry, Happy, Sad, etc.) on the video frame
Press q to exit

## 7. Summary

Pipeline:
1. Download FER-2013 → place fer2013.csv in project root
2. Train CNN model on 48×48 grayscale faces → save emotion_detection_model.h5
3. Evaluate accuracy on test set
4. Run realtime_emotion.py for live webcam-based emotion detection
