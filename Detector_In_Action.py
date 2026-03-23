import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ==============================
# Load Model
# ==============================
model = tf.keras.models.load_model("emotion_detection_model.h5")

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

suggestions = {
    "Angry": "Take a deep breath.",
    "Disgusted": "Shift focus.",
    "Fearful": "Relax, you're safe.",
    "Happy": "Keep smiling 😊",
    "Neutral": "Calm and steady.",
    "Sad": "Take a pause ❤️",
    "Surprised": "Unexpected moment!"
}

# ==============================
# Buffers
# ==============================
emotion_history = deque(maxlen=7)
right_wrist_x_history = deque(maxlen=10)

action_history = deque(maxlen=10)
gesture_history = deque(maxlen=10)
emotion_seq = deque(maxlen=10)

# ==============================
# MediaPipe Setup
# ==============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

# ==============================
# Face Detection
# ==============================
facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ==============================
# Logic Functions
# ==============================
def get_intent(emotion, gesture, action):
    if emotion == "Happy" and gesture == "Waving":
        return "Greeting"
    if emotion == "Happy" and gesture == "Clapping":
        return "Appreciation"
    if emotion == "Sad":
        return "Low mood"
    return "Neutral"

def predict_next(action_hist, gesture_hist, emotion_hist):
    if gesture_hist.count("Waving") > 5:
        return "Will continue interaction"
    if gesture_hist.count("Clapping") > 3:
        return "Showing appreciation"
    if emotion_hist.count("Sad") > 5:
        return "Likely disengaged"
    return "No strong prediction"

def detect_hand(lm):
    fingers = [
        1 if lm[4].x > lm[3].x else 0,
        1 if lm[8].y < lm[6].y else 0,
        1 if lm[12].y < lm[10].y else 0,
        1 if lm[16].y < lm[14].y else 0,
        1 if lm[20].y < lm[18].y else 0,
    ]
    if fingers == [1,0,0,0,0]: return "Thumbs Up"
    if fingers == [0,1,1,0,0]: return "Peace"
    if fingers == [0,0,0,0,0]: return "Fist"
    if fingers == [1,1,1,1,1]: return "Open Palm"
    return "Unknown"

def panel(img,x1,y1,x2,y2):
    overlay = img.copy()
    cv2.rectangle(overlay,(x1,y1),(x2,y2),(30,30,30),-1)
    cv2.addWeighted(overlay,0.4,img,0.6,0,img)

# ==============================
# Webcam
# ==============================
cap = cv2.VideoCapture(0)

last_emotion, last_action, last_gesture, last_hand = "Detecting...", "Idle", "None", "None"

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==============================
    # EMOTION (ACCURATE FIXED)
    # ==============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, 1.3, 5)

    emotion = last_emotion

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48)) / 255.0
        roi = np.reshape(roi, (1,48,48,1))

        pred = model.predict(roi, verbose=0)
        confidence = np.max(pred)

        new_emotion = emotion_dict[np.argmax(pred)]

        if confidence > 0.5:
            emotion_history.append(new_emotion)

        if len(emotion_history) > 0:
            emotion = max(set(emotion_history), key=emotion_history.count)

    # ==============================
    # POSE
    # ==============================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    action = last_action
    gesture = "None"

    if res.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,150), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(200,200,200), thickness=1)
        )

        lm = res.pose_landmarks.landmark
        ls, rs = lm[11], lm[12]
        lw, rw = lm[15], lm[16]

        if rw.y < rs.y: action = "Right Hand"
        elif lw.y < ls.y: action = "Left Hand"
        elif lw.y < ls.y and rw.y < rs.y: action = "Both Hands"
        else: action = "Idle"

        right_wrist_x_history.append(rw.x)

        # Priority-based gesture logic
        if abs(lw.x - rw.x) < 0.05:
            gesture = "Clapping"

        elif rw.x > rs.x + 0.2:
            gesture = "Pointing Right"

        elif len(right_wrist_x_history) == 10:
            if max(right_wrist_x_history) - min(right_wrist_x_history) > 0.15:
                gesture = "Waving"

    # ==============================
    # HANDS
    # ==============================
    hand = last_hand

    hand_res = hands.process(rgb)
    if hand_res.multi_hand_landmarks:
        for hl in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hl,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,100,255), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(200,200,200), thickness=1)
            )
            hand = detect_hand(hl.landmark)

    # ==============================
    # HISTORY + PREDICTION
    # ==============================
    action_history.append(action)
    gesture_history.append(gesture)
    emotion_seq.append(emotion)

    intent = get_intent(emotion, gesture, action)
    next_move = predict_next(action_history, gesture_history, emotion_seq)

    last_emotion, last_action, last_gesture, last_hand = emotion, action, gesture, hand

    # ==============================
    # UI (CLEAN)
    # ==============================
    display = cv2.resize(frame, (1280,720))

    # Header
    panel(display,0,0,1280,70)
    cv2.putText(display,"Human Behavior AI System",(30,45),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

    # Left
    panel(display,0,80,320,200)
    cv2.putText(display,f"Emotion: {emotion}",(20,140),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    # Center
    panel(display,320,80,900,200)
    cv2.putText(display,f"Intent: {intent}",(350,140),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,150),2)

    # Right
    panel(display,900,80,1280,250)
    cv2.putText(display,f"Action: {action}",(920,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,255),2)
    cv2.putText(display,f"Gesture: {gesture}",(920,170),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,200,0),2)
    cv2.putText(display,f"Hand: {hand}",(920,210),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,100,255),2)

    # Bottom (NOT covering face)
    panel(display,0,650,1280,720)
    cv2.putText(display,f"Next: {next_move}",(30,680),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(display,suggestions.get(emotion,""),
                (30,710),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,150),2)

    cv2.imshow("AI System", display)

    # ==============================
    # EXIT FIX
    # ==============================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()