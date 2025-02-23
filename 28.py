import cv2
import mediapipe as mp
import random
import time
import numpy as np
from collections import deque

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils

# Exercise prompts
exercises = [
    "Raise your hands!",
    "Stretch your arms wide!",
    "Do a squat!",
    "Touch your toes!",
    "Left arm forward!",
    "Right arm forward!",
    "Clap your hands!",
    "Make a T with your arms!",
    "Bend your left elbow!",
    "Bend your right elbow!",
    "Raise your left leg!",
    "Raise your right leg!",
]

# Moving average for pose smoothing
pose_history = deque(maxlen=5)

def smooth_pose(new_pose):
    """Apply a moving average filter to smooth pose detection."""
    pose_history.append(new_pose)
    return np.mean(pose_history)


def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def detect_pose(results):
    """Detects user pose and returns the matching exercise."""
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    # Extract key points
    left_wrist, right_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
    left_shoulder, right_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    left_elbow, right_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
    left_hip, right_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP], landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
    left_knee, right_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE], landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
    left_ankle, right_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE], landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    
    # Hand raise detection
    if left_wrist.y < left_shoulder.y - 0.05 and right_wrist.y < right_shoulder.y - 0.05:
        return "Raise your hands!"

    # T-Pose detection
    if abs(left_wrist.x - right_wrist.x) > shoulder_width * 1.9 and abs(left_shoulder.y - right_shoulder.y) < shoulder_width * 0.2:
        return "Make a T with your arms!"

    # Clap hands detection (refined for better accuracy)
    if abs(left_wrist.x - right_wrist.x) < shoulder_width * 0.15 and abs(left_wrist.y - right_wrist.y) < shoulder_width * 0.15:
        return "Clap your hands!"

    # Bend elbow detection
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if left_elbow_angle < 60:
        return "Bend your left elbow!"
    if right_elbow_angle < 60:
        return "Bend your right elbow!"

    # Leg raise detection
    if left_knee.y < left_hip.y - 0.05 and left_ankle.y < left_knee.y:
        return "Raise your left leg!"
    if right_knee.y < right_hip.y - 0.05 and right_ankle.y < right_knee.y:
        return "Raise your right leg!"
    
    return None


# Start the game
cap = cv2.VideoCapture(0)
current_exercise = random.choice(exercises)
score = 0
game_duration = 3 * 60  # 3 minutes
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame_bgr, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
    elapsed_time = time.time() - start_time
    remaining_time = max(0, game_duration - elapsed_time)
    
    detected_pose = detect_pose(result)
    if detected_pose:
        smoothed_pose = smooth_pose(1 if detected_pose == current_exercise else 0)
        if smoothed_pose > 0.7:
            score += 1
            current_exercise = random.choice(exercises)
    
    cv2.imshow("Exercise Game", frame_bgr)
    if remaining_time <= 0 or (cv2.waitKey(10) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()