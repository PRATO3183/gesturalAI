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
    "Do a squat!",
    "Touch your toes!",
    "Left arm forward!",
    "Right arm forward!",
    "Clap your hands!",
    "Make a T with your arms straight!",
    "Bend your left elbow!",
    "Bend your right elbow!",
    "Raise your left leg!",
    "Raise your right leg!",
]

# Moving average for pose smoothing
pose_history = deque(maxlen=5)

def smooth_pose(new_pose):
    pose_history.append(new_pose)
    return np.mean(pose_history)

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def detect_pose(results):
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    left_wrist, right_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
    left_shoulder, right_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    left_elbow, right_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
    left_hip, right_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP], landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
    left_knee, right_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE], landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
    left_ankle, right_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE], landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]
    
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)

    # Left arm forward
    if left_shoulder.y < left_elbow.y and left_elbow.y < left_wrist.y and right_shoulder.y > right_elbow.y:
        return "Left arm forward!"

    # Right arm forward
    if right_shoulder.y < right_elbow.y and right_elbow.y < right_wrist.y and left_shoulder.y > left_elbow.y:
        return "Right arm forward!"
    
    if left_wrist.y < left_shoulder.y - 0.05 and right_wrist.y < right_shoulder.y - 0.05:
        return "Raise your hands!"
    
    if abs(left_wrist.x - right_wrist.x) > shoulder_width * 1.8 and abs(left_shoulder.y - right_shoulder.y) < shoulder_width * 0.2:
        return "Make a T with your arms!"
    
    if left_hip.y > left_knee.y and right_hip.y > right_knee.y:
        return "Do a squat!"
    
    if left_wrist.y > left_ankle.y - 0.05 and right_wrist.y > right_ankle.y - 0.05:
        return "Touch your toes!"
    
    if abs(left_wrist.x - right_wrist.x) < shoulder_width * 0.1 and abs(left_wrist.y - right_wrist.y) < shoulder_width * 0.1:
        return "Clap your hands!"
    
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    if 60 < left_elbow_angle < 120:
        return "Bend your left elbow!"
    
    if 60 < right_elbow_angle < 120:
        return "Bend your right elbow!"
    
    if left_knee.y < left_hip.y:
        return "Raise your left leg!"
    
    if right_knee.y < right_hip.y:
        return "Raise your right leg!"
    
    return None

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale=1, font_color=(255, 255, 255), 
                              background_color=(0, 0, 0), thickness=2, padding=5):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x - padding, y - text_size[1] - padding),
                  (x + text_size[0] + padding, y + padding), background_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness)

cap = cv2.VideoCapture(0)
current_exercise = random.choice(exercises)
score = 0
game_duration = 180
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
    draw_text_with_background(frame_bgr, f"Exercise: {current_exercise}", (50, 50), background_color=(0, 255, 0))
    draw_text_with_background(frame_bgr, f"Score: {score}", (50, 100), background_color=(0, 0, 255))
    draw_text_with_background(frame_bgr, f"Time Left: {int(remaining_time)}s", (50, 150), background_color=(255, 165, 0))
    
    detected_pose = detect_pose(result)
    if detected_pose and detected_pose == current_exercise:
        score += 1
        current_exercise = random.choice(exercises)

    cv2.namedWindow("PhysioPlay Game",cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty("PhysioPlay Game",cv2.WINDOW_FREERATIO,cv2.WND_PROP_ASPECT_RATIO)
    cv2.imshow("PhysioPlay Game", frame_bgr)
    if remaining_time <= 0 or (cv2.waitKey(10) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
