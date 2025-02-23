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

# Exercise prompts (expanded)
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


def detect_pose(results):
    """Detects user pose and returns the matching exercise."""
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark  # Ensure landmarks exist

    # Extract key points
    left_wrist, right_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST], landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
    left_shoulder, right_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER], landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    left_elbow, right_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW], landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
    left_hip, right_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP], landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
    left_knee, right_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE], landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
    left_ankle, right_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE], landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]

    # Define reference shoulder width
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)

    # Hand raise detection
    if left_wrist.y < left_shoulder.y - 0.05 and right_wrist.y < right_shoulder.y - 0.05:
        return "Raise your hands!"

    # Arm stretch detection (T-Pose) - Improved threshold
    if abs(left_wrist.x - right_wrist.x) > shoulder_width * 1.8 and abs(left_shoulder.y - right_shoulder.y) < shoulder_width * 0.2:
        return "Make a T with your arms!"

    # Squat detection
    if left_hip.y > left_knee.y and right_hip.y > right_knee.y:
        return "Do a squat!"

    # Touch toes detection
    if left_wrist.y > left_ankle.y - 0.05 and right_wrist.y > right_ankle.y - 0.05:
        return "Touch your toes!"

    # Left arm forward
    if left_shoulder.y < left_elbow.y and left_elbow.y < left_wrist.y and right_shoulder.y > right_elbow.y:
        return "Left arm forward!"

    # Right arm forward
    if right_shoulder.y < right_elbow.y and right_elbow.y < right_wrist.y and left_shoulder.y > left_elbow.y:
        return "Right arm forward!"

    # Clap hands - Improved accuracy
    if abs(left_wrist.x - right_wrist.x) < shoulder_width * 0.2 and abs(left_wrist.y - right_wrist.y) < shoulder_width * 0.2:
        return "Clap your hands!"

    # Bend left elbow
    if left_elbow.y > left_shoulder.y and left_elbow.y > left_wrist.y:
        return "Bend your left elbow!"

    # Bend right elbow
    if right_elbow.y > right_shoulder.y and right_elbow.y > right_wrist.y:
        return "Bend your right elbow!"

    # Raise left leg
    if left_knee.y < left_hip.y and left_ankle.y < left_knee.y and right_knee.y > right_hip.y:
        return "Raise your left leg!"

    # Raise right leg
    if right_knee.y < right_hip.y and right_ankle.y < right_knee.y and left_knee.y > left_hip.y:
        return "Raise your right leg!"

    return None


def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale=1, font_color=(255, 255, 255), 
                              background_color=(0, 0, 0), thickness=2, padding=5):
    """Draws text with a colored background."""
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x - padding, y - text_size[1] - padding),
                  (x + text_size[0] + padding, y + padding), background_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness)


# Start the game
while True:
    cap = cv2.VideoCapture(0)
    current_exercise = random.choice(exercises)
    score = 0

    # Game timer setup
    game_duration = 3 * 60  # 3 minutes in seconds
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Holistic detection
        result = holistic.process(frame_rgb)

        # Draw landmarks on frame
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame_bgr, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Calculate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = max(0, game_duration - elapsed_time)

        # Display exercise prompt, score, and timer
        draw_text_with_background(frame_bgr, f"Exercise: {current_exercise}", (50, 50), background_color=(0, 255, 0))
        draw_text_with_background(frame_bgr, f"Score: {score}", (50, 100), background_color=(0, 0, 255))
        draw_text_with_background(frame_bgr, f"Time Left: {int(remaining_time)}s", (50, 150), background_color=(255, 165, 0))

        # Detect and match pose to current exercise
        detected_pose = detect_pose(result)
        if detected_pose:
            smoothed_pose = smooth_pose(1 if detected_pose == current_exercise else 0)
            if smoothed_pose > 0.7:
                score += 1
                current_exercise = random.choice(exercises)

        # Show the frame
        # cv2.namedWindow("PhysioPlay Game",cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("PhysioPlay Game",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow("PhysioPlay Game",cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty("PhysioPlay Game",cv2.WINDOW_FREERATIO,cv2.WND_PROP_ASPECT_RATIO)
        cv2.imshow("PhysioPlay Game", frame_bgr)

        # End the game when time is up or user presses 'q'
        if remaining_time <= 0 or (cv2.waitKey(10) & 0xFF == ord('q')):
            quit=1
            break

    cap.release()
    cv2.destroyAllWindows()
    if quit==1:
        break

    # if input("Play again? (y/n): ").strip().lower() != 'y':
    #     break