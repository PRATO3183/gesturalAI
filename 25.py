import cv2
import mediapipe as mp
import random
import time
import numpy as np

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
    "Left arm forward!",  # New
    "Right arm forward!", # New
    "Clap your hands!",  # New
    "Make a T with your arms!", # New
    "Bend your left elbow!", # New
    "Bend your right elbow!",# New
    "Raise your left leg!", # New (might be challenging for some)
    "Raise your right leg!",# New (might be challenging for some)

]


# Moving average for pose smoothing
pose_history = []

def smooth_pose(new_pose, history, window_size=5):
    """Apply a moving average filter to smooth pose detection."""
    history.append(new_pose)
    if len(history) > window_size:
        history.pop(0)
    return np.mean(history)


def detect_pose(results):
    """Detects user pose and returns the matching exercise."""
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark  # Make sure landmarks is defined here

    left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW]
    left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]

    # ... (Existing code for hand raise, arm stretch, squat, touch toes)
    # Hand raise detection (both hands comfortably above shoulders)
    if left_wrist.y < left_shoulder.y - 0.05 and right_wrist.y < right_shoulder.y - 0.05:
        return "Raise your hands!"

    # Arm stretch detection (hands significantly far apart, using shoulder width as reference)
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if abs(left_wrist.x - right_wrist.x) > shoulder_width * 2:
        return "Stretch your arms wide!"

    # Squat detection (both hips below both knees)
    if left_hip.y > left_knee.y and right_hip.y > right_knee.y:
        return "Do a squat!"

    # Touch toes detection (both hands near ankles)
    if left_wrist.y > left_ankle.y - 0.05 and right_wrist.y > right_ankle.y - 0.05:
        return "Touch your toes!"

    # ... (New code for arms forward, clap hands, make T, bend elbows)
    # Left/Right arm forward
    if left_shoulder.y < left_elbow.y and left_elbow.y < left_wrist.y and right_shoulder.y > right_elbow.y :  # Left arm forward
        return "Left arm forward!"
    if right_shoulder.y < right_elbow.y and right_elbow.y < right_wrist.y and left_shoulder.y > left_elbow.y: # Right arm forward
        return "Right arm forward!"

    # Clap hands
    if abs(left_wrist.x - right_wrist.x) < 0.1 and left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        return "Clap your hands!"

    # Make a T
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if abs(left_wrist.x - right_wrist.x) > shoulder_width * 1.5 and abs(left_shoulder.y - right_shoulder.y) < shoulder_width * 0.5: # Adjust threshold as needed
        return "Make a T with your arms!"

    #Bend Elbows
    if left_elbow.y > left_shoulder.y and left_elbow.y > left_wrist.y and right_elbow.y > right_shoulder.y and right_elbow.y > right_wrist.y:
        return "Bend your elbows!"

    #Leg Raises (more complex, might need tuning)
    if left_knee.y < left_hip.y and left_ankle.y < left_knee.y and right_knee.y > right_hip.y: # Left leg raise
        return "Raise your left leg!"
    if right_knee.y < right_hip.y and right_ankle.y < right_knee.y and left_knee.y > left_hip.y: # Right leg raise
        return "Raise your right leg!"

    return None

# ... (rest of the code - game loop, etc.)
def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale=1, font_color=(255, 255, 255), 
                              background_color=(0, 0, 0), thickness=2, padding=5):
    """Draws text with a colored background."""
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x - padding, y - text_size[1] - padding),
                  (x + text_size[0] + padding, y + padding), background_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness)


def draw_thick_landmarks(frame, results, connections, thickness=5, circle_radius=10, landmark_color=(0, 255, 0), connection_color=(0, 0, 255)):
    """Draws landmarks and connections with increased thickness."""
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), circle_radius, landmark_color, -1)  # Filled circle

        for connection in connections:
            start_landmark = results.pose_landmarks.landmark[connection[0]]
            end_landmark = results.pose_landmarks.landmark[connection[1]]

            x1 = int(start_landmark.x * frame.shape[1])
            y1 = int(start_landmark.y * frame.shape[0])
            x2 = int(end_landmark.x * frame.shape[1])
            y2 = int(end_landmark.y * frame.shape[0])

            cv2.line(frame, (x1, y1), (x2, y2), connection_color, thickness)



# ... (rest of your functions: smooth_pose, detect_pose, draw_text_with_background)

# Game loop
while True:
    # ... (video capture and frame processing)

    # Holistic detection
    result = holistic.process(frame_rgb)

    # Draw landmarks on frame - Use the new function
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if result.pose_landmarks:
        draw_thick_landmarks(frame_bgr, result, mp_holistic.POSE_CONNECTIONS) # Call the function


    # ... (rest of your game logic)

    cv2.imshow("PhysioPlay Game", frame_bgr) # Make sure you are using frame_bgr here

    # ... (end of game loop)