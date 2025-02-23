import cv2
import mediapipe as mp
import random

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Exercise prompts
exercises = [
    "Raise your hands!", 
    "Stretch your arms wide!", 
    "Do a squat!", 
    "Touch your toes!"
]
current_exercise = random.choice(exercises)
score = 0

# Start webcam
cap = cv2.VideoCapture(0)

def detect_pose(results):
    """Detects user pose based on landmarks and returns the detected exercise."""
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    # Coordinates
    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y

    # Hand raise detection (both hands above shoulders)
    if left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
        return "Raise your hands!"

    # Stretch detection (hands far apart)
    left_hand_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
    right_hand_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
    if abs(left_hand_x - right_hand_x) > 0.8:
        return "Stretch your arms wide!"

    # Squat detection (hips below knees)
    if left_hip_y > left_knee_y:
        return "Do a squat!"

    # Touch toes detection (hands near ankles)
    if left_wrist_y > left_ankle_y:
        return "Touch your toes!"

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    result = pose.process(frame_rgb)

    # Draw landmarks on frame
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame_bgr, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display exercise prompt and score with background
    draw_text_with_background(frame_bgr, f"Exercise: {current_exercise}", (50, 50),
                               background_color=(0, 255, 0))
    draw_text_with_background(frame_bgr, f"Score: {score}", (50, 100), 
                               background_color=(0, 0, 255))

    # Detect and match pose to current exercise
    detected_pose = detect_pose(result)
    if detected_pose == current_exercise:
        score += 1
        current_exercise = random.choice(exercises)  # Change exercise

    # Show the frame
    cv2.imshow("PhysioPlay Game", frame_bgr)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
