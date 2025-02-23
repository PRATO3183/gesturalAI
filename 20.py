# woring most accurately 

import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Holistic for better pose, hand, and face tracking
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils

# Exercise prompts
exercises = [
    "Raise your hands!", 
    "Stretch your arms wide!", 
    "Do a squat!", 
    "Touch your toes!"
]

def detect_pose(results):
    """Detects user pose and returns the matching exercise."""
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    # Extract required coordinates
    left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
    left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
    left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]

    # Hand raise detection (both hands above shoulders)
    if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        return "Raise your hands!"

    # Arm stretch detection (hands far apart)
    if abs(left_wrist.x - right_wrist.x) > 0.7:
        return "Stretch your arms wide!"

    # Squat detection (hips below knees)
    if left_hip.y > left_knee.y:
        return "Do a squat!"

    # Touch toes detection (hands near ankles)
    if left_wrist.y > left_ankle.y:
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

        # Display exercise prompt, score, and timer with background
        draw_text_with_background(frame_bgr, f"Exercise: {current_exercise}", (50, 50),
                                   background_color=(0, 255, 0))
        draw_text_with_background(frame_bgr, f"Score: {score}", (50, 100), 
                                   background_color=(0, 0, 255))
        draw_text_with_background(frame_bgr, f"Time Left: {int(remaining_time)}s", (50, 150), 
                                   background_color=(255, 165, 0))

        # Detect and match pose to current exercise
        detected_pose = detect_pose(result)
        if detected_pose == current_exercise:
            score += 1
            current_exercise = random.choice(exercises)  # Change exercise

        # Show the frame
        cv2.imshow("PhysioPlay Game", frame_bgr)

        # End the game when time is up or user presses 'q'
        if remaining_time <= 0 or (cv2.waitKey(10) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Display final score after the game ends
    print(f"\nGame Over! Final Score: {score}")
    play_again = input("Play again? (y/n): ").strip().lower()
    if play_again != 'y':
        break
