import cv2
import mediapipe as mp
import random
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Exercises and corresponding emoji hints
exercises = {
    "Raise your hands!": "🙌",
    "Stretch your arms wide!": "👐",
}
current_exercise = random.choice(list(exercises.keys()))
score = 0

# Start webcam
cap = cv2.VideoCapture(0)

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

    detected_pose = None
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Hand raise detection (wrists above shoulders)
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        if left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
            detected_pose = "Raise your hands!"

        # Arms stretch detection (hands far apart)
        left_hand_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
        right_hand_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
        if abs(left_hand_x - right_hand_x) > 0.8:
            detected_pose = "Stretch your arms wide!"

    # Display exercise prompt and feedback
    cv2.putText(frame_bgr, f"Do this: {current_exercise} {exercises[current_exercise]}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f"Score: {score}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if the detected pose matches the exercise
    if detected_pose == current_exercise:
        score += 1
        current_exercise = random.choice(list(exercises.keys()))  # Change exercise

    # Emoji hint (convert emoji text to image placeholder)
    emoji_hint = exercises[current_exercise]
    emoji_img = np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder emoji effect
    cv2.putText(emoji_img, emoji_hint, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    # Show emoji hint next to the feed
    frame_bgr[50:150, 600:700] = cv2.resize(emoji_img, (100, 100))

    # Show the frame
    cv2.imshow("PhysioPlay Game", frame_bgr)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
