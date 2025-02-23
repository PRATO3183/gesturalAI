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
    "Raise your hands!", "Stretch your arms wide!", "Do a squat!", "Touch your toes!",
    "Left arm forward!", "Right arm forward!", "Clap your hands!", "Make a T with your arms!",
    "Bend your left elbow!", "Bend your right elbow!", "Raise your left leg!", "Raise your right leg!"
]

# Pose history for smoothing (deque for efficient pop operations)
pose_history = deque(maxlen=5)

def smooth_pose(value):
    pose_history.append(value)
    return np.mean(pose_history) > 0.7  # Requires stable pose detection over multiple frames

def detect_pose(results):
    """Detects user pose and returns the matching exercise."""
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    get_coord = lambda part: (landmarks[part].x, landmarks[part].y)  # Helper function
    
    # Extract coordinates
    lw_x, lw_y = get_coord(mp_holistic.PoseLandmark.LEFT_WRIST)
    rw_x, rw_y = get_coord(mp_holistic.PoseLandmark.RIGHT_WRIST)
    ls_x, ls_y = get_coord(mp_holistic.PoseLandmark.LEFT_SHOULDER)
    rs_x, rs_y = get_coord(mp_holistic.PoseLandmark.RIGHT_SHOULDER)
    le_x, le_y = get_coord(mp_holistic.PoseLandmark.LEFT_ELBOW)
    re_x, re_y = get_coord(mp_holistic.PoseLandmark.RIGHT_ELBOW)
    lh_x, lh_y = get_coord(mp_holistic.PoseLandmark.LEFT_HIP)
    rh_x, rh_y = get_coord(mp_holistic.PoseLandmark.RIGHT_HIP)
    lk_x, lk_y = get_coord(mp_holistic.PoseLandmark.LEFT_KNEE)
    rk_x, rk_y = get_coord(mp_holistic.PoseLandmark.RIGHT_KNEE)
    la_x, la_y = get_coord(mp_holistic.PoseLandmark.LEFT_ANKLE)
    ra_x, ra_y = get_coord(mp_holistic.PoseLandmark.RIGHT_ANKLE)

    shoulder_width = abs(ls_x - rs_x)

    if lw_y < ls_y - 0.1 and rw_y < rs_y - 0.1:
        return "Raise your hands!"
    if abs(lw_x - rw_x) > shoulder_width * 2:
        return "Stretch your arms wide!"
    if lh_y > lk_y and rh_y > rk_y:
        return "Do a squat!"
    if lw_y > la_y - 0.05 and rw_y > ra_y - 0.05:
        return "Touch your toes!"
    if ls_y < le_y < lw_y and rs_y > re_y:
        return "Left arm forward!"
    if rs_y < re_y < rw_y and ls_y > le_y:
        return "Right arm forward!"
    if abs(lw_x - rw_x) < 0.1 and lw_y < ls_y and rw_y < rs_y:
        return "Clap your hands!"
    if abs(lw_x - rw_x) > shoulder_width * 1.5 and abs(ls_y - rs_y) < 0.05:
        return "Make a T with your arms!"
    if le_y > ls_y and le_y > lw_y and re_y > rs_y and re_y > rw_y:
        return "Bend your elbows!"
    if lk_y < lh_y and la_y < lk_y and rk_y > rh_y:
        return "Raise your left leg!"
    if rk_y < rh_y and ra_y < rk_y and lk_y > lh_y:
        return "Raise your right leg!"

    return None

def draw_text(frame, text, position, bg_color, font_color=(255, 255, 255)):
    """Draws text with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 1, 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    score = 0
    game_duration = 180  # 3 minutes
    start_time = time.time()
    current_exercise = random.choice(exercises)

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

        draw_text(frame_bgr, f"Exercise: {current_exercise}", (50, 50), (0, 255, 0))
        draw_text(frame_bgr, f"Score: {score}", (50, 100), (0, 0, 255))
        draw_text(frame_bgr, f"Time Left: {int(remaining_time)}s", (50, 150), (255, 165, 0))

        detected_pose = detect_pose(result)
        if detected_pose == current_exercise:
            if smooth_pose(1):
                score += 1
                current_exercise = random.choice(exercises)
        else:
            smooth_pose(0)

        cv2.imshow("PhysioPlay Game", frame_bgr)
        if remaining_time <= 0 or (cv2.waitKey(10) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nGame Over! Final Score: {score}")
    
    if input("Play again? (y/n): ").strip().lower() == 'y':
        main()

if __name__ == "__main__":
    main()