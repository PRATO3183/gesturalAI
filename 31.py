import cv2
import mediapipe as mp
import random
import time
import numpy as np
import os

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

# Function to save and display scoreboard
SCOREBOARD_FILE = "scoreboard.txt"

def save_score(name, score):
    scores = []
    
    # Read existing scores
    if os.path.exists(SCOREBOARD_FILE):
        with open(SCOREBOARD_FILE, "r") as file:
            for line in file:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    scores.append((parts[0], int(parts[1])))

    # Add new score
    scores.append((name, score))
    
    # Sort scores in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Keep only top 5 scores
    scores = scores[:5]

    # Save back to file
    with open(SCOREBOARD_FILE, "w") as file:
        for entry in scores:
            file.write(f"{entry[0]}:{entry[1]}\n")

    return scores

def display_scoreboard(scores):
    scoreboard = np.zeros((400, 500, 3), dtype=np.uint8)  # Create a black image for scoreboard
    cv2.putText(scoreboard, "🏆 High Scores 🏆", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    for i, (player, score) in enumerate(scores):
        cv2.putText(scoreboard, f"{i+1}. {player} - {score} pts", (100, 100 + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Scoreboard", scoreboard)
    cv2.waitKey(5000)  # Display for 5 seconds
    cv2.destroyWindow("Scoreboard")

# Get player's name before starting the game
player_name = input("Enter your name: ")

# Initialize video capture
cap = cv2.VideoCapture(0)
current_exercise = random.choice(exercises)
score = 0
game_duration = 180  # 3 minutes
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
    
    # Display game info
    cv2.putText(frame_bgr, f"Player: {player_name}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f"Exercise: {current_exercise}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"Score: {score}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame_bgr, f"Time Left: {int(remaining_time)}s", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Detect pose (if exercise is completed)
    detected_pose = None  # Replace with pose detection logic if needed
    if detected_pose and detected_pose == current_exercise:
        score += 1
        current_exercise = random.choice(exercises)

    # Allow skipping exercises
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # Quit the game
        break
    elif key == ord('s') or key == 83:  # Right arrow key (83) or "s" key
        score += 1  # Add a point when skipping
        current_exercise = random.choice(exercises)

    # Show window
    cv2.imshow("PhysioPlay Game", frame_bgr)
    
    # End game when time is up
    if remaining_time <= 0:
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Save and display the final scoreboard in a separate window
top_scores = save_score(player_name, score)
display_scoreboard(top_scores)