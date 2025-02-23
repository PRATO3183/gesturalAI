import pygame
import cv2
import mediapipe as mp
import random
import time
import numpy as np

# Warining supression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Initialize Mediapipe Holistic for pose detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Pygame setup
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 30
GAME_DURATION = 180  # 3 minutes
FONT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
FONT = pygame.font.SysFont("Arial", 30)

# Create the display window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("PhysioPlay - Kids Exercise Game")

# Exercise prompts
exercises = [
    "Raise your left hand", 
    "Raise your right hand", 
    "Stretch both arms wide", 
    "Do a squat", 
    "Touch your toes", 
    "Jump in place"
]

# Initialize game variables
score = 0
current_exercise = random.choice(exercises)
start_time = time.time()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    pygame.quit()
    exit()

# Function to detect pose and return matched exercise
def detect_pose(results):
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    # Extract coordinates
    left_wrist = landmarks[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
    left_knee = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
    left_ankle = landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]

    # Hand raise detection
    if left_wrist.y < left_shoulder.y and right_wrist.y > right_shoulder.y:
        return "Raise your left hand"
    if right_wrist.y < right_shoulder.y and left_wrist.y > left_shoulder.y:
        return "Raise your right hand"

    # Arm stretch detection
    if abs(left_wrist.x - right_wrist.x) > 0.7:
        return "Stretch both arms wide"

    # Squat detection
    if left_hip.y > left_knee.y:
        return "Do a squat"

    # Touch toes detection
    if left_wrist.y > left_ankle.y:
        return "Touch your toes"

    # Jump detection (rough estimate based on vertical wrist movement)
    if abs(left_wrist.y - left_ankle.y) < 0.1:
        return "Jump in place"

    return None

# Game loop
running = True
with mp_holistic.Holistic() as holistic:
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert frame to RGB
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe
        result = holistic.process(frame_rgb)

        # Detect exercise pose
        detected_pose = detect_pose(result)
        if detected_pose == current_exercise:
            score += 1
            current_exercise = random.choice(exercises)

        # Draw pose landmarks on the frame
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Calculate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = max(0, GAME_DURATION - elapsed_time)

        # Convert frame to Pygame format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame_surface = pygame.surfarray.make_surface(np.flipud(frame))
        frame_surface = pygame.surfarray.make_surface(frame)

        # Clear the screen and draw elements
        screen.fill(BACKGROUND_COLOR)
        screen.blit(frame_surface, (0, 0))
        exercise_text = FONT.render(f"Exercise: {current_exercise}", True, FONT_COLOR)
        score_text = FONT.render(f"Score: {score}", True, FONT_COLOR)
        timer_text = FONT.render(f"Time Left: {int(remaining_time)}s", True, FONT_COLOR)
        
        screen.blit(exercise_text, (20, 20))
        screen.blit(score_text, (20, 60))
        screen.blit(timer_text, (20, 100))

        # End the game if time is up
        if remaining_time <= 0:
            running = False

        # Update the display
        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

# Display final score
screen.fill(BACKGROUND_COLOR)
final_score_text = FONT.render(f"Game Over! Final Score: {score}", True, FONT_COLOR)
screen.blit(final_score_text, (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2))
pygame.display.flip()

# Wait before closing
pygame.time.wait(5000)

# Release resources
cap.release()
pygame.quit()
