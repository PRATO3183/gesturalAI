import cv2
import mediapipe as mp
import random
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Initialize Mediapipe Holistic for better tracking
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Exercise prompts
exercises = [
    "Raise your left hand!", 
    "Raise your right hand!", 
    "Stretch both arms wide!", 
    "Do a squat!", 
    "Touch your toes!", 
    "Jump in place!"
]

# Initialize score and exercise tracker
score = 0
current_exercise = ""
game_duration = 180  # 3 minutes
cap = None

# Tkinter GUI setup
window = tk.Tk()
window.title("PhysioPlay Game")

# Frame to display the video feed
video_frame = tk.Label(window)
video_frame.pack()

# Display text instructions and score
instruction_label = tk.Label(window, text="", font=("Helvetica", 16), fg="green")
instruction_label.pack()
score_label = tk.Label(window, text="Score: 0", font=("Helvetica", 14))
score_label.pack()
timer_label = tk.Label(window, text="Time Left: 180s", font=("Helvetica", 14))
timer_label.pack()

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
        return "Raise your left hand!"
    if right_wrist.y < right_shoulder.y and left_wrist.y > left_shoulder.y:
        return "Raise your right hand!"

    # Arm stretch detection
    if abs(left_wrist.x - right_wrist.x) > 0.7:
        return "Stretch both arms wide!"

    # Squat detection
    if left_hip.y > left_knee.y:
        return "Do a squat!"

    # Touch toes detection
    if left_wrist.y > left_ankle.y:
        return "Touch your toes!"

    # Jump detection (rough estimate based on vertical wrist movement)
    if abs(left_wrist.y - left_ankle.y) < 0.1:
        return "Jump in place!"

    return None

# Start game logic
def start_game():
    global score, current_exercise, start_time, cap

    score = 0
    current_exercise = random.choice(exercises)
    start_time = time.time()

    instruction_label.config(text=f"Exercise: {current_exercise}")

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access the camera!")
        return

    update_frame()

# Function to update video frames and game logic
def update_frame():
    global score, current_exercise, cap

    if not cap or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        return

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    with mp_holistic.Holistic() as holistic:
        result = holistic.process(frame_rgb)

        # Draw landmarks
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Detect exercise pose
        detected_pose = detect_pose(result)
        if detected_pose == current_exercise:
            score += 1
            current_exercise = random.choice(exercises)
            instruction_label.config(text=f"Exercise: {current_exercise}")
            score_label.config(text=f"Score: {score}")

        # Update timer
        elapsed_time = time.time() - start_time
        remaining_time = max(0, game_duration - elapsed_time)
        timer_label.config(text=f"Time Left: {int(remaining_time)}s")

        # Convert frame to image for Tkinter
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.imgtk = imgtk
        video_frame.configure(image=imgtk)

        # End the game when timer reaches zero
        if remaining_time <= 0:
            end_game()
        else:
            window.after(10, update_frame)

# End game function
def end_game():
    global score, cap
    if cap:
        cap.release()

    messagebox.showinfo("Game Over", f"Final Score: {score}")
    reset_game()

# Reset game to initial state
def reset_game():
    instruction_label.config(text="")
    score_label.config(text="Score: 0")
    timer_label.config(text="Time Left: 180s")

# Start button
start_button = tk.Button(window, text="Start Game", command=start_game, font=("Helvetica", 14), bg="blue", fg="white")
start_button.pack()

# Run the GUI event loop
window.mainloop()
