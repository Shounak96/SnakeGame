# Virtual Snake Game — Hand Gesture Controlled

A real-time Snake game controlled using **hand gestures** captured through a webcam. The game tracks the player’s **index finger** to control the snake’s movement using swipe gestures. Built using OpenCV and CVZone, the project demonstrates computer vision–based interaction, real-time rendering, and game logic implementation.

---

## Features Implemented

- Hand gesture–based snake control using index finger tracking
- Swipe detection for changing direction (up, down, left, right)
- Real-time gameplay using webcam input
- Dynamic food generation with image overlay
- Obstacle generation with increasing difficulty
- Boundary, self, and obstacle collision detection
- Score and high-score tracking
- Game states: start screen, pause, restart, and game over
- Fullscreen game rendering

---

## Technologies Used

- **Python**
- **OpenCV** – video capture, rendering, image processing
- **NumPy** – numerical operations
- **CVZone** – hand tracking and gesture detection (MediaPipe-based)

---

## Project Files

```text
.
├── SnakeGame.py              # Main game logic and webcam loop
├── Donut.png                 # Food sprite image
├── obs.png                   # Obstacle sprite image
├── Demo Video.mp4            # Gameplay demonstration video
├── Project_Report_Team4.pdf  # Project report
├── SnakeGame_Poster.pptx     # Project poster
├── SnakeGame_Project (1).pptx# Project presentation slides
└── README.md

---

##  How to Run

- Prerequisites
- Python 3.9 or higher
- A working webcam

- Step 1: Install dependencies
  pip install opencv-python numpy cvzone

  If CVZone gives issues, also install:
  pip install mediapipe

- Step 2: Verify file placement
  Ensure the following files are in the same directory:
  
  SnakeGame.py
  Donut.png
  obs.png

- Step 3: Run the game

  python SnakeGame.py

- Controls

  ENTER → Start the game
  Swipe with index finger → Change snake direction
  P → Pause / Resume
  R → Restart the game
  ESC → Exit


---