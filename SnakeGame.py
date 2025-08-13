import math
import random
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


class SnakeGame:
    def __init__(self):
        self.width, self.height = 1280, 720

        try:
            cv2.namedWindow("Temp", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Temp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            screen_info = cv2.getWindowImageRect("Temp")
            if screen_info[2] > 0 and screen_info[3] > 0:
                self.width, self.height = screen_info[2], screen_info[3]
            cv2.destroyWindow("Temp")
        except:
            pass

        self.margin = int(min(self.width, self.height) * 0.05)

        self.food_size = int(min(self.width, self.height) * 0.04)
        self.food_img = cv2.imread('Donut.png', cv2.IMREAD_UNCHANGED)
        if self.food_img is None:
            print("Error: Donut.png not found in the directory")
            self.food_img = None
        else:
            self.food_img = cv2.resize(self.food_img, (self.food_size, self.food_size))

        self.obstacle_size = int(min(self.width, self.height) * 0.08)
        self.obstacle_img = cv2.imread('obs.png', cv2.IMREAD_UNCHANGED)
        if self.obstacle_img is None:
            print("Error: obs.png not found in the directory")
            self.obstacle_img = None
        else:
            self.obstacle_img = cv2.resize(self.obstacle_img, (self.obstacle_size, self.obstacle_size))

        self.num_obstacles = 5
        self.obstacles = []
        self.reset_game()
        self.snake_color = (0, 255, 0)
        self.head_color = (0, 200, 0)
        self.snake_thickness = int(min(self.width, self.height) * 0.02)
        self.head_radius = int(min(self.width, self.height) * 0.02)
        self.growth_amount = int(min(self.width, self.height) * 0.01)
        self.move_distance = int(min(self.width, self.height) * 0.02)
        self.direction = "RIGHT"
        self.move_delay = 0.1
        self.last_move_time = 0

        self.waiting_for_start = True
        self.game_over = False
        self.paused = False
        self.score = 0
        self.high_score = 0

        self.prev_finger_pos = None
        self.swipe_threshold = int(min(self.width, self.height) * 0.12)

    def generate_obstacles(self):
        self.obstacles = []
        for _ in range(self.num_obstacles):
            valid_position = False
            while not valid_position:
                pos = [
                    random.randint(self.margin + self.obstacle_size,
                                   self.width - self.margin - self.obstacle_size),
                    random.randint(self.margin + self.obstacle_size,
                                   self.height - self.margin - self.obstacle_size)
                ]

                head = self.snake_pos[-1]
                dist_to_head = math.hypot(pos[0] - head[0], pos[1] - head[1])
                dist_to_food = math.hypot(pos[0] - self.food_pos[0], pos[1] - self.food_pos[1])

                if dist_to_head > self.obstacle_size * 2 and dist_to_food > self.obstacle_size * 2:
                    valid_position = True
                    self.obstacles.append(pos)

    def reset_game(self):
        self.snake_pos = [[self.width // 2, self.height // 2]]
        self.snake_length = 2
        self.score = 0
        self.game_over = False
        self.waiting_for_start = True
        self.food_pos = self.random_food_position()
        self.direction = "RIGHT"
        self.prev_finger_pos = None
        self.last_move_time = 0
        self.generate_obstacles()

    def random_food_position(self):
        valid_position = False
        while not valid_position:
            pos = [
                random.randint(self.margin + self.food_size, self.width - self.margin - self.food_size),
                random.randint(self.margin + self.food_size, self.height - self.margin - self.food_size)
            ]

            valid_position = True
            for obstacle in self.obstacles:
                dist = math.hypot(pos[0] - obstacle[0], pos[1] - obstacle[1])
                if dist < self.obstacle_size + self.food_size:
                    valid_position = False
                    break

            if valid_position:
                return pos

    def update_direction(self, finger_pos):
        if self.prev_finger_pos is None:
            self.prev_finger_pos = finger_pos
            return

        dx = finger_pos[0] - self.prev_finger_pos[0]
        dy = finger_pos[1] - self.prev_finger_pos[1]

        if abs(dx) < self.swipe_threshold and abs(dy) < self.swipe_threshold:
            return

        if abs(dx) > abs(dy):
            if dx > 0 and self.direction != "LEFT":
                self.direction = "RIGHT"
            elif dx < 0 and self.direction != "RIGHT":
                self.direction = "LEFT"
        else:
            if dy > 0 and self.direction != "UP":
                self.direction = "DOWN"
            elif dy < 0 and self.direction != "DOWN":
                self.direction = "UP"

        self.prev_finger_pos = finger_pos

    def update(self, img, finger_pos=None, current_time=None):
        if self.waiting_for_start:
            self.show_start_screen(img)
            return img

        if self.game_over:
            self.show_game_over(img)
            return img

        if self.paused:
            self.show_paused(img)
            return img

        if finger_pos:
            self.update_direction(finger_pos)

        if current_time is None:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()

        if current_time - self.last_move_time > self.move_delay:
            head = self.snake_pos[-1].copy()

            if self.direction == "RIGHT":
                head[0] += self.move_distance
            elif self.direction == "LEFT":
                head[0] -= self.move_distance
            elif self.direction == "DOWN":
                head[1] += self.move_distance
            elif self.direction == "UP":
                head[1] -= self.move_distance

            self.snake_pos.append(head)
            self.last_move_time = current_time

            while len(self.snake_pos) > self.snake_length:
                self.snake_pos.pop(0)

            if self.check_food_collision(head):
                self.snake_length += self.growth_amount
                self.score += 1
                if self.score > self.high_score:
                    self.high_score = self.score
                self.food_pos = self.random_food_position()

                if self.score % 3 == 0:
                    self.num_obstacles = min(10, self.num_obstacles + 1)
                    self.generate_obstacles()

            if self.check_collision():
                self.game_over = True

        self.draw_snake(img)
        self.draw_food(img)
        self.draw_obstacles(img)
        self.draw_score(img)
        self.draw_boundaries(img)

        return img

    def check_food_collision(self, head_pos):
        if self.food_img is None:
            return False

        distance = math.hypot(head_pos[0] - self.food_pos[0], head_pos[1] - self.food_pos[1])
        return distance < self.food_size // 2

    def check_collision(self):
        head = self.snake_pos[-1]

        if (head[0] < self.margin or head[0] > self.width - self.margin or
                head[1] < self.margin or head[1] > self.height - self.margin):
            return True

        if len(self.snake_pos) > 5:
            for pos in self.snake_pos[:-5]:
                if math.hypot(head[0] - pos[0], head[1] - pos[1]) < self.snake_thickness // 2:
                    return True

        if self.obstacle_img is not None:
            for obstacle in self.obstacles:
                distance = math.hypot(head[0] - obstacle[0], head[1] - obstacle[1])
                if distance < self.obstacle_size // 2 + self.snake_thickness // 2:
                    return True

        return False

    def draw_snake(self, img):
        for i in range(1, len(self.snake_pos)):
            cv2.line(img, tuple(map(int, self.snake_pos[i - 1])),
                     tuple(map(int, self.snake_pos[i])),
                     self.snake_color, self.snake_thickness)

        if len(self.snake_pos) > 0:
            cv2.circle(img, tuple(map(int, self.snake_pos[-1])),
                       self.head_radius, self.head_color, cv2.FILLED)

    def draw_food(self, img):
        if self.food_img is None:
            return

        x, y = self.food_pos
        half_size = self.food_size // 2
        y1, y2 = int(y - half_size), int(y + half_size)
        x1, x2 = int(x - half_size), int(x + half_size)

        if y1 >= 0 and y2 <= img.shape[0] and x1 >= 0 and x2 <= img.shape[1]:
            if self.food_img.shape[2] == 4:
                alpha = self.food_img[:, :, 3] / 255.0
                alpha_inv = 1.0 - alpha
                for c in range(0, 3):
                    img[y1:y2, x1:x2, c] = (
                            alpha * self.food_img[:, :, c] +
                            alpha_inv * img[y1:y2, x1:x2, c]
                    )
            else:
                img[y1:y2, x1:x2] = self.food_img

    def draw_obstacles(self, img):
        if self.obstacle_img is None:
            return

        for obstacle in self.obstacles:
            x, y = obstacle
            half_size = self.obstacle_size // 2
            y1, y2 = int(y - half_size), int(y + half_size)
            x1, x2 = int(x - half_size), int(x + half_size)

            target_height = y2 - y1
            target_width = x2 - x1

            resized_obstacle = cv2.resize(self.obstacle_img, (target_width, target_height))

            if y1 >= 0 and y2 <= img.shape[0] and x1 >= 0 and x2 <= img.shape[1]:
                if resized_obstacle.shape[2] == 4:
                    alpha = resized_obstacle[:, :, 3] / 255.0
                    alpha_inv = 1.0 - alpha
                    for c in range(0, 3):
                        img[y1:y2, x1:x2, c] = (
                                alpha * resized_obstacle[:, :, c] +
                                alpha_inv * img[y1:y2, x1:x2, c]
                        )
                else:
                    img[y1:y2, x1:x2] = resized_obstacle

    def draw_score(self, img):
        font_scale = min(self.width, self.height) / 600
        thickness = max(2, int(font_scale * 2))
        text = f"Score: {self.score}"

        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = int(self.width * 0.05), int(self.height * 0.1)
        cv2.rectangle(img, (x - 10, y - h - 10), (x + w + 10, y + 10), (0, 0, 0), -1)

        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def draw_boundaries(self, img):
        cv2.rectangle(img, (self.margin, self.margin),
                      (self.width - self.margin, self.height - self.margin),
                      (100, 100, 100), 2)

    def show_start_screen(self, img):
        font_scale = min(self.width, self.height) / 500
        thickness = max(2, int(font_scale * 2))

        (text_width, text_height), _ = cv2.getTextSize("VIRTUAL SNAKE GAME", cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                       thickness)
        cv2.putText(img, "VIRTUAL SNAKE GAME",
                    (int((self.width - text_width) / 2), int(self.height * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

        (text_width, text_height), _ = cv2.getTextSize("Press ENTER to Start", cv2.FONT_HERSHEY_SIMPLEX,
                                                       font_scale * 0.8, thickness)
        cv2.putText(img, "Press ENTER to Start",
                    (int((self.width - text_width) / 2), int(self.height * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 0, 0), thickness)


        (text_width, text_height), _ = cv2.getTextSize("Swipe to change direction", cv2.FONT_HERSHEY_SIMPLEX,
                                                       font_scale * 0.6, thickness)
        cv2.putText(img, "Swipe to change direction",
                    (int((self.width - text_width) / 2), int(self.height * 0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 0, 0), thickness)


        score_text = f"Score: {self.high_score}"
        (text_width, text_height), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                                                       thickness)
        cv2.putText(img, score_text,
                    (int((self.width - text_width) / 2), int(self.height * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 0, 0), thickness)

    def show_game_over(self, img):
        font_scale = min(self.width, self.height) / 500
        thickness = max(1, int(font_scale * 2))

        cv2.putText(img, "GAME OVER",
                    (int(self.width * 0.35), int(self.height * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        cv2.putText(img, f"Score: {self.score}",
                    (int(self.width * 0.38), int(self.height * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 0, 0), thickness)

        cv2.putText(img, "Press R to Restart",
                    (int(self.width * 0.33), int(self.height * 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 0, 0), thickness)

    def show_paused(self, img):
        font_scale = min(self.width, self.height) / 500
        thickness = max(1, int(font_scale * 2))

        cv2.putText(img, "PAUSED",
                    (int(self.width * 0.45), int(self.height * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")

game = SnakeGame()
detector = HandDetector(detectionCon=0.7, maxHands=1)

cv2.namedWindow("Virtual Snake Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Virtual Snake Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.waitKey(1)

print("Game controls:")
print("- Press ENTER to start")
print("- Swipe in any direction to change snake direction")
print("- Press P to pause")
print("- Press R to restart")
print("- ESC to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Camera error")
        break

    img = cv2.flip(img, 1)
    hands, _ = detector.findHands(img, flipType=False, draw=False)
    index_finger = None

    if hands:
        lmList = hands[0]['lmList']
        index_finger = lmList[8][0:2]

    current_time = cv2.getTickCount() / cv2.getTickFrequency()

    img = cv2.resize(img, (game.width, game.height))

    img = game.update(img, index_finger, current_time)

    cv2.imshow("Virtual Snake Game", img)

    key = cv2.waitKey(1)
    if key == 13:
        if game.waiting_for_start:
            game.waiting_for_start = False
    elif key == ord('r'):
        game.reset_game()
        game.waiting_for_start = False
    elif key == ord('p') and not game.waiting_for_start:
        game.paused = not game.paused
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()