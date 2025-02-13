import pygame
import sys
import csv
from uuid import uuid4

# Initialize Pygame and its font module
pygame.init()
pygame.font.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong Game with Data Generation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle settings
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
PADDLE_SPEED = 5

# Ball settings
BALL_SIZE = 15
ball_speed_x = 4
ball_speed_y = 4

# Starting positions
left_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
right_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
ball_x = WIDTH // 2 - BALL_SIZE // 2
ball_y = HEIGHT // 2 - BALL_SIZE // 2

# Score variables
left_score = 0
right_score = 0

# Font for score display
score_font = pygame.font.SysFont("comicsans", 40)

# Set up CSV file for demonstration data (for supervised learning)
demo_file = open(f"ping-pong/training-data/'{uuid4()}'.csv", mode="w", newline="")
demo_writer = csv.writer(demo_file)
# Write header row (customize features as needed)
demo_writer.writerow(["ball_x", "ball_y", "ball_speed_x", "ball_speed_y", "left_paddle_y", "action"])

# Create a clock object to manage the frame rate
clock = pygame.time.Clock()

# Main game loop
while True:
    action = None  # Will record the human's action per frame (0: up, 1: down, 2: none)

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            demo_file.close()  # Make sure to close the file when quitting
            pygame.quit()
            sys.exit()

    # --- Input Handling for the Human-Controlled Paddle (Left Paddle) ---
    keys = pygame.key.get_pressed()
    if (keys[pygame.K_w] or keys[pygame.K_UP]) and left_paddle_y > 0:
        left_paddle_y -= PADDLE_SPEED
        action = 0  # Moving up
    if (keys[pygame.K_s] or keys[pygame.K_DOWN]) and left_paddle_y < HEIGHT - PADDLE_HEIGHT:
        left_paddle_y += PADDLE_SPEED
        action = 1  # Moving down

    # If no movement keys were pressed, record "no action"
    if action is None:
        action = 2  # No movement

    # --- Record Demonstration Data ---
    # State includes the ball's position and velocity, and the left paddle's position.
    state = [ball_x, ball_y, ball_speed_x, ball_speed_y, left_paddle_y]
    demo_writer.writerow(state + [action])

    # --- Update the Ball Position ---
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # --- Collision with Top and Bottom Walls ---
    if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
        ball_speed_y *= -1

    # --- Collision with Left Paddle (Human) ---
    if (ball_x <= 10 + PADDLE_WIDTH and
        left_paddle_y < ball_y + BALL_SIZE and
        left_paddle_y + PADDLE_HEIGHT > ball_y):
        ball_speed_x *= -1

    # --- Collision with Right Paddle (AI) ---
    if (ball_x + BALL_SIZE >= WIDTH - 10 - PADDLE_WIDTH and
        right_paddle_y < ball_y + BALL_SIZE and
        right_paddle_y + PADDLE_HEIGHT > ball_y):
        ball_speed_x *= -1

    # --- Check for Scoring and Reset Ball ---
    if ball_x < 0:
        # Ball passed the left side: AI scores
        right_score += 1
        ball_x = WIDTH // 2 - BALL_SIZE // 2
        ball_y = HEIGHT // 2 - BALL_SIZE // 2
        ball_speed_x *= -1  # Reverse ball direction for variety

    if ball_x > WIDTH:
        # Ball passed the right side: Human scores
        left_score += 1
        ball_x = WIDTH // 2 - BALL_SIZE // 2
        ball_y = HEIGHT // 2 - BALL_SIZE // 2
        ball_speed_x *= -1  # Reverse ball direction for variety

    # --- Basic AI for the Right Paddle ---
    # The AI paddle follows the ball's vertical position.
    if right_paddle_y + PADDLE_HEIGHT / 2 < ball_y + BALL_SIZE / 2 and right_paddle_y < HEIGHT - PADDLE_HEIGHT:
        right_paddle_y += PADDLE_SPEED
    elif right_paddle_y + PADDLE_HEIGHT / 2 > ball_y + BALL_SIZE / 2 and right_paddle_y > 0:
        right_paddle_y -= PADDLE_SPEED

    # --- Drawing Section ---
    window.fill(BLACK)  # Clear the screen with black

    # Draw paddles
    pygame.draw.rect(window, WHITE, (10, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(window, WHITE, (WIDTH - 10 - PADDLE_WIDTH, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    # Draw the ball
    pygame.draw.rect(window, WHITE, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))

    # Render and display the scores
    left_score_text = score_font.render(f"{left_score}", True, WHITE)
    right_score_text = score_font.render(f"{right_score}", True, WHITE)
    window.blit(left_score_text, (WIDTH // 4 - left_score_text.get_width() // 2, 20))
    window.blit(right_score_text, (WIDTH * 3 // 4 - right_score_text.get_width() // 2, 20))

    # Update the display
    pygame.display.flip()

    # Maintain a frame rate of 60 frames per second
    clock.tick(60)
