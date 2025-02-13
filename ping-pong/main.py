import os
import sys
import csv
import ast
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from uuid import uuid4

#############################################
#          RL Agent (DQN) Definition        #
#############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

#############################################
#         Hyperparameters & Setup           #
#############################################

# RL hyperparameters
STATE_DIM = 5      # [ball_x, ball_y, ball_speed_x, ball_speed_y, right_paddle_y]
ACTION_DIM = 3     # 0: move up, 1: move down, 2: do nothing
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
REPLAY_CAPACITY = 1000000000

# Epsilon-greedy parameters for exploration (for RL-controlled actions)
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01

# Create the DQN and its optimizer
policy_net = DQN(STATE_DIM, ACTION_DIM).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Create two buffers:
# - replay_buffer: for machine-generated RL transitions
# - demonstration_buffer: for human demonstration transitions (logged when in demo mode)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
demonstration_buffer = []  # simple list for demonstration transitions

# File paths for persistence
MODEL_FILE = "ping-pong/dqn_model.pth"
RL_LOG_FILE = "ping-pong/rl_data.csv"
HUMAN_LOG_FILE = "ping-pong/demo_data.csv"  # (left-paddle log, not used for RL training here)

#############################################
#         Persistence: Load Previous Data   #
#############################################

def load_rl_logs():
    """
    Load previous RL transitions from RL_LOG_FILE.
    Expected CSV header: episode, step, state, action, reward, next_state, done, source
    The state and next_state columns are assumed to be string representations of lists.
    """
    if not os.path.exists(RL_LOG_FILE):
        print("No previous RL log file found.")
        return

    print("Loading previous RL log data...")
    with open(RL_LOG_FILE, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                state = np.array(ast.literal_eval(row["state"]), dtype=np.float32)
                action = int(row["action"])
                reward = float(row["reward"])
                next_state = np.array(ast.literal_eval(row["next_state"]), dtype=np.float32)
                done = bool(float("1" if row["done"] else "0"))
                source = row["source"]
                transition = (state, action, reward, next_state, float(done))
                if source == "demo":
                    demonstration_buffer.append(transition)
                else:
                    replay_buffer.push(transition)
            except Exception as e:
                print("Error processing row:", row, e)
    print(f"Loaded {len(replay_buffer)} machine transitions and {len(demonstration_buffer)} demonstration transitions.")

# Load previously logged RL data (if any)
load_rl_logs()

# Load the saved model (if exists)
if os.path.exists(MODEL_FILE):
    print("Loading saved model...")
    policy_net.load_state_dict(torch.load(MODEL_FILE, map_location=device))
else:
    print("No saved model found. Training from scratch.")

#############################################
#           Utility Functions               #
#############################################

def get_rl_state():
    """
    Returns a normalized state for the RL agent controlling the right paddle.
    State vector: [ball_x, ball_y, ball_speed_x, ball_speed_y, right_paddle_y]
    Normalization: positions divided by screen dimensions, speeds by 10.
    """
    return np.array([
        ball_x / WIDTH,
        ball_y / HEIGHT,
        ball_speed_x / 10.0,
        ball_speed_y / 10.0,
        right_paddle_y / HEIGHT
    ], dtype=np.float32)

def select_action(state):
    """
    Epsilon-greedy action selection.
    """
    global epsilon
    if random.random() < epsilon:
        return random.randrange(ACTION_DIM)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

def train_rl_agent():
    """
    Perform one optimization step from the machine's replay buffer.
    """
    print("Training RL agent...")
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = replay_buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
    
    current_q = policy_net(states).gather(1, actions)
    next_q = policy_net(next_states).max(1)[0].unsqueeze(1)
    target_q = rewards + GAMMA * next_q * (1 - dones)
    
    loss = loss_fn(current_q, target_q.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_rl_combined(iterations=50):
    """
    After each episode, run extra training iterations using the combined data
    from machine transitions and human demonstration transitions.
    """
    print("Training RL agent with combined data...")
    combined = replay_buffer.buffer + demonstration_buffer
    if len(combined) < BATCH_SIZE:
        return
    for _ in range(iterations):
        batch = random.sample(combined, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        current_q = policy_net(states).gather(1, actions)
        next_q = policy_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + GAMMA * next_q * (1 - dones)
        
        loss = loss_fn(current_q, target_q.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#############################################
#        Pygame & Environment Setup         #
#############################################

pygame.init()
pygame.font.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong: RL (Machine) vs Human (with Demo & Persistence)")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle settings (using floats for smoother increases)
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
PADDLE_SPEED = 5.0

# Ball settings
BALL_SIZE = 15
ball_speed_x = 4.0
ball_speed_y = 4.0

# Starting positions
left_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2   # Human-controlled (left)
right_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2    # RL-controlled (right)
ball_x = WIDTH // 2 - BALL_SIZE // 2
ball_y = HEIGHT // 2 - BALL_SIZE // 2

# Score variables (for display)
left_score = 0
right_score = 0

# Font for score display
score_font = pygame.font.SysFont("comicsans", 40)

#############################################
#          CSV Logging Setup                #
#############################################

# Open human log file (for left paddle) in append mode.
if os.path.exists(HUMAN_LOG_FILE):
    human_log_file = open(HUMAN_LOG_FILE, mode="a", newline="")
    human_writer = csv.writer(human_log_file)
else:
    human_log_file = open(HUMAN_LOG_FILE, mode="w", newline="")
    human_writer = csv.writer(human_log_file)
    human_writer.writerow(["time_ms", "ball_x", "ball_y", "ball_speed_x", "ball_speed_y", "left_paddle_y", "human_action"])

# Open RL log file in append mode.
new_rl_file = not os.path.exists(RL_LOG_FILE)
rl_log_file = open(RL_LOG_FILE, mode="a", newline="")
rl_writer = csv.writer(rl_log_file)
if new_rl_file:
    # Write header only if file is new.
    rl_writer.writerow(["id","time_ms","episode", "step", "state", "action", "reward", "next_state", "done", "source"])

#############################################
#           Main Game Loop                  #
#############################################

clock = pygame.time.Clock()
episode = 1
step_in_episode = 0
id = str(uuid4())

running = True
while running:
    machine_reward = 0   # reward for the RL agent this step
    done = False         # flag indicating the end of a rally/episode

    # -------------- Event Handling --------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    # -------------- Human Input (Left Paddle) --------------
    human_action = 2  # default: no movement
    keys = pygame.key.get_pressed()
    if (keys[pygame.K_w] or keys[pygame.K_UP]) and left_paddle_y > 0:
        left_paddle_y -= PADDLE_SPEED
        human_action = 0  # up
    if (keys[pygame.K_s] or keys[pygame.K_DOWN]) and left_paddle_y < HEIGHT - PADDLE_HEIGHT:
        left_paddle_y += PADDLE_SPEED
        human_action = 1  # down

    # Log human left-paddle data
    human_writer.writerow([
        pygame.time.get_ticks(),
        ball_x, ball_y, ball_speed_x, ball_speed_y,
        left_paddle_y, human_action
    ])

    # -------------- Right Paddle Control --------------
    # Demo mode is activated if Left Shift is held; then human controls right paddle.
    demo_mode = keys[pygame.K_LSHIFT]
    current_state = get_rl_state()  # state before updating the right paddle

    if demo_mode:
        # Use human key for right paddle as well.
        demo_action = human_action  
        if demo_action == 0 and right_paddle_y > 0:
            right_paddle_y -= PADDLE_SPEED
        elif demo_action == 1 and right_paddle_y < HEIGHT - PADDLE_HEIGHT:
            right_paddle_y += PADDLE_SPEED
        chosen_action = demo_action  # for logging and transition
        source = "demo"
    else:
        chosen_action = select_action(current_state)
        if chosen_action == 0 and right_paddle_y > 0:
            right_paddle_y -= PADDLE_SPEED
        elif chosen_action == 1 and right_paddle_y < HEIGHT - PADDLE_HEIGHT:
            right_paddle_y += PADDLE_SPEED
        # Action 2: do nothing.
        source = "rl"

    # -------------- Update the Ball --------------
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Bounce off top and bottom
    if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
        ball_speed_y *= -1

    # Collision with left paddle (human)
    if (ball_x <= 10 + PADDLE_WIDTH and
        left_paddle_y < ball_y + BALL_SIZE and
        left_paddle_y + PADDLE_HEIGHT > ball_y):
        ball_speed_x *= -1

    # Collision with right paddle (RL or demo)
    if (ball_x + BALL_SIZE >= WIDTH - 10 - PADDLE_WIDTH and
        right_paddle_y < ball_y + BALL_SIZE and
        right_paddle_y + PADDLE_HEIGHT > ball_y):
        ball_speed_x *= -1
        machine_reward = 1

    # -------------- Check for Scoring --------------
    if ball_x < 0:
        left_score += 1
        done = True
    if ball_x > WIDTH:
        right_score += 1
        machine_reward = -1
        done = True

    next_state = get_rl_state()

    # -------------- Record RL Transition --------------
    transition = (current_state, chosen_action, machine_reward, next_state, float(done))
    # Save to appropriate buffer based on source.
    if demo_mode:
        demonstration_buffer.append(transition)
    else:
        replay_buffer.push(transition)
    # Log to CSV with extra column "source"
    rl_writer.writerow([
        id,
        pygame.time.get_ticks(),
        episode,
        step_in_episode,
        current_state.tolist(),
        chosen_action,
        machine_reward,
        next_state.tolist(),
        done,
        source
    ])

    # -------------- RL Agent Training Step --------------
    if not demo_mode:
        train_rl_agent()

    # -------------- End-of-Episode Handling --------------
    if done:
        # Run additional training on the combined buffer after each rally.
        train_rl_combined(iterations=50)
        # Increase difficulty (speed up ball and paddles)
        ball_speed_x *= 1.0005
        ball_speed_y *= 1.0005
        PADDLE_SPEED *= 1.0002

        # Reset positions for new rally
        ball_x = WIDTH // 2 - BALL_SIZE // 2
        ball_y = HEIGHT // 2 - BALL_SIZE // 2

        step_in_episode = 0
        episode += 1

        # Save the model after each episode
        torch.save(policy_net.state_dict(), MODEL_FILE)
    else:
        step_in_episode += 1

    # -------------- Decay Epsilon --------------
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_min)

    # -------------- Drawing --------------
    window.fill(BLACK)
    # Draw left (human) paddle
    pygame.draw.rect(window, WHITE, (10, int(left_paddle_y), PADDLE_WIDTH, PADDLE_HEIGHT))
    # Draw right (RL or demo) paddle
    pygame.draw.rect(window, WHITE, (WIDTH - 10 - PADDLE_WIDTH, int(right_paddle_y), PADDLE_WIDTH, PADDLE_HEIGHT))
    # Draw ball
    pygame.draw.rect(window, WHITE, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))

    # Draw scores
    left_score_text = score_font.render(f"{left_score}", True, WHITE)
    right_score_text = score_font.render(f"{right_score}", True, WHITE)
    window.blit(left_score_text, (WIDTH // 4 - left_score_text.get_width() // 2, 20))
    window.blit(right_score_text, (WIDTH * 3 // 4 - right_score_text.get_width() // 2, 20))

    pygame.display.flip()
    clock.tick(60)

# Cleanup on exit
human_log_file.close()
rl_log_file.close()
pygame.quit()
sys.exit()
