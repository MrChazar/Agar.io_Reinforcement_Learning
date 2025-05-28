import contextlib
import sys
import asyncio
import platform
import pygame
from client import Network
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import numpy as np
import math

# Suppress pygame output
with contextlib.redirect_stdout(None):
    import pygame

# Constants
PLAYER_RADIUS = 10
START_VEL = 9
BALL_RADIUS = 4
TRAP_RADIUS = 10
W, H = 300, 300  # Match server dimensions
FPS = 3000

# Pygame setup
pygame.font.init()
NAME_FONT = pygame.font.SysFont("comicsans", 20)
TIME_FONT = pygame.font.SysFont("comicsans", 30)
SCORE_FONT = pygame.font.SysFont("comicsans", 26)
WIN = pygame.display.set_mode((W, H))
pygame.display.set_caption("Blobs")

COLORS = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0),
          (0, 255, 128), (0, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255),
          (255, 0, 255), (255, 0, 128), (128, 128, 128), (0, 0, 0)]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# DQN Model
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

# RL Parameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
N_ACTIONS = 4  # Left, Right, Up, Down
N_OBSERVATIONS = 86  # From get_state: 4 (player) + 40 (balls) + 30 (traps) + 12 (other players)

# Initialize models
policy_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
target_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0
episode_durations = []
episode_rewards = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                 device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def save_model(episode):
    # In Pyodide, we avoid local file I/O; simulate saving by storing in memory or logging
    torch.save(policy_net.state_dict(), f'model_weights_episode_{episode}.pth')
    print(f"[SAVE] Model weights would be saved for episode {episode}")

def load_model(episode):
    try:
        # For local execution, load weights
        policy_net.load_state_dict(torch.load(f'model_weights_episode_{episode}.pth'))
        policy_net.eval()
        print(f"[LOAD] Model weights would be loaded for episode {episode}")
        return True
    except Exception as e:
        print(f"[LOAD ERROR] Could not load weights: {e}")
        return False

def get_state(player, balls, traps, players):
    state = [
        player['x'] / W,  # Normalize to [0,1]
        player['y'] / H,
        float(player['alive']),
        player['score'] / 100.0  # Normalize score
    ]
    sorted_balls = sorted(balls, key=lambda b: (b[0] - player['x'])**2 + (b[1] - player['y'])**2)
    for i in range(min(20, len(sorted_balls))):
        state.extend([sorted_balls[i][0] / W, sorted_balls[i][1] / H])
    for i in range(20 - min(20, len(sorted_balls))):
        state.extend([0, 0])

    sorted_traps = sorted(traps, key=lambda t: (t[0] - player['x'])**2 + (t[1] - player['y'])**2)
    for i in range(min(15, len(sorted_traps))):
        state.extend([sorted_traps[i][0] / W, sorted_traps[i][1] / H])
    for i in range(15 - min(15, len(sorted_traps))):
        state.extend([0, 0])

    player_list = [p for p in players.values() if p['name'] != player['name']]
    if player_list:
        sorted_players = sorted(player_list,
                              key=lambda p: (p['x'] - player['x'])**2 + (p['y'] - player['y'])**2)
        for i in range(min(3, len(sorted_players))):
            other_player = sorted_players[i]
            state.extend([
                other_player['x'] / W,
                other_player['y'] / H,
                other_player['score'] / 100.0,
                0 if other_player.get('score', 0) > player.get('score', 0) else 1
            ])
        for i in range(3 - min(3, len(sorted_players))):
            state.extend([0, 0, 0, 0])
    else:
        for i in range(3):
            state.extend([0, 0, 0, 0])

    return torch.tensor([state], device=device, dtype=torch.float)

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.title('Training...' if not show_result else 'Result')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.subplot(2, 1, 2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.savefig('training_progress.png')

def redraw_window(players, balls, traps, game_time, score, episodes_count):
    WIN.fill((255, 255, 255))
    for ball in balls:
        pygame.draw.circle(WIN, ball[2], (ball[0], ball[1]), BALL_RADIUS)
    for trap in traps:
        pygame.draw.circle(WIN, trap[2], (trap[0], trap[1]), TRAP_RADIUS)
    for player in sorted(players, key=lambda x: players[x]["score"]):
        p = players[player]
        pygame.draw.circle(WIN, p["color"], (p["x"], p["y"]), PLAYER_RADIUS)
        text = NAME_FONT.render(p["name"], 1, (0, 0, 0))
        WIN.blit(text, (p["x"] - text.get_width() / 2, p["y"] - text.get_height() / 2))
    sort_players = list(reversed(sorted(players, key=lambda x: players[x]["score"])))
    title = TIME_FONT.render("Scoreboard", 1, (0, 0, 0))
    start_y = 25
    x = W - title.get_width() - 10
    WIN.blit(title, (x, 5))
    ran = min(len(players), 3)
    for count, i in enumerate(sort_players[:ran]):
        text = SCORE_FONT.render(str(count + 1) + ". " + players[i]["name"], 1, (0, 0, 0))
        WIN.blit(text, (x, start_y + count * 20))
    text = TIME_FONT.render("Time: " + str(round(game_time)), 1, (0, 0, 0))
    WIN.blit(text, (10, 10))
    text = TIME_FONT.render("Score: " + str(round(score)), 1, (0, 0, 0))
    WIN.blit(text, (10, 15 + text.get_height()))
    text = TIME_FONT.render("Episode: " + str(round(episodes_count)), 1, (0, 0, 0))
    WIN.blit(text, (10, 40 + text.get_height()))

async def main(name, load_episode="956"):
    server = Network()
    current_id = server.connect(name)
    balls, traps, players, game_time, episodes_count = server.send("get")
    clock = pygame.time.Clock()

    if load_episode is not None:
        if load_model(load_episode):
            policy_net.eval()
            global EPS_START, EPS_END
            EPS_START = 0.0
            EPS_END = 0.0  # Disable exploration for testing

    prev_episode = episodes_count
    total_reward = 0
    last_pos = None

    while True:
        clock.tick(FPS)
        player = players[current_id]

        # Get current state
        state = get_state(player, balls, traps, players)
        alive = player.get("alive", True)

        # Select action
        action = select_action(state)
        action_idx = action.item()
        vel = START_VEL - round(player["score"] / 14)
        if vel < 1:
            vel = 1

        # Execute action
        new_x, new_y = player["x"], player["y"]
        if action_idx == 0 and new_x - vel >= 0:  # Left
            new_x -= vel
        elif action_idx == 1 and new_x + vel <= W:  # Right
            new_x += vel
        elif action_idx == 2 and new_y - vel >= 0:  # Up
            new_y -= vel
        elif action_idx == 3 and new_y + vel <= H:  # Down
            new_y += vel

        # Penalize staying in place
        reward = player["reward"]
        if last_pos == (new_x, new_y) and alive:
            reward -= 0.01
        last_pos = (new_x, new_y)

        data = f"move {new_x} {new_y}"
        balls, traps, players, game_time, episodes_count = server.send(data)

        # Get next state and reward
        next_state = get_state(players[current_id], balls, traps, players) if alive else None
        reward = torch.tensor([reward], device=device, dtype=torch.float)
        total_reward += reward.item()

        # Store transition
        if not load_episode:
            memory.push(state, action, next_state, reward)
            optimize_model()

            # Soft update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        # Handle episode end
        if episodes_count < prev_episode:
            episode_durations.append(game_time)
            episode_rewards.append(total_reward)
            print(f"[EPISODE {prev_episode}] Duration: {game_time}, Total Reward: {total_reward}")
            total_reward = 0
            plot_durations()
            if not load_episode and prev_episode % 10 == 0:
                save_model(prev_episode)
            prev_episode = episodes_count

        # Handle game over
        if not alive:
            font = pygame.font.SysFont("comicsans", 50)
            text = font.render("GAME OVER - YOU DIED", 1, (255, 0, 0))
            WIN.blit(text, (W / 2 - text.get_width() / 2, H / 2 - text.get_height() / 2))

        # Render
        redraw_window(players, balls, traps, game_time, player["score"], episodes_count)
        pygame.display.update()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                server.disconnect()
                save_model("956")
                plot_durations(show_result=True)
                pygame.quit()
                return

        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main("DQN_Agent"))
else:
    if __name__ == "__main__":
        asyncio.run(main("DQN_Agent"))