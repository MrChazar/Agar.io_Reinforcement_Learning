import contextlib
import sys
import torch
from collections import deque
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import os
import time

with contextlib.redirect_stdout(None):
    import pygame
from client import Network

# Constants
PLAYER_RADIUS = 10
START_VEL = 9
BALL_RADIUS = 4
TRAP_RADIUS = 10
W, H = 300, 300
SAVE_INTERVAL = 100
GRID_SIZE = 64  # Rozmiar siatki dla CNN (64x64)


# Model CNN
class DQN(nn.Module):
    def __init__(self, in_channels, out_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Oblicz rozmiar po warstwach konwolucyjnych i pooling
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - kernel_size) // stride + 1

        conv_size = GRID_SIZE // 4  # Po dwóch max poolingach (2x2, stride=2)
        fc_input_size = 64 * conv_size * conv_size

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, out_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Spłaszczenie
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Define memory for Experience Replay
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(list(self.memory), sample_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, state_channels, action_size):
        self.state_channels = state_channels
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.batch_size = 64
        self.memory = ReplayMemory(10000)
        self.update_target_every = 10

        # Two networks
        self.policy_net = DQN(state_channels, action_size).to(self.device)
        self.target_net = DQN(state_channels, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.steps = 0
        self.episode = 0
        self.rewards_history = []

    def get_state(self, player, balls, traps, players):
        # Tworzenie siatki 2D dla stanu (6 kanałów)
        state = np.zeros((self.state_channels, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        scale_x = GRID_SIZE / W
        scale_y = GRID_SIZE / H

        # Funkcja do rysowania gaussowskich plam
        def draw_gaussian(grid, x, y, radius, value=1.0):
            x = int(x * scale_x)
            y = int(y * scale_y)
            radius = int(radius * scale_x)
            for i in range(max(0, x - radius), min(GRID_SIZE, x + radius + 1)):
                for j in range(max(0, y - radius), min(GRID_SIZE, y + radius + 1)):
                    dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                    if dist <= radius:
                        grid[i, j] += value * np.exp(-dist ** 2 / (2 * (radius / 2) ** 2))

        # Kanał 1: Gracz
        draw_gaussian(state[0], player['x'], player['y'], PLAYER_RADIUS)

        # Kanał 2: Piłki
        for ball in balls:
            draw_gaussian(state[1], ball[0], ball[1], BALL_RADIUS)

        # Kanał 3: Pułapki
        for trap in traps:
            draw_gaussian(state[2], trap[0], trap[1], TRAP_RADIUS)

        # Kanał 4: Inni gracze (wartość zależy od różnicy score)
        for p in players.values():
            if p['name'] != player['name']:
                value = 1.0 if p['score'] > player['score'] else 0.5
                draw_gaussian(state[3], p['x'], p['y'], PLAYER_RADIUS, value)

        # Kanał 5: Score gracza (jednolita wartość)
        state[4] = player['score'] / 100.0

        # Kanał 6: Prędkość gracza
        vel = START_VEL - round(player["score"] / 14)
        vel = max(1, vel)
        state[5] = vel / START_VEL

        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.cat(next_states)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, filename):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode,
            'rewards_history': self.rewards_history
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode = checkpoint['episode']
        self.rewards_history = checkpoint['rewards_history']
        print(f"Loaded model from {filename}, epsilon: {self.epsilon}")


# Game initialization
pygame.font.init()
NAME_FONT = pygame.font.SysFont("comicsans", 20)
TIME_FONT = pygame.font.SysFont("comicsans", 30)
SCORE_FONT = pygame.font.SysFont("comicsans", 26)


def main(name, train_mode=True, model_file=None):
    WIN = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Blobs - CNN DQN Agent")

    server = Network()
    current_id = server.connect(name)
    balls, traps, players, game_time, episodes_count = server.send("get")

    state_channels = 6  # Gracz, piłki, pułapki, inni gracze, score, prędkość
    action_size = 4
    agent = Agent(state_channels, action_size)

    if model_file and os.path.exists(model_file):
        agent.load(model_file)

    clock = pygame.time.Clock()
    run = True
    total_reward = 0

    while run:
        clock.tick(30)
        player = players[current_id]

        state = agent.get_state(player, balls, traps, players)
        action = agent.act(state, training=train_mode)

        vel = START_VEL - round(player["score"] / 14)
        if vel <= 1:
            vel = 1

        if action == 0:  # Left
            player["x"] = max(0, player["x"] - vel)
        elif action == 1:  # Right
            player["x"] = min(W, player["x"] + vel)
        elif action == 2:  # Up
            player["y"] = max(0, player["y"] - vel)
        elif action == 3:  # Down
            player["y"] = min(H, player["y"] + vel)

        data = "move " + str(player["x"]) + " " + str(player["y"])
        balls, traps, players, game_time, episodes_count = server.send(data)

        next_state = agent.get_state(player, balls, traps, players)
        reward = player["reward"]
        done = not player.get("alive", True)

        if train_mode:
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()

            total_reward += reward

            print(
                f"Ep: {agent.episode} Step: {agent.steps} Act: {action} Done: {done} Reward: {reward:.2f} Eps: {agent.epsilon:.2f} Total: {total_reward:.2f} Loss: {loss:.4f}")

            if done:
                agent.episode = episodes_count  # Synchronizacja z serwerem
                agent.rewards_history.append(total_reward)
                total_reward = 0
                agent.steps = 0

                if agent.episode % SAVE_INTERVAL == 0:
                    agent.save(f"cnn_dqn_model_ep{agent.episode}.pt")
                    print(f"Saved model at episode {agent.episode}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False

        WIN.fill((255, 255, 255))

        for ball in balls:
            pygame.draw.circle(WIN, ball[2], (ball[0], ball[1]), BALL_RADIUS)

        for trap in traps:
            pygame.draw.circle(WIN, trap[2], (trap[0], trap[1]), TRAP_RADIUS)

        for p in sorted(players.values(), key=lambda x: x["score"]):
            pygame.draw.circle(WIN, p["color"], (p["x"], p["y"]), PLAYER_RADIUS)
            text = NAME_FONT.render(p["name"], 1, (0, 0, 0))
            WIN.blit(text, (p["x"] - text.get_width() / 2, p["y"] - text.get_height() / 2))

        text = TIME_FONT.render(f"Score: {player['score']}", 1, (0, 0, 0))
        WIN.blit(text, (10, 10))

        text = TIME_FONT.render(f"Episode: {episodes_count}", 1, (0, 0, 0))
        WIN.blit(text, (10, 40))

        if not train_mode:
            text = TIME_FONT.render("EVALUATION MODE", 1, (255, 0, 0))
            WIN.blit(text, (W // 2 - text.get_width() // 2, 10))

        pygame.display.update()

    if train_mode:
        agent.save("cnn_dqn_model_final.pt")
    server.disconnect()
    pygame.quit()


if __name__ == "__main__":
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)
    main("cnn_dqn_agent", train_mode=True, model_file="cnn_dqn_model_final.pt")