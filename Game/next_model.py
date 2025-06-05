# small network game that has differnt blobs
# moving around the screen
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
PLAYER_RADIUS = 15
START_VEL = 9
BALL_RADIUS = 4
TRAP_RADIUS = 10
W, H = 300, 300
SAVE_INTERVAL = 100  # Save weights every 20 episodes
polyak = 0.995

#Model
class DQN(nn.Module):
    def __init__(self, input_dim, hidden, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)

        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.bn2 = nn.BatchNorm1d(hidden // 2)

        self.fc3 = nn.Linear(hidden // 2, hidden // 4)
        self.bn3 = nn.BatchNorm1d(hidden // 4)

        self.out = nn.Linear(hidden // 4, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))

        x = F.relu(self.bn2(self.fc2(x)))

        x = F.relu(self.bn3(self.fc3(x)))

        return self.out(x)

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        # Konwertujemy deque na listę przed próbkowaniem
        return random.sample(list(self.memory), sample_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99991
        self.batch_size = 192
        self.memory = ReplayMemory(25000)
        self.update_target_every = 10

        # Two networks
        self.policy_net = DQN(state_size, 192, action_size).to(self.device)
        self.target_net = DQN(state_size, 192, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),  lr=self.learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()

        self.steps = 0
        self.episode = 0

    def get_state(self, player, balls, traps, players):

        is_at_left_wall = 1.0 if player['x'] == 0 else 0.0
        is_at_right_wall = 1.0 if player['x'] == W else 0.0
        is_at_top_wall = 1.0 if player['y'] == 0 else 0.0
        is_at_bottom_wall = 1.0 if player['y'] == H else 0.0

        state = [
            player['x'] / W,
            player['y'] / H,
            is_at_left_wall,
            is_at_right_wall,
            is_at_top_wall,
            is_at_bottom_wall,
            float(player['alive']),
            (START_VEL - round(player["score"] / 14)) / START_VEL,
            player["score"] / 50
        ]

        # Add closest balls
        sorted_balls = sorted(balls, key=lambda b: (b[0] - player['x']) ** 2 + (b[1] - player['y']) ** 2)
        for i in range(min(60, len(sorted_balls))):
            state.extend([(sorted_balls[i][0] - player['x'])/ W, (sorted_balls[i][1] - player['y']) / H])
        for i in range(60 - min(60, len(sorted_balls))):
            state.extend([0, 0])

        # Add closest traps
        sorted_traps = sorted(traps, key=lambda t: (t[0] - player['x']) ** 2 + (t[1] - player['y']) ** 2)
        for i in range(min(15, len(sorted_traps))):
            state.extend([(sorted_traps[i][0] - player['x']) / W, (sorted_traps[i][1] - player['y'])/ H])
        for i in range(15 - min(15, len(sorted_traps))):
            state.extend([0, 0])

        # Add other players
        player_list = [p for p in players.values() if p['name'] != player['name']]
        if player_list:
            sorted_players = sorted(player_list,
                                    key=lambda p: (p['x'] - player['x']) ** 2 + (p['y'] - player['y']) ** 2)
            for i in range(min(3, len(sorted_players))):
                other_player = sorted_players[i]
                state.extend([
                    (other_player['x'] - player['x']) / W,
                    (other_player['y'] - player['y']) / H,
                    other_player['score'] / 1000,
                    0 if other_player.get('score', 0) > player.get('score', 0) else 1
                ])
            for i in range(3 - min(3, len(sorted_players))):
                state.extend([0, 0, 0, 0])
        else:
            for i in range(3):
                state.extend([0, 0, 0, 0])

        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0.0  # Zwracamy 0 jako wartość loss gdy nie ma wystarczająco próbek

        # Pobieramy próbkę jako listę
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Konwertujemy na tensory
        states = torch.cat(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.cat(next_states)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_action_indices = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_action_indices.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.loss_fn(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            with torch.no_grad():
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.mul_(polyak).add_(policy_param.data, alpha=1 - polyak)

        return loss.item()

    def save(self, filename):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode,
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode = checkpoint['episode']
        print(f"Loaded model from {filename}, epsilon: {self.epsilon}")


# Game initialization
pygame.font.init()
NAME_FONT = pygame.font.SysFont("comicsans", 20)
TIME_FONT = pygame.font.SysFont("comicsans", 30)
SCORE_FONT = pygame.font.SysFont("comicsans", 26)

def main(name, train_mode=True, model_file=None):
    # Setup pygame window
    WIN = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Blobs - DQN Agent")

    # Connect to server
    server = Network()
    current_id = server.connect(name)
    balls, traps, players, game_time, episodes_count = server.send("get")

    # Initialize agent
    state_size = 171 # player + balls + traps + other players
    action_size = 4  # Left, Right, Up, Down
    agent = Agent(state_size, action_size)

    if model_file and os.path.exists(model_file):
        agent.load(model_file)

    clock = pygame.time.Clock()
    run = True
    total_reward = 0

    while run:
        clock.tick(30)
        player = players[current_id]

        # Get current state
        state = agent.get_state(player, balls, traps, players)

        # Choose action
        action = agent.act(state, training=train_mode)

        # Execute action
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

        # Send move to server
        data = "move " + str(player["x"]) + " " + str(player["y"])
        balls, traps, players, game_time, episodes_count = server.send(data)

        # Get new state and reward
        next_state = agent.get_state(player, balls, traps, players)
        reward = player["reward"]
        done = not player.get("alive", True)

        if train_mode:
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()

            total_reward += reward

            # Print debug info
            print(
                f"Ep: {agent.episode} Step: {agent.steps} Act: {action} Done: {done} Reward: {reward:.2f} Eps: {agent.epsilon:.2f} Total: {total_reward:.2f}")

            if done:
                agent.episode += 1

                total_reward = 0

                if agent.episode % SAVE_INTERVAL == 0:
                    agent.save(f"dqn_model_ep{agent.episode}.pt")

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False

        # Render
        WIN.fill((255, 255, 255))

        # Draw game elements
        for ball in balls:
            pygame.draw.circle(WIN, ball[2], (ball[0], ball[1]), BALL_RADIUS)

        for trap in traps:
            pygame.draw.circle(WIN, trap[2], (trap[0], trap[1]), TRAP_RADIUS)

        for p in sorted(players.values(), key=lambda x: x["score"]):
            pygame.draw.circle(WIN, p["color"], (p["x"], p["y"]), PLAYER_RADIUS)
            text = NAME_FONT.render(p["name"], 1, (0, 0, 0))
            WIN.blit(text, (p["x"] - text.get_width() / 2, p["y"] - text.get_height() / 2))

        # Draw UI
        text = TIME_FONT.render(f"Score: {player['score']}", 1, (0, 0, 0))
        WIN.blit(text, (10, 10))

        text = TIME_FONT.render(f"Time: {game_time}", 1, (0, 0, 0))
        WIN.blit(text, (10, 40))

        if not train_mode:
            text = TIME_FONT.render("EVALUATION MODE", 1, (255, 0, 0))
            WIN.blit(text, (W // 2 - text.get_width() // 2, 10))

        pygame.display.update()

    # Clean up
    if train_mode:
        agent.save("dqn_model_final.pt")
    server.disconnect()
    pygame.quit()


if __name__ == "__main__":
    # To train: python game.py --train
    # To evaluate: python game.py --model dqn_model_final.pt

    import argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the DQN agent")
    parser.add_argument("--model", type=str, help="Path to model file for evaluation")
    args = parser.parse_args()
    """
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)
    main("dqn_agent", train_mode=True, model_file="dqn_model_final.pt")