import sys
from client import Network
import sys
import random
import os
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame


### implementacja modelu

# hiperparametry
BATCH_SIZE = 64        # batch size for training
GAMMA = 0.99           # Współczynnik dyskontowania przyszłych nagród
EPS_START = 1.0        # exploration
EPS_END = 0.01         # minimal exploration
EPS_DECAY = 0.995      # exploration decrease factor
MEMORY_CAPACITY = 10000 #
LEARNING_RATE = 0.001
NUM_ACTIONS = 4

# get users name
NAME = "agent_2"



# Agent model
# remade into convolutional network
class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS))

    def forward(self, x):
        return self.fc(x)

# Agent definition
class Agent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.epsilon = EPS_START
        self.model = DQN(54)
        # our game state: [x_player, y_player, alive, player_score
        # closest_ball_x_1, closest_ball_y_1 ...closest_ball_x_10, closest_ball_y_10,
        # closest_trap_x_1, closest_trap_y_1 ... closest_trap_x_10, closest_trap_y_10
        # closest_player_x_1, closest_player_y_1 closest_player_edibile_1 ... closest_player_x_3, closest_player_y_3 closest_player_edibile_3  ]
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

        if os.path.isfile(f"{NAME}.pth"):
            self.model.load_state_dict(torch.load(f"{NAME}.pth"))
            self.model.eval()
            print(f"Model {NAME}.pth loaded")
        else:
            print("No model")

    # pobieranie stanu gry
    def get_state(self, player, balls, traps, players):

        #4
        state = [player['x'] / W, player['y'] / H, float(player['alive']), player['score']]
        sorted_balls = sorted(balls, key=lambda b: (b[0] - player['x']) ** 2 + (b[1] - player['y']) ** 2)

        #20
        for i in range(min(10, len(sorted_balls))):
            state.extend([sorted_balls[i][0] / W, sorted_balls[i][1] / H])

        for i in range(10 - min(10, len(sorted_balls))):
            state.extend([0, 0])

        sorted_traps = sorted(traps, key=lambda t: (t[0] - player['x']) ** 2 + (t[1] - player['y']) ** 2)

        #20
        for i in range(min(10, len(sorted_traps))):
            state.extend([sorted_traps[i][0] / W, sorted_traps[i][1] / H])

        for i in range(10 - min(10, len(sorted_traps))):
            state.extend([0, 0])

        # 8
        print(f"PLAYERS {player.values()}")
        player_list = list(players.values())
        if player_list:
            sorted_players = sorted(player_list,
                                    key=lambda p: (p['x'] - player['x']) ** 2 + (p['y'] - player['y']) ** 2)

            for i in range(min(3, len(sorted_players))):
                other_player = sorted_players[i]
                state.extend([
                    other_player['x'] / W,
                    other_player['y'] / H,
                    1 if other_player.get('score', 0) > player.get('score', 0) else 0 # 1 if smaller 0 if bigger
                ])

            for i in range(3 - min(3, len(sorted_players))):
                state.extend([0, 0, 0])
        else:
            for i in range(3):
                state.extend([0, 0, 0])


        print(f"State that goes into model: {state}")
        return np.array(state, dtype=np.float32)


    # save state for later learning
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # if random smaller explore
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # check what it does
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * GAMMA * next_q

        loss = self.criterion(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Zmniejsz epsilon
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


# Inicjalizacja agenta
agent = Agent()

# GAME ENV
s_balls = []
s_traps = []
s_player = {}
s_players = {}
last_score = 0

# Agent action
action = None

### Game implementation
pygame.font.init()

# Constants
PLAYER_RADIUS = 10
START_VEL = 9
BALL_RADIUS = 4
TRAP_RADIUS = 10
data = []
W, H = 1024, 640

NAME_FONT = pygame.font.SysFont("comicsans", 20)
TIME_FONT = pygame.font.SysFont("comicsans", 30)
SCORE_FONT = pygame.font.SysFont("comicsans", 26)

COLORS = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0), (0, 255, 128), (0, 255, 255),
          (0, 128, 255), (0, 0, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128), (128, 128, 128),
          (0, 0, 0)]
# Dynamic Variables
players = {}
balls = []
traps = []


# FUNCTIONS
def convert_time(t):
    """
	converts a time given in seconds to a time in
	minutes

	:param t: int
	:return: string
	"""
    if type(t) == str:
        return t

    if int(t) < 60:
        return str(t) + "s"
    else:
        minutes = str(t // 60)
        seconds = str(t % 60)

        if int(seconds) < 10:
            seconds = "0" + seconds

        return minutes + ":" + seconds


def redraw_window(players, balls, traps, game_time, score, episodes_count):
    """
	draws each frame
	:return: None
	"""
    WIN.fill((255, 255, 255))  # fill screen white, to clear old frames

    # draw all the orbs/balls
    for ball in balls:
        pygame.draw.circle(WIN, ball[2], (ball[0], ball[1]), BALL_RADIUS)

    for trap in traps:
        pygame.draw.circle(WIN, trap[2], (trap[0], trap[1]), TRAP_RADIUS)

    # draw each player in the list
    for player in sorted(players, key=lambda x: players[x]["score"]):
        p = players[player]
        pygame.draw.circle(WIN, p["color"], (p["x"], p["y"]), PLAYER_RADIUS + round(p["score"]))

        # render and draw name for each player
        text = NAME_FONT.render(p["name"], 1, (0, 0, 0))
        WIN.blit(text, (p["x"] - text.get_width() / 2, p["y"] - text.get_height() / 2))

    # draw scoreboard
    sort_players = list(reversed(sorted(players, key=lambda x: players[x]["score"])))
    title = TIME_FONT.render("Scoreboard", 1, (0, 0, 0))
    start_y = 25
    x = W - title.get_width() - 10
    WIN.blit(title, (x, 5))

    ran = min(len(players), 3)
    for count, i in enumerate(sort_players[:ran]):
        text = SCORE_FONT.render(str(count + 1) + ". " + str(players[i]["name"]), 1, (0, 0, 0))
        WIN.blit(text, (x, start_y + count * 20))

    # draw time
    text = TIME_FONT.render("Time: " + convert_time(game_time), 1, (0, 0, 0))
    WIN.blit(text, (10, 10))

    # draw score
    text = TIME_FONT.render("Score: " + str(round(score)), 1, (0, 0, 0))
    WIN.blit(text, (10, 15 + text.get_height()))

    # draw episode
    text = TIME_FONT.render("Episode: " + str(round(episodes_count)), 1, (0, 0, 0))
    WIN.blit(text, (10, 40 + text.get_height()))


def main(name):
    """
	function for running the game,
	includes the main loop of the game

	:param players: a list of dicts represting a player
	:return: None
	"""
    global players
    global last_score

    # start by connecting to the network
    server = Network()
    current_id = server.connect(name)
    balls, traps, players, game_time, episodes_count = server.send("get")

    # setup the clock, limit to 30fps
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(30)  # 30 fps max
        player = players[current_id]

        # Death
        if not player.get("alive", True):
            font = pygame.font.SysFont("comicsans", 50)
            text = font.render("GAME OVER - YOU DIED", 1, (255, 0, 0))
            WIN.blit(text, (W / 2 - text.get_width() / 2, H / 2 - text.get_height() / 2))
            pygame.display.update()
            redraw_window(players, balls, traps, game_time, player["score"], episodes_count)

        # End of training
        if episodes_count == 0:
            print(f"Training is OVER")
            run = False

        s_balls = balls
        s_traps = traps
        s_player = player
        s_players = {k: v for k, v in players.items() if k != current_id}

        # get game state
        state = agent.get_state(player, balls, traps, s_players)

        # agent choose action
        action = agent.act(state)

        vel = START_VEL - round(player["score"] / 14)
        if vel <= 1:
            vel = 1

        # movement based on key presses when player alive
        if player.get("alive"):
            if action == 0:
                if player["x"] - vel - PLAYER_RADIUS - player["score"] >= 0:
                    player["x"] = player["x"] - vel

            if action == 1:
                if player["x"] + vel + PLAYER_RADIUS + player["score"] <= W:
                    player["x"] = player["x"] + vel

            if action == 2:
                if player["y"] - vel - PLAYER_RADIUS - player["score"] >= 0:
                    player["y"] = player["y"] - vel

            if action == 3:
                if player["y"] + vel + PLAYER_RADIUS + player["score"] <= H:
                    player["y"] = player["y"] + vel


        data = "move " + str(player["x"]) + " " + str(player["y"])

        # send data to server and recieve back all players information
        balls, traps, players, game_time, episodes_count = server.send(data)
        s_players = {k: v for k, v in players.items() if k != current_id}

        current_score = player["score"]
        reward = current_score - last_score

        # for discouriging deaths
        if player.get("alive") == False:
            reward = -0.5

        # for """handling""" episode change
        if reward < -2:
            reward = 0

        print(f"Score: {player['score']} | Last Score: {last_score} | Reward: {reward}")
        last_score = current_score

        # get new state
        new_state = agent.get_state(player, balls, traps, s_players)


        # remember experience
        done = not player.get("alive", True)
        agent.remember(state, action, reward, new_state, done)

        agent.replay()

        for event in pygame.event.get():
            # if user hits red x button close window
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                # if user hits a escape key close program
                if event.key == pygame.K_ESCAPE:
                    run = False

        # redraw and update
        redraw_window(players, balls, traps, game_time, player["score"], episodes_count)
        pygame.display.update()
    try:
        torch.save(agent.model.state_dict(), f"{NAME}.pth")
        print(f"Model saved")
    except:
        print("Error during saving")
    server.disconnect()
    pygame.quit()
    quit()




# make window start in top left hand corner
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)

# setup pygame window
WIN = pygame.display.set_mode((W, H))
pygame.display.set_caption("Blobs")

if __name__ == "__main__":
	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		main(NAME)