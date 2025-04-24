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
BATCH_SIZE = 64        # Rozmiar batcha do treningu
GAMMA = 0.99           # Współczynnik dyskontowania przyszłych nagród
EPS_START = 1.0        # Początkowa wartość epsilon (eksploracja)
EPS_END = 0.01         # Minimalna wartość epsilon
EPS_DECAY = 0.995      # Współczynnik zmniejszania epsilon
MEMORY_CAPACITY = 10000 # Pojemność pamięci doświadczeń
LEARNING_RATE = 0.001
NUM_ACTIONS = 4

# Akcje: 0-lewo, 1-prawo, 2-góra, 3-dół
NUM_ACTIONS = 4


# Model który na podstawie stanu podejmuje akcje
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

# definicja agenta
class Agent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.epsilon = EPS_START
        self.model = DQN(7)  # Uproszczony stan: [x_gracza, y_gracza, najblizsza_pilka_x, najblizsza_pilka_y, najblizsza_pulapka_x, najblizsza_pulapka_y
        #alive ]
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    # pobieranie stanu gry
    def get_state(self, player, balls, traps, players, alive):
        if not balls:
            return np.array([player['x'] / W, player['y'] / H, 0, 0], dtype=np.float32)

        """
        if not players:
            return np.array([player['x'] / W, player['y'] / H, 0, 0], dtype=np.float32)
        """

        # Znajdź najbliższą piłkę
        closest_ball = min(balls, key=lambda b: (b[0] - player['x']) ** 2 + (b[1] - player['y']) ** 2)
        closest_trap = min(traps, key=lambda t: (t[0] - player['x']) ** 2 + (t[1] - player['y']) ** 2)
        #closest_player = min(players, key=lambda p: (p['x'] - player['x']) ** 2 + (p['y'] - player['y']) ** 2)

        #trap_distance = np.sqrt((player['x'] - closest_trap[0]) ** 2 + (player['y'] - closest_trap[1]) ** 2)

        return np.array([
            player['x'] / W,
            player['y'] / H,
            closest_ball[0] / W,
            closest_ball[1] / H,
            closest_trap[0] / W,
            closest_trap[1] / H,
            alive
            #closest_player[0] / W,
            #closest_player[1] / H
        ], dtype=np.float32)


    # zapisuje stan do późniejszego uczenia
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # działaj jeśli random jest mniejsze to eksploruje
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # chwilowe wyłączenie gradientu by przyśpieszyć
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

# Akcje agenta
action = None

### Game implementation
pygame.font.init()

# Constants
PLAYER_RADIUS = 10
START_VEL = 9
BALL_RADIUS = 4
TRAP_RADIUS = 10
data = []
W, H = 1280, 720

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
            pygame.time.delay(3000)  # Wait 3 seconds

        # End of training
        if episodes_count == 0:
            print(f"Training is OVER")
            run = False

        s_balls = balls
        s_traps = traps
        s_player = player
        s_players = {k: v for k, v in players.items() if k != current_id}

        # Pobierz aktualny stan
        state = agent.get_state(player, balls, traps, s_players, player.get("alive"))

        # Wybierz akcję
        action = agent.act(state)

        vel = START_VEL - round(player["score"] / 14)
        if vel <= 1:
            vel = 1


        print(f"Stan aktualny: piłki:{s_balls}\n pułapki:{s_traps}\n gracz:{s_player}\n gracze: {s_players}\n")
        last_score = player["score"]

        # movement based on key presses
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
        last_score = current_score

        # get new state
        new_state = agent.get_state(player, balls, traps, s_players, player.get("alive"))

        # remember experience
        done = not player.get("alive", True)
        agent.remember(state, action, reward, new_state, done)

        # Learning based on experience
        if player.get("alive") == False and episodes_count :
            print(f"Player is dead no learning")
            continue

        agent.replay()

        for event in pygame.event.get():
            # if user hits red x button close window
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                # if user hits a escape key close program
                if event.key == pygame.K_ESCAPE:
                    run = False

        # przerysuj i aktualizuj
        redraw_window(players, balls, traps, game_time, player["score"], episodes_count)
        pygame.display.update()

    server.disconnect()
    pygame.quit()
    quit()


# get users name
name = "agent_4"

# make window start in top left hand corner
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)

# setup pygame window
WIN = pygame.display.set_mode((W, H))
pygame.display.set_caption("Blobs")

if __name__ == "__main__":
	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		main(name)