"""
main server script for running agar.io server

can handle multiple/infinite connections on the same
local network
"""

import socket
from _thread import *
import _pickle as pickle
import time
import random
import math

# setup sockets
S = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
S.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Set constants
PORT = 5555

BALL_RADIUS = 5
START_RADIUS = 15

ROUND_TIME = 120

MASS_LOSS_TIME = 7

episodes_count = 1000

LOGGING = True
W, H = 1024, 640

HOST_NAME = socket.gethostname()
SERVER_IP = socket.gethostbyname(HOST_NAME)

# try to connect to server
try:
    S.bind((SERVER_IP, PORT))
except socket.error as e:
    print(str(e))
    print("[SERVER] Server could not start")
    quit()

S.listen()  # listen for connections

print(f"[SERVER] Server Started with local ip {SERVER_IP}")

# dynamic variables
players = {}
balls = []
traps = []
connections = 0

_id = 0
colors = [(255,0,0), (255, 128, 0), (255,255,0), (128,255,0),(0,255,0),(0,255,128),(0,255,255),(0, 128, 255), (0,0,255), (0,0,255), (128,0,255),(255,0,255), (255,0,128),(128,128,128)]
start = False
stat_time = 0
game_time = "Starting Soon"
nxt = 1



def reset_game():
	global players, balls, traps, start, start_time, game_time, nxt, episodes_count
	print(f"[GAME] Resetting game. Starting episode {episodes_count + 1}")


	# Reset all players
	for pid in players:
		print(f"PLAYER BEFORE RESET {players[pid]['x']}  {players[pid]['y']}")
		players[pid]["score"] = 0
		players[pid]["reward"] = 0.0
		players[pid]["x"], players[pid]["y"] = get_start_location(players)
		players[pid]["alive"] = True
		print(f"PLAYER {pid} reset new coordinates {players[pid]['x']} {players[pid]['y']}")

	# Clear and recreate balls and traps
	balls.clear()
	traps.clear()
	create_balls(balls, random.randrange(200, 250))
	create_traps(traps, random.randrange(10, 15))

	# Reset time and flags
	start = True
	start_time = time.time()
	game_time = 0
	nxt = 1
	episodes_count -= 1

# FUNCTIONS
def check_collision_balls(players, balls):
	"""
	checks if any of the player have collided with any of the balls or traps

	:param players: a dictonary of players
	:param balls: a list of balls
	:return: None
	"""
	for player in players:
		p = players[player]
		x = p["x"]
		y = p["y"]
		for ball in balls:

			# ball coordinates
			bx = ball[0]
			by = ball[1]

			dis_ball = math.sqrt((x - bx)**2 + (y-by)**2)
			if dis_ball <= START_RADIUS + p["score"]:
				p["score"] = p["score"] + 1
				p["reward"] = 2
				balls.remove(ball)


def check_collision_traps(players, traps):
	"""
	checks if any of the player have collided with any of the balls or traps

	:param players: a dictonary of players
	:param traps: a list of traps
	:return: None
	"""
	for player in players:
		p = players[player]
		x = p["x"]
		y = p["y"]
		for ball,trap in zip(balls, traps):

			# trap coordinates
			tx = trap[0]
			ty = trap[1]

			dis_trap = math.sqrt((x - tx)**2 + (y-ty)**2)
			if dis_trap <= START_RADIUS + p["score"]:
				p["score"] = p["score"] + 1
				players[player]["score"] = 0
				players[player]["reward"] = -100
				players[player]["x"], players[player]["y"] = get_start_location(players)
				players[player]["alive"] = False


def player_collision(players):
	"""
	checks for player collision and handles that collision

	:param players: dict
	:return: None
	"""
	sort_players = sorted(players, key=lambda x: players[x]["score"])
	for x, player1 in enumerate(sort_players):
		for player2 in sort_players[x+1:]:
			p1x = players[player1]["x"]
			p1y = players[player1]["y"]

			p2x = players[player2]["x"]
			p2y = players[player2]["y"]

			dis = math.sqrt((p1x - p2x)**2 + (p1y-p2y)**2)
			if dis < players[player2]["score"] - players[player1]["score"]*0.85:
				players[player2]["score"] = players[player2]["score"] + players[player1]["score"]
				players[player2]["reward"] = 200
				players[player1]["score"] = 0
				players[player1]["reward"] = -200
				players[player1]["alive"] = False
				players[player1]["x"], players[player1]["y"] = get_start_location(players)

				print(f"[GAME] " + players[player2]["name"] + " ATE " + players[player1]["name"])


def create_balls(balls, n):
	"""
	creates orbs/balls on the screen

	:param balls: a list to add balls/orbs to
	:param n: the amount of balls to make
	:return: None
	"""
	for i in range(n):
		while True:
			stop = True
			x = random.randrange(0,W)
			y = random.randrange(0,H)
			for player in players:
				p = players[player]
				dis = math.sqrt((x - p["x"])**2 + (y-p["y"])**2)
				if dis <= START_RADIUS + p["score"]:
					stop = False
			if stop:
				break

		balls.append((x,y, random.choice(colors)))


def create_traps(traps, n):
	"""
	creates traps on the screen

	:param traps: a list to add traps to
	:param n: the amount of traps to make
	:return: None
	"""
	for i in range(n):
		while True:
			stop = True

			x = random.randrange(0,W)
			y = random.randrange(0,H)

			for player in players:
				p = players[player]
				dis = math.sqrt((x - p["x"])**2 + (y-p["y"])**2)
				if dis <= START_RADIUS + p["score"]:
					stop = False
			if stop:
				break

		traps.append((x,y, (0,0,0)))


def get_start_location(players):
	"""
	picks a start location for a player based on other player
	locations. It wiill ensure it does not spawn inside another player

	:param players: dict
	:return: tuple (x,y)
	"""
	while True:
		stop = True
		x = random.randrange(0, W)
		y = random.randrange(0, H )
		for player in players:
			p = players[player]
			dis = math.sqrt((x - p["x"])**2 + (y-p["y"])**2)
			if dis <= START_RADIUS + p["score"]:
				stop = False
				break
		if stop:
			break
	return (x,y)


def threaded_client(conn, _id):
	"""
	runs in a new thread for each player connected to the server

	:param con: ip address of connection
	:param _id: int
	:return: None
	"""
	global connections, players, balls, traps, game_time, nxt, start, episodes_count

	current_id = _id

	# recieve a name from the client
	data = conn.recv(16)
	name = data.decode("utf-8")
	print("[LOG]", name, "connected to the server.")

	# Setup properties for each new player
	color = colors[current_id]
	x, y = get_start_location(players)
	players[current_id] = {"x":x, "y":y,"color":color,"score":0,"name":name, "alive": True, "episode": episodes_count, "reward": 0}

	# pickle data and send initial info to clients
	conn.send(str.encode(str(current_id)))

	# server will recieve basic commands from client
	# it will send back all of the other clients info
	'''
	commands start with:
	move
	jump
	get
	id - returns id of client
	'''
	while True:

		if start:
			game_time = round(time.time()-start_time)

			# if the game time passes the round time the game will stop
			if game_time >= ROUND_TIME:
				start = False
				if episodes_count > 0:
					reset_game()
				else:
					global LOGGING
					LOGGING = False
			else:
				if game_time // MASS_LOSS_TIME == nxt:
					nxt += 1
					print(f"[GAME] {name}'s Mass depleting")

			# checks whether all players are dead
			all_dead = all(not players[pid]["alive"] for pid in players)
			if all_dead:
				reset_game()
		try:
			# Recieve data from client
			data = conn.recv(32)

			if not data:
				break

			data = data.decode("utf-8")
			#print("[DATA] Recieved", data, "from client id:", current_id)

			# look for specific commands from recieved data
			if data.split(" ")[0] == "move":
				split_data = data.split(" ")
				x = int(split_data[1])
				y = int(split_data[2])
				players[current_id]["x"] = x
				players[current_id]["y"] = y
				players[current_id]["reward"] = 0

				# only check for collison if the game has started
				if start:
					check_collision_balls(players, balls)
					check_collision_traps(players, traps)
					player_collision(players)

				# if the amount of balls is less than 150 create more
				if len(balls) < 150:
					create_balls(balls, random.randrange(100,150))
					print("[GAME] Generating more orbs")

				send_data = pickle.dumps((balls,traps,players, game_time, episodes_count))

			elif data.split(" ")[0] == "id":
				send_data = str.encode(str(current_id))  # if user requests id then send it

			elif data.split(" ")[0] == "jump":
				send_data = pickle.dumps((balls,traps,players, game_time, episodes_count))
			else:
				# any other command just send back list of players
				send_data = pickle.dumps((balls,traps,players, game_time, episodes_count))

			# send data back to clients
			conn.send(send_data)

		except Exception as e:
			print(e)
			break  # if an exception has been reached disconnect client

		time.sleep(0.001)

	# When user disconnects	
	print("[DISCONNECT] Name:", name, ", Client Id:", current_id, "disconnected")

	connections -= 1 
	del players[current_id]  # remove client information from players list
	conn.close()  # close connection


# MAINLOOP

# setup level with balls
create_balls(balls, random.randrange(200,250))
create_traps(traps, random.randrange(10,15))

print("[GAME] Setting up level")
print("[SERVER] Waiting for connections")

# Keep looping to accept new connections
while LOGGING:
	host, addr = S.accept()
	print("[CONNECTION] Connected to:", addr)

	# start game when a client on the server computer connects
	if addr[0] == SERVER_IP and not(start):
		start = True
		start_time = time.time()
		print("[STARTED] Game Started")

	# increment connections start new thread then increment ids
	connections += 1
	start_new_thread(threaded_client,(host,_id))
	_id += 1

# when program ends
print("[SERVER] Server offline")
