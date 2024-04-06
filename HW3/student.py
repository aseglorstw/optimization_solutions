import random, time
import numpy as np

import ox

class MCTSBot:
    def __init__(self, play_as: int, time_limit: float):
        self.play_as = play_as
        self.time_limit = time_limit * 0.9

    def play_action(self, board):
        # TODO: implement MCTS bot

        start_time = time.time()
        while (time.time() - start_time) < self.time_limit:
        	pass # do some thinking...

        return random.choice( list(board.get_actions()) )

if __name__ == '__main__':
	board = ox.Board(8)  # 8x8
	bots = [MCTSBot(0, 0.1), MCTSBot(1, 1.0)]

	# try your bot against itself
	while not board.is_terminal():
		current_player = board.current_player()
		current_player_mark = ox.MARKS_AS_CHAR[ ox.PLAYER_TO_MARK[current_player] ]

		current_bot = bots[current_player]
		a = current_bot.play_action(board)
		board.apply_action(a)

		print(f"{current_player_mark}: {a} -> \n{board}\n")