from blockworld import BlockWorld
import numpy as np
from queue import PriorityQueue


class BlockWorldHeuristic(BlockWorld):

	def __init__(self, num_blocks=5, state=None):
		BlockWorld.__init__(self, num_blocks, state)
		self.cost = 0
		self.history = (None, None)

	def heuristic(self, goal_):
		self_state = list(self.get_state())
		goal_state = list(goal_.get_state())
		hamming_distance = sum(a != b for a, b in zip(sorted(self_state), sorted(goal_state)))
		misplaced_blocks = len(set(self_state) - set(goal_state))
		current_positions = {block: (i, j) for i, stack in enumerate(self_state) for j, block in enumerate(stack)}
		target_positions = {block: (i, j) for i, stack in enumerate(goal_state) for j, block in enumerate(stack)}
		chebyshev_distance = 0
		manhattan_distance = 0
		for block, current_pos in current_positions.items():
			target_pos = target_positions[block]
			chebyshev_distance += max(abs(current_pos[0] - target_pos[0]), abs(current_pos[1] - target_pos[1]))
			manhattan_distance += abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
		return hamming_distance + chebyshev_distance + manhattan_distance


class AStar:

	def __init__(self):
		self.closed_nodes = dict()

	def search(self, init, goal_):
		priority_queue = PriorityQueue()
		priority_queue.put((0, init))
		while not priority_queue.empty():
			_, current = priority_queue.get()
			action, prev = current.history
			if current.get_state() == goal_.get_state():
				self.closed_nodes[current] = (action, prev)
				return self.reconstruct_path(init, current)
			if current in self.closed_nodes:
				continue
			else:
				self.closed_nodes[current] = action, prev
			for action, neighbor in current.get_neighbors():
				next_ = BlockWorldHeuristic(state=str(neighbor))
				next_.history = (action, current)
				next_.cost = current.cost + 0.01
				priority = next_.heuristic(goal_) + next_.cost
				priority_queue.put((priority, next_))
		return []

	def reconstruct_path(self, init, last):
		action, prev = self.closed_nodes[last]
		path_to_goal = [action]
		while prev != init:
			action, prev = self.closed_nodes[prev]
			path_to_goal.append(action)
		return list(reversed(path_to_goal))

