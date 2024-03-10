from blockworld import BlockWorld
import copy
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
		current_positions = {block: (i, j) for i, stack in enumerate(self_state) for j, block in enumerate(stack)}
		target_positions = {block: (i, j) for i, stack in enumerate(goal_state) for j, block in enumerate(stack)}
		chebyshev_distance = 0
		manhattan_distance = 0
		diffs = 0
		for block, current_pos in current_positions.items():
			target_pos = target_positions[block]
			chebyshev_distance += max(abs(current_pos[0] - target_pos[0]), abs(current_pos[1] - target_pos[1]))
			manhattan_distance += abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
			if target_pos != current_pos:
				diffs += 1
		return 1/3*manhattan_distance


class AStar:

	def reconstruct_path(self, visited, start, goal):
		action, prev_state, _ = visited[goal]
		path = [action]
		while prev_state != start:
			action, prev_state, _ = visited[prev_state]
			path.append(action)
		return list(reversed(path))

	def search(self, start, goal):
		opened = PriorityQueue()
		opened.put((0, start))
		closed = dict()
		closed[start] = (None, 0, 0)
		while not opened.empty():
			_, current = opened.get()
			if current == goal:
				return self.reconstruct_path(closed, start, goal)
			for action, neighbor in current.get_neighbors():
				new_cost = closed[current][2] + 1
				if neighbor not in closed or new_cost < closed[neighbor][2]:
					priority = neighbor.heuristic(goal) + new_cost
					closed[neighbor] = (action, current, new_cost)
					opened.put((priority, neighbor))
