from blockworld import BlockWorld
import copy
from queue import PriorityQueue


class BlockWorldHeuristic(BlockWorld):

	def __init__(self, num_blocks=5, state=None):
		BlockWorld.__init__(self, num_blocks, state)
		self.cost = 0
		self.history = (None, None)

	def heuristic(self, goal_):
		self_state = sorted(self.get_state())
		goal_state = sorted(goal_.get_state())
		num_corr_boxs = 0
		for stack_goal in goal_state:
			first_block_goal = list(reversed(stack_goal))[0]
			for stack_current in self_state:
				if first_block_goal in stack_current:
					num_corr_boxs += self.get_sum(list(reversed(stack_goal)), list(reversed(stack_current)))
		return -num_corr_boxs


	def get_sum(self, stack_goal, stack_current):
		sum_boxs = 0
		len_stack = min(len(stack_goal), len(stack_current))
		for i in range(len_stack):
			if stack_goal[i] == stack_current[i]:
				sum_boxs += 1
			else:
				break
		return sum_boxs


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
			if current.get_state() == goal.get_state():
				return self.reconstruct_path(closed, start, goal)
			for action, neighbor in current.get_neighbors():
				new_cost = closed[current][2] + 1
				if neighbor not in closed or new_cost < closed[neighbor][2]:
					priority = neighbor.heuristic(goal) + new_cost
					closed[neighbor] = (action, current, new_cost)
					opened.put((priority, neighbor))
