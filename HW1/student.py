from blockworld import BlockWorld
from copy import deepcopy
from queue import PriorityQueue


class BlockWorldHeuristic(BlockWorld):

	def __init__(self, num_blocks=5, state=None):
		BlockWorld.__init__(self, num_blocks, state)

	def heuristic(self, goal_):
		self_state = self.get_state()
		goal_state = goal_.get_state()
		return 0.


class AStar:

	def __init__(self):
		self.closed_nodes, self.histories, self.costs, self.priorities = dict(), dict(), dict(), dict()

	def search(self, init, goal_):
		priority_queue = PriorityQueue()
		self.histories[init] = (None, None)
		self.costs[init] = 0
		self.priorities[init] = 0
		priority_queue.put((self.costs[init], init))

		while not priority_queue.empty():
			priority, current = priority_queue.get()
			action, prev = self.histories[current]
			if current.get_state() == goal_.get_state():
				self.closed_nodes[current] = (action, prev)
				return self.reconstruct_path(init, current)
			if current in self.closed_nodes:
				continue
			else:
				self.closed_nodes[current] = action, prev
			for action, neighbor in current.get_neighbors():
				next_ = BlockWorldHeuristic(state=str(neighbor))
				self.histories[next_] = (action, current)
				self.costs[next_] = self.costs[current] + 1
				self.priorities[next_] = 0
				priority_queue.put((self.costs[next_], next_))
		return []

	def reconstruct_path(self, init, last):
		action, prev = self.closed_nodes[last]
		path_to_goal = [action]
		while prev != init:
			action, prev = self.closed_nodes[prev]
			path_to_goal.append(action)
		return list(reversed(path_to_goal))


if __name__ == '__main__':
	# Here you can test your algorithm. You can try different N values, e.g. 6, 7.
	N = 5

	start = BlockWorldHeuristic(N)
	goal = BlockWorldHeuristic(N)

	print("Searching for a path:")
	print(f"{start} -> {goal}")
	print()

	astar = AStar()
	path = astar.search(start, goal)

	if path is not None:
		print("Found a path:")
		print(path)

		print("\nHere's how it goes:")

		s = start.clone()
		print(s)

		for a in path:
			s.apply(a)
			print(s)

	else:
		print("No path exists.")

	print("Total expanded nodes:", BlockWorld.expanded)
