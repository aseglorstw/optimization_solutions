import numpy as np

from blockworld import BlockWorldEnv
import random

class QLearning():
	# don't modify the methods' signatures!
	def __init__(self, env: BlockWorldEnv):
		self.env = env
		self.q_function = dict()
		self.episode = 1

	def get_alpha(self):
		alpha = 0.1
		return alpha

	def get_max_q_value(self, state, goal_state):
		possible_actions = state.get_actions()
		q_values = np.array([self.q_function.get((state, goal_state, action), 0) for action in possible_actions])
		return np.max(q_values)

	def train(self):
		gamma = 0.9
		while self.episode <= 8001:
			is_terminate_state = False
			current_state, goal_state = self.env.reset()
			while not is_terminate_state:
				action = self.act((current_state, goal_state))
				(new_state, _), reward, is_terminate_state = self.env.step(action)
				max_q = self.get_max_q_value(new_state, goal_state)
				q_old = self.q_function.get((current_state, goal_state, action), 0)
				self.q_function[(current_state, goal_state, action)] \
					= q_old + self.get_alpha() * (reward + gamma * max_q - q_old)
				current_state = new_state
			self.episode += 1

	def act(self, state):
		epsilon = 0.1
		current_state, goal_state = state
		possible_actions = current_state.get_actions()
		q_values = np.array(
			[self.q_function.get((current_state, goal_state, action), 0) for action in possible_actions])
		if np.random.rand() > epsilon:
			return possible_actions[np.argmax(q_values)]
		return possible_actions[np.random.randint(len(possible_actions))]


if __name__ == '__main__':
	# Here you can test your algorithm. Stick with N <= 4
	N = 4

	env = BlockWorldEnv(N)
	qlearning = QLearning(env)

	# Train
	qlearning.train()

	# Evaluate
	test_env = BlockWorldEnv(N)

	test_problems = 10
	solved = 0
	avg_steps = []

	for test_id in range(test_problems):
		s = test_env.reset()
		done = False

		print(f"\nProblem {test_id}:")
		print(f"{s[0]} -> {s[1]}")

		for step in range(50): 	# max 50 steps per problem
			a = qlearning.act(s)
			s_, r, done = test_env.step(a)

			print(f"{a}: {s[0]}")

			s = s_

			if done:
				solved += 1
				avg_steps.append(step + 1)
				break

	avg_steps = sum(avg_steps) / len(avg_steps)
	print(f"Solved {solved}/{test_problems} problems, with average number of steps {avg_steps}.")