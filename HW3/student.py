import random, time
import ox
import numpy as np

#E greedy
# class UcbBandit:
#     def __init__(self, state):
#         self.actions = list(state.get_actions())
#         self.qs = np.zeros(len(self.actions))
#         self.visits = np.zeros(len(self.actions))
#         self.last_idx = None
#         self.c_uct = 2
#
#     def total_visits(self) -> int:
#         return np.sum(self.visits)
#
#     def select(self):
#         total_visits = self.total_visits()
#         ucb_vals = (
#             self.qs + self.c_uct * np.sqrt(np.log(total_visits)/self.visits)
#         )
#         self.last_idx = np.argmax(ucb_vals)
#         return self.actions[self.last_idx]
#
#     def update(self, value: float):
#         self.visits[self.last_idx] += 1
#         self.qs[self.last_idx] += (value - self.qs[self.last_idx]) / self.visits[self.last_idx]
#
#     def best_action(self):
#         return self.actions[np.argmax(self.qs)]

class EpsGreedyBandit:
    def __init__(self, state):
        self.epsilon = 0.2
        self.actions = list(state.get_actions())
        self.qs = np.zeros(len(self.actions))
        self.visits = np.zeros(len(self.actions))
        self.last_idx = None

    def select(self):
        if np.random.rand() < self.epsilon:
            self.last_idx = np.random.randint(len(self.actions))
        else:
            self.last_idx = np.argmax(self.qs)
        return self.actions[self.last_idx]

    def best_action(self):
        return self.actions[np.argmax(self.qs)]

    def update(self, value: float):
        self.visits[self.last_idx] += 1
        self.qs[self.last_idx] += (value - self.qs[self.last_idx]) / self.visits[self.last_idx]


class MCTSBot:
    def __init__(self, play_as: int, time_limit: float):
        self.play_as = play_as
        self.time_limit = time_limit * 0.9
        self.node_table = dict()

    def random_simulation(self, state):
        while not state.is_terminal():
            random_action = np.random.choice(list(state.get_actions()))
            state.apply_action(random_action)
        return state.get_rewards()

    def has_node(self, state):
        history_str = ','.join(map(str, state.history))
        return history_str in self.node_table

    def get_node(self, state):
        history_str = ','.join(map(str, state.history))
        return self.node_table[history_str]

    def make_node(self, state):
        history_str = ','.join(map(str, state.history))
        self.node_table[history_str] = EpsGreedyBandit(state)

    def best_action(self, state):
        return self.get_node(state).best_action()

    def select(self, state):
        trace = []
        while self.has_node(state):
            if state.is_terminal():
                break
            trace.append(state.clone())
            bandit = self.get_node(state)
            action = bandit.select()
            state.apply_action(action)
        return trace, state.clone()

    def backpropagate(self, trace, values) -> None:
        for state in reversed(trace):
            self.get_node(state).update(values[state.current_player()])

    def expand(self, state):
        self.make_node(state)

    def step(self, state):
        if not self.has_node(state):
            self.make_node(state)
        trace, last_state = self.select(state.clone())
        if last_state.is_terminal():
            values = last_state.get_rewards()
        else:
            self.expand(last_state)
            values = self.random_simulation(last_state.clone())
        self.backpropagate(trace, values)


    def play_action(self, board):
        start_time = time.time()
        while (time.time() - start_time) < self.time_limit:
            self.step(board)
        node = self.get_node(board)
        return node.best_action()


if __name__ == '__main__':
    board_0 = ox.Board(8)
    bots = [MCTSBot(0, 0.1), MCTSBot(1, 1.0)]
    while not board_0.is_terminal():
        current_player = board_0.current_player()
        current_player_mark = ox.MARKS_AS_CHAR[ox.PLAYER_TO_MARK[current_player]]

        current_bot = bots[current_player]
        a = current_bot.play_action(board_0)
        board_0.apply_action(a)
        print(f"{current_player_mark}: {a} -> \n{board_0}\n")
