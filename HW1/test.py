def heuristic(goal_):
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
    return chebyshev_distance


print(heuristic())