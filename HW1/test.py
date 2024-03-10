import numpy as np


def chebyshev_distance(state, target):
    current_positions = {block: (i, j) for i, stack in enumerate(state) for j, block in enumerate(stack)}
    target_positions = {block: (i, j) for i, stack in enumerate(target) for j, block in enumerate(stack)}
    chebyshev_distance = 0
    for block, current_pos in current_positions.items():
        target_pos = target_positions[block]
        chebyshev_distance += max(abs(current_pos[0] - target_pos[0]), abs(current_pos[1] - target_pos[1]))
    return chebyshev_distance



# Пример использования
current_state = [[1, 2, 3], [4]]
target_state = [[4, 2], [3, 1]]

result = chebyshev_distance(current_state, target_state)
print(result)
