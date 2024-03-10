# def get_sum(stack_g, stack_cur):
#     r = 0
#     l = min(len(stack_g), len(stack_cur))
#     for i in range(l):
#         if stack_g[i] == stack_cur[i]:
#             print(stack_cur, stack_g, i)
#             r += 1
#         else:
#             break
#     return r
#
#
# current = sorted([[3, 4, 6, 7, 5, 1], [2], [8]])
# goal = sorted([[3], [7], [6, 2, 5, 1], [6], [8]])
# res = 0
# for stack_goal in goal:
#     first_block_goal = list(reversed(stack_goal))[0]
#     for stack_current in current:
#         if first_block_goal in stack_current:
#             res += get_sum(list(reversed(stack_goal)), list(reversed(stack_current)))
#
# print(res)


def heuristic(self, goal_):
    self_state = sorted(self.get_state())
    goal_state = sorted(goal_.get_state())
    num_corr_boxs = 0
    for stack_goal in goal_state:
        first_block_goal = list(reversed(stack_goal))[0]
        for stack_current in self_state:
            if first_block_goal in stack_current:
                num_corr_boxs += get_sum(list(reversed(stack_goal)), list(reversed(stack_current)))


def get_sum(stack_goal, stack_current):
    sum_boxs = 0
    len_stack = min(len(stack_goal), len(stack_current))
    for i in range(len_stack):
        if stack_goal[i] == stack_current[i]:
            sum_boxs += 1
        else:
            break
    return sum_boxs








