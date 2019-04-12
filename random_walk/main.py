import random
import math


class Example():
    def __init__(self, path, reward):
        self.path = path
        self.reward = reward

    def __str__(self):
        return 'path: {} \t\t reward: {}'.format(self.path, self.reward)

    @classmethod
    def generate_example(cls):
        path = [2]
        while 'TERM' not in idx_to_state[path[-1]]:
            if random.random() > 0.5:
                path.append(path[-1] + 1)
            else:
                path.append(path[-1] - 1)
        path = [idx_to_state[i] for i in path]
        reward = 1 if path[-1] == 'RTERM' else 0
        return cls(path, reward)


class MarkovProcess():
    def __init__(self, states):
        self.states = states
        self.values = self.reset_values()

    def reset_values(self):
        values = {s: random.random() for s in states}
        for val in values:
            if 'TERM' in val:
                values[val] = 0
        return values

    def move_right(self, state):
        return idx_to_state[state_to_idx[state] + 1]

    def move_left(self, state):
        return idx_to_state[state_to_idx[state] - 1]

    def value_prediction(self):
        raise NotImplementedError



class MarkovRewardProcess(MarkovProcess):
    def __init__(self, states):
        super().__init__(states)

    def expected_return(self, s, values):
        right_state = self.move_right(s)
        reward = 1 if right_state == 'RTERM' else 0
        left_state = self.move_left(s) # no reward
        ret = 0.5 * (reward + values[right_state]) +\
            0.5 * (0 + values[left_state])
        return ret

    def value_prediction(self):
        values = self.reset_values()
        while True:
            delta = 0.
            for s in self.states:
                if 'TERM' in s: continue
                old_v = values[s]
                values[s] = self.expected_return(s, values)
                delta = max(delta, abs(old_v - values[s]))
            if delta < 1e-4: break
        return values


class MonteCarlo(MarkovProcess):
    def __init__(self, states, alpha):
        super().__init__(states)
        self.alpha = alpha

    def value_prediction(self, n):
        values = self.reset_values()
        examples = [Example.generate_example() for _ in range(n)]
        for ex in examples:
            visited = []
            ret = ex.reward
            for s in ex.path[-2::-1]:
                if s not in visited:
                    values[s] = values[s] + self.alpha * (ret - values[s])
        return values


class TemporalDifference(MarkovProcess):
    def __init__(self, states, alpha):
        super().__init__(states)
        self.alpha = alpha

    def value_prediction(self, n):
        values = self.reset_values()
        examples = [Example.generate_example() for _ in range(n)]
        for ex in examples:
            for s, next_s in zip(ex.path[:-1], ex.path[1:]):
                reward = 1 if next_s == 'RTERM' else 0
                values[s] = values[s] +\
                    self.alpha * (reward + values[next_s]- values[s])
        return values


def root_mean_squared(values1, values2):
    ret = 0.
    for val1, val2 in zip(values1.values(), values2.values()):
        ret += math.sqrt((val1 - val2) ** 2)
    return ret / math.sqrt(len(values1))


if __name__ == "__main__":

    states = ['LTERM', 'A', 'B', 'C', 'D', 'E', 'RTERM']
    idx_to_state = {idx: state for idx, state in zip(range(-1, 6, 1), states)}
    state_to_idx = {state: idx for idx, state in idx_to_state.items()}
    num_examples = [0, 10, 100]

    MRP = MarkovRewardProcess(states)
    values_MRP = MRP.value_prediction()
    print('values predicted by MRP: ', values_MRP)

    MC = MonteCarlo(states, alpha=0.03)
    for n in num_examples:
        values_MC = MC.value_prediction(n)
        print('values predicted by MC, with n={}'.format(n), values_MC)
        print(root_mean_squared(values_MRP, values_MC))

    TD = TemporalDifference(states, alpha=0.03)
    for n in num_examples:
        values_TD = TD.value_prediction(n)
        print('values predicted by TD, with n={}'.format(n), values_TD)
        print(root_mean_squared(values_MRP, values_TD))




