import random
from collections import defaultdict

import matplotlib.pyplot as plt


def get_states(row=6, col=10):
# states = [(0,0), (0,1), ..., (6,9)]
    return [(i, j) for i in range(row) for j in range(col)]

def get_actions(move):
# actions = {'East': 'East', 'West': 'West', 'South': 'South', 'North': 'North'}
    possible_actions = ['East', 'West', 'South', 'North']
    act_to_idx = {'East': (0, 1), 'West': (0, -1), 'South': (1, 0), 'North': (-1, 0)}
    actions = {a: a for a in possible_actions}
    return actions, act_to_idx

def initialize_Q(states, actions):
#   Q = {
#        '(0,0) : {'East': 0.3, 'West': 0.2, 'South': 0.1, 'North': 0.2},
#        ...
#        '(6,9) : {'East': 0.2, 'West': 0.1, 'South': 0.3, 'North': 0.9}
#       }
    return {state: {action: 0. for action in actions} for state in states}

def initialize_S(states, fixed=None):
    if fixed is not None: return fixed
    return random.choice(states)

class QLearning():
    def __init__(self, actions, alpha, gamma=1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, S, A, R, S_prime, Q):
        max_Q = max(Q[S_prime][a] for a in self.actions)
        Q[S][A] += self.alpha * (R + self.gamma * max_Q - Q[S][A])

class GreedyPolicy():
    def __init__(self, actions, eps=None):
        self.actions = actions
        self.eps = eps

    def __call__(self, Q, S, is_test=False):
        if is_test: self.eps = 0
        if random.random() < self.eps: # explore
            return random.choice(list(self.actions))
        else: # act greedily
            actions = Q[S]
            max_q = max([a for a in actions.values()])
            for action, value in actions.items():
                if value == max_q:
                    return action

class Maze():
    def __init__(self, states, goal, obstacles, act_to_idx):
        self.states = states
        self.goal = goal
        self.obstacles = obstacles
        self.act_to_idx = act_to_idx
        self.const_reward = 0
        self.goal_reward = 1

    def _corner_check(self, S):
        row, col = S
        max_row = max([state[0] for state in self.states])
        max_col = max([state[1] for state in self.states])
        row = min(max(0, row), max_row)
        col = min(max(0, col), max_col)
        return (row, col)

    def _transition(self, S, A):
        S_prime = elem_sum(S, self.act_to_idx[A])
        if S_prime in self.obstacles:
            return S
        else:
            return S_prime

    def __call__(self, S, A):
        S_prime = self._transition(S, A)
        S_prime = self._corner_check(S_prime)
        R = self.goal_reward if S_prime == self.goal else self.const_reward
        return R, S_prime

class Model():
    def __init__(self):
        self.history = dict()

    def learn(self, S, A, R, S_prime):
        self.history[(S, A)] = (R, S_prime)

    def __call__(self, S, A):
        return self.history[(S, A)]

def elem_sum(a, b):
    return tuple(map(lambda x,y: x+y, a, b))

def list_running_average(mu_list, x_list, k):
    if mu_list is None:
        mu_list = [0.] * len(x_list)

    def running_avg(mu, x, k):
        return mu + ((x - mu) / k)

    return [running_avg(mu, x, k) for mu, x in zip(mu_list, x_list)]

def plot_graph(results, filename='result.png'):
    fig = plt.figure(1, figsize=(12,8))
    plt.title('Maze with Dyna-Q', fontsize=20)
    plt.xlabel('Episodes', fontsize=15)
    plt.ylabel('Steps per episode', fontsize=15)
    for N, result in results.items():
        plt.plot(range(1, len(result)+1), result, label='{} planning steps'.format(N))
    plt.legend()
    fig.savefig(filename)
    print('plot saved at {}'.format(filename))

def experiment(N):
    row = 6
    col = 10
    start = (2, 0)
    goal = (0, 9)
    obstacles = [(1, 2), (2, 2), (2,3),
                 (0, 7), (1, 7), (2, 7),
                 (4, 5)]
    states = get_states(row, col)
    actions, act_to_idx = get_actions('standard')
    environment = Maze(states, goal, obstacles, act_to_idx)
    model = Model()
    Q = initialize_Q(states, actions)
    eps_greedy = GreedyPolicy(actions, eps=0.1)
    qlearning = QLearning(actions, alpha=0.1, gamma=0.95)

    steps_per_episode = list()
    for i in range(1, 51):
        observed_states = set()
        observed_actions = defaultdict(set)
        S = initialize_S(states, fixed=start)
        step=0
        while S != goal:
            observed_states.add(S)
            step+= 1
            A = eps_greedy(Q, S)
            observed_actions[S].add(A)
            R, S_prime = environment(S, A)
            qlearning(S, A, R, S_prime, Q) # Q updated in-place
            model.learn(S, A, R, S_prime)
            S = S_prime
            for n in range(N):
                S_model = random.choice(list(observed_states))
                A_model = random.choice(list(observed_actions[S_model]))
                R_model, S_prime_model = model(S_model, A_model)
                qlearning(S_model, A_model, R_model, S_prime_model, Q)
        steps_per_episode.append(step)
    return steps_per_episode

if __name__ == "__main__":
    N_list = [0, 5, 50]
    repetition = 30
    results = {N: [] for N in N_list}

    for N in N_list:
        average = None
        for k in range(1, repetition+1):
            result = experiment(N)
            average = list_running_average(average, result, k)
        results[N] = average

    for N, result in results.items():
        print('steps_per_episode({}): \n'.format(N), result)
    plot_graph(results)


