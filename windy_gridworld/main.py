import random


def get_states(row=7, col=10):
# states = [(0,0), (0,1), ..., (6,9)]
    return [(i, j) for i in range(row) for j in range(col)]

def get_actions(move):
# actions = {'East': 'East', 'West': 'West', 'South': 'South', 'North': 'North'}
    if move == 'standard':
        possible_actions = ['East', 'West', 'South', 'North']
        act_in_idx = {'East': (0, 1), 'West': (0, -1), 'South': (1, 0), 'North': (-1, 0)}
    elif move == 'king':
        possible_actions = ['NN', 'NE', 'EE', 'SE', 'SS', 'SW', 'WW', 'NW']
        act_in_idx = {'NN': (-1, 0), 'NE': (-1, 1), 'EE': (0, 1), 'SE': (1, 1),
                      'SS': (1, 0), 'SW': (1, -1), 'WW': (0, -1), 'NW': (-1, -1)}
    actions = {a: a for a in possible_actions}
    return actions, act_in_idx

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

class SARSA():
    def __init__(self, alpha, gamma=1):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, S, A, R, S_prime, A_prime, Q):
        Q[S][A] += self.alpha * (R + self.gamma * Q[S_prime][A_prime] - Q[S][A])

class GreedyPolicy():
    def __init__(self, actions, eps=None):
        self.actions = actions
        self.eps = eps

    def __call__(self, Q, state, is_test=False):
        if is_test: self.eps = 0
        if random.random() < self.eps: # explore
            return random.choice(list(self.actions))
        else: # act greedily
            actions = Q[state]
            max_q = max([a for a in actions.values()])
            for action, value in actions.items():
                if value == max_q:
                    return action

class WindyGridWorld():
    def __init__(self, states, wind):
        self.states = states
        self.wind = wind
        self.reward = -1

    def _wind(self, state):
        return (self.wind[state[1]], 0)

    def _corner_check(self, state):
        row, col = state
        max_row, max_col = max([s[0] for s in self.states]),\
                           max([s[1] for s in self.states])
        row = max(0, row)
        row = min(row, max_row)
        col = max(0, col)
        col = min(col, max_col)
        return (row, col)

    def transition(self, state, action_idx):
        S_prime = elem_sum(state, self._wind(state))
        S_prime = elem_sum(S_prime, action_idx)
        S_prime = self._corner_check(S_prime)
        return self.reward, S_prime

def test(S, policy, environment, q):
    A = policy(q, S, is_test=True)
    actions_taken = [A]
    path = [S]
    while S != goal:
        _, S = environment.transition(S, act_in_idx[A])
        A = policy(q, S, is_test=True)
        actions_taken.append(A)
        path.append(S)
    return actions_taken[:-1], path

def elem_sum(a, b):
    return tuple(map(lambda x,y: x+y, a, b))

if __name__ == "__main__":
    row = 7
    col = 10
    start = (3, 0)
    goal = (3, 7)
    wind = (0, 0, 0, -1, -1, -1, -2, -2, -1, 0)
    assert len(wind) == col, 'specify wind for every column'

    states = get_states(row, col)
    actions, act_in_idx = get_actions('standard')
    env = WindyGridWorld(states, wind)
    Q = initialize_Q(states, actions)
    eps_greedy = GreedyPolicy(actions, eps=0.1)
    sarsa = SARSA(alpha=0.5)

    for i in range(1000): # stop when?
        print('episode num {}'.format(i))
        S = initialize_S(states, fixed=start)
        A = eps_greedy(Q, S)
        step_taken = 0
        while S != goal:
            step_taken += 1
            R, S_prime = env.transition(S, act_in_idx[A])
            A_prime = eps_greedy(Q, S_prime)
            sarsa(S, A, R, S_prime, A_prime, Q) # Q updated in-place
            S = S_prime
            A = A_prime
        print('step taken for this episode: {}'.format(step_taken))

    actions_taken, path= test(start, eps_greedy, env, Q)
    print('number of actions :', len(actions_taken))
    print('actions taken: ', actions_taken)
    print('path: ', path)

