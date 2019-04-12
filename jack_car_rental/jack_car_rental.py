
# coding: utf-8

# # Jack's Car Rental

# 120180237 안휘진

# In[2]:


from math import pow, factorial, exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[3]:


CREDIT_RENT = 10
COST_MOVE = -2

MAX_CARS = 20
MOVE_LIMIT = 5

LAMBDA_A_RENTAL = 3
LAMBDA_A_RETURN = 3
LAMBDA_B_RENTAL = 4
LAMBDA_B_RETURN = 2
POISSON_UPPER_LIMIT = 11

THETA = 1e-4
GAMMA = 0.9


# In[4]:


states = [(i, j) for i in range(MAX_CARS+1) for j in range(MAX_CARS+1)]
values = {k:0 for k in states}
policy = {k:0 for k in states}
actions = range(-MOVE_LIMIT, MOVE_LIMIT+1, 1)


# In[5]:


poisson_cache = dict()
def p_poisson(lam, n):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = exp(-lam) * pow(lam, n) / factorial(n)
    return poisson_cache[key]


# In[6]:


def expected_return(s, a, states, values):
    # how to prevent from moving 2 when only 1 available?
    s_after_action = (min(s[1] - a, MAX_CARS), min(s[1] + a, MAX_CARS))
    returns = abs(a) * COST_MOVE

    for rent_a in range(POISSON_UPPER_LIMIT):
        for rent_b in range(POISSON_UPPER_LIMIT):
            real_rent_a = min(s_after_action[0], rent_a)
            real_rent_b = min(s_after_action[1], rent_b)
            s_after_rental = (s_after_action[0] - real_rent_a,
                              s_after_action[1] - real_rent_b)
            reward = (real_rent_a + real_rent_b) * CREDIT_RENT
            # prob is calculated with rent_a, not real_rent_a
            _prob = p_poisson(LAMBDA_A_RENTAL, rent_a) *            p_poisson(LAMBDA_B_RENTAL, rent_b)

            for ret_a in range(POISSON_UPPER_LIMIT):
                for ret_b in range(POISSON_UPPER_LIMIT):
                    s_after_return = (min((s_after_rental[0] + ret_a), MAX_CARS),
                                      min((s_after_rental[1] + ret_b), MAX_CARS))
                    prob = _prob * p_poisson(LAMBDA_A_RETURN, ret_a) *                    p_poisson(LAMBDA_B_RETURN, ret_b)
                    returns += prob * (reward + GAMMA * values[s_after_return])
    return returns


# In[7]:


def policy_evaluation(states, values, policy):
    while True:
        delta = 0.
        for s in states:
            old_v = values[s]
            a = policy[s]
            values[s] = expected_return(s, a, states, values) # inplace algorithm
            delta = max(delta, abs(old_v - values[s]))
        #print('value change: {}'.format(delta))
        if delta < THETA: 
            print('value converged!')
            break


# In[8]:


def policy_improvement(states, values, policy):
    policy_changed = []
    for s in states:
        old_action = policy[s]
        policy[s] = actions[np.argmax([expected_return(s, a, states, values)
                                       for a in actions])]
        policy_changed.append(True if old_action != policy[s] else False)
    return sum(policy_changed) 


# In[9]:


def policy_iteration(states, values, policy):
    policies = []
    policy_changed = True
    while policy_changed:
        policy_evaluation(states, values, policy)
        policy_changed = policy_improvement(states, values, policy)
        policies.append(policy.copy())
        print('=== policy changed at {} states in {}th iteration ==='.format(
            policy_changed, len(policies)))
    print('DONE!')
    return values, policies


# In[10]:


def draw_policy_contour(policy, i):
    policy_matrix = np.zeros((MAX_CARS+1, MAX_CARS+1))
    for idx in policy:
        policy_matrix[idx] = policy[idx]
    plt.title('policy at {}th iteration'.format(i))
    plt.imshow(policy_matrix, extent=[0, 20, 0, 20], origin='lower', cmap='RdGy')
    plt.show()


# In[11]:


def draw_policy_contour(policy, i):
    policy_matrix = np.zeros((MAX_CARS+1, MAX_CARS+1))
    for idx in policy:
        policy_matrix[idx] = policy[idx]
    plt.title('policy at {}th iteration'.format(i))
    plt.imshow(policy_matrix, extent=[0, 20, 0, 20], origin='lower', cmap='RdGy')
    plt.show()


# In[12]:


def draw_value_surface(values):
    value_matrix = np.zeros((MAX_CARS+1, MAX_CARS+1))
    for idx in values:
        value_matrix[idx] = values[idx]
        
    fig  = plt.figure()
    fig.suptitle('value surface', fontsize=16)
    ax = Axes3D(fig)
    x,y = np.meshgrid(range(MAX_CARS+1), range(MAX_CARS+1))
    ax.plot_surface(x, y, value_matrix, rstride=1, cstride=1, cmap='summer')
    plt.show()


# In[11]:


v_optimized, policies = policy_iteration(states, values, policy)


# In[12]:


for i, policy in enumerate(policies, 1):
    draw_policy_contour(policy, i)


# In[21]:


draw_value_surface(values)


# In[13]:


def draw_value_surface(values):        
    fig  = plt.figure()
    fig.suptitle('value surface', fontsize=16)
    ax = Axes3D(fig)
    x,y = np.meshgrid(range(MAX_CARS+1), range(MAX_CARS+1))
    ax.plot_surface(x, y, value_matrix, rstride=1, cstride=1, cmap='summer')
    plt.show()


# In[15]:


def draw_policy_contour(policy_matrix):
    plt.title('policy')
    plt.imshow(policy_matrix, extent=[0, 20, 0, 20], origin='lower', cmap='RdGy')
    plt.show()
sample_policy_matrix = np.zeros((MAX_CARS+1, MAX_CARS+1))
draw_policy_contour(sample_policy_matrix)

