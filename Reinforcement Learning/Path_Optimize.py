'''

Path Optimization Puzzle

'''

import numpy as np
import random

random.seed(1)



square = [[1, 4, 2, 4, 4],
          [3, 3, 2, 3, 1],
          [5, 1, 3, 5, 2],
          [3, 2, 4, 3, 4],
          [2, 3, 5, 4, 1]]
square = np.array(square)

path = []
min = 0
max = 4


# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

def step(action, state):
    # next state
    v, h = state
    if action == 0:
        v += 1
    else:
        h += 1
    env = [v, h]
    next_state = env[0] * 5 + env[1]

    # reward
    reward = square[v][h]

    # done
    done = False
    if v == 4 and h == 4:
        done = True

    return (next_state, reward, done)

q_table = np.zeros([25, 2])
popular = []
l = []
large = 0
best = []

for i in range(1, 100000):
    env = [0, 0]
    state = env[0] * 5 + env[1]

    done = False

    total = 1
    count = 0
    popular = []

    while not done:
        if random.random() < epsilon:
            action = random.randrange(2)
        else:
            action = np.argmax(q_table[state])

        #next_state, reward, done, info = step(action)
        env = [state // 5, state % 5]
        if (action == 0 and env[0] + 1 > 4) or (action == 1 and env[1] + 1 > 4):
            continue

        next_state, reward, done = step(action, env)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        #if reward == -10:
        #    penalties += 1

        state = next_state
        #epochs += 1

        total += reward
        count += 1

        popular.append(env)

    if total > large:
        large = total
        best = popular
    l.append(count)
    print('episode:', i, 'reward:', total)

print(q_table)
print(l)
print(popular)
print('best:', large, best)







