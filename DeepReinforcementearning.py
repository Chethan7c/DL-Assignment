# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning
Refactored for reproducibility and modularity
"""

import numpy as np
import pylab as pl
import networkx as nx

# -------------------------------
# Reproducibility (ADDED)
# -------------------------------
np.random.seed(42)

# -------------------------------
# Graph definition
# -------------------------------
edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2),
         (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
         (8, 9), (7, 8), (1, 7), (3, 9)]

goal = 10

G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
pl.show()

# -------------------------------
# Reward matrix
# -------------------------------
MATRIX_SIZE = 11
M = np.matrix(np.ones((MATRIX_SIZE, MATRIX_SIZE))) * -1

for point in edges:
    if point[1] == goal:
        M[point] = 100
    else:
        M[point] = 0

    if point[0] == goal:
        M[point[::-1]] = 100
    else:
        M[point[::-1]] = 0

M[goal, goal] = 100
print("Reward Matrix:")
print(M)

# -------------------------------
# Q-learning setup
# -------------------------------
Q = np.matrix(np.zeros((MATRIX_SIZE, MATRIX_SIZE)))
gamma = 0.75
initial_state = 1

# -------------------------------
# Helper functions
# -------------------------------
def available_actions(state):
    current_state_row = M[state, ]
    return np.where(current_state_row >= 0)[1]

def sample_next_action(available_actions_range):
    return int(np.random.choice(available_actions_range, 1))

def update(current_state, action, gamma):
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
    max_index = int(np.random.choice(max_index, 1)) if len(max_index) > 1 else int(max_index)
    Q[current_state, action] = M[current_state, action] + gamma * Q[action, max_index]

    if np.max(Q) > 0:
        return np.sum(Q / np.max(Q) * 100)
    else:
        return 0

# -------------------------------
# Training function (ADDED)
# -------------------------------
def train_q_learning(iterations=1000):
    scores = []
    for _ in range(iterations):
        state = np.random.randint(0, MATRIX_SIZE)
        action = sample_next_action(available_actions(state))
        score = update(state, action, gamma)
        scores.append(score)
    return scores

# -------------------------------
# Train agent
# -------------------------------
scores = train_q_learning(1000)

# -------------------------------
# Test learned policy
# -------------------------------
current_state = 0
steps = [current_state]

while current_state != goal:
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]
    next_step_index = int(np.random.choice(next_step_index, 1)) if len(next_step_index) > 1 else int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

pl.plot(scores)
pl.xlabel('Number of iterations')
pl.ylabel('Reward gained')
pl.show()

# -------------------------------
# Environment-aware extension
# -------------------------------
police = [2, 4, 5]
drug_traces = [3, 8, 9]

env_police = np.matrix(np.zeros((MATRIX_SIZE, MATRIX_SIZE)))
env_drugs = np.matrix(np.zeros((MATRIX_SIZE, MATRIX_SIZE)))
Q = np.matrix(np.zeros((MATRIX_SIZE, MATRIX_SIZE)))

def collect_environmental_data(action):
    found = []
    if action in police:
        found.append('p')
    if action in drug_traces:
        found.append('d')
    return found

def update_env(current_state, action, gamma):
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
    max_index = int(np.random.choice(max_index, 1)) if len(max_index) > 1 else int(max_index)
    Q[current_state, action] = M[current_state, action] + gamma * Q[action, max_index]

    env = collect_environmental_data(action)
    if 'p' in env:
        env_police[current_state, action] += 1
    if 'd' in env:
        env_drugs[current_state, action] += 1

def available_actions_with_env_help(state):
    actions = available_actions(state)
    values = Q[state, actions]
    if np.sum(values < 0):
        actions = actions[np.array(values)[0] >= 0]
    return actions

# Train with environmental feedback
scores = []
for _ in range(1000):
    state = np.random.randint(0, MATRIX_SIZE)
    action = sample_next_action(available_actions_with_env_help(state))
    update_env(state, action, gamma)
    scores.append(np.sum(Q))

print("Police Found:")
print(env_police)
print("\nDrug Traces Found:")
print(env_drugs)

pl.plot(scores)
pl.xlabel('Number of iterations')
pl.ylabel('Reward gained')
pl.show()
