#Question 1. Part d.

import numpy as np
from scipy.sparse import csr_matrix

#Get the helper function from part a.
from simulator_a import get_valid_next_loc

# Action mapping:
# 0: stay, 1: up, 2: down, 3: left, 4: right
ACTIONS = [0, 1, 2, 3, 4]

def state_to_idx(N, pred_loc, prey_loc):
    #Converting into a 1-D based index array.
    px, py = pred_loc[0] - 1, pred_loc[1] - 1
    rx, ry = prey_loc[0] - 1, prey_loc[1] - 1
    return px * (N**3) + py * (N**2) + rx * N + ry

def manhattan_distance(loc1, loc2):
    
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

def sample_policy(N):
    """Generates a stochastic policy π(s,a) of size |S| x |A|.
    Policy:
    With probability 0.5 → choose action(s) minimizing distance to prey
    With probability 0.5 → choose uniformly among remaining actions
    """

    num_states = N**4
    num_actions = len(ACTIONS)

    row_indices = []
    col_indices = []
    data_values = []

    # Go through every possible state
    for pred_x in range(1, N+1):
        for pred_y in range(1, N+1):
            for prey_x in range(1, N+1):
                for prey_y in range(1, N+1):

                    pred_loc = (pred_x, pred_y)
                    prey_loc = (prey_x, prey_y)
                    s_idx = state_to_idx(N, pred_loc, prey_loc)

                    # Calculate manhattan distance for each action
                    action_distances = []
                    for a in ACTIONS:
                        next_pred_loc = get_valid_next_loc(N, pred_loc, a)
                        dist = manhattan_distance(next_pred_loc, prey_loc)
                        action_distances.append((a, dist))

                    # Getting the min distance.
                    min_dist = min(d for _, d in action_distances)

                    # Separating the best and other actions.
                    best_actions = [a for a, d in action_distances if d == min_dist]
                    other_actions = [a for a, d in action_distances if d != min_dist]

                    # Assigning the probabilities.
                    if len(other_actions) > 0:
                        prob_best = 0.5 / len(best_actions)
                        prob_other = 0.5 / len(other_actions)
                    else:
                        # If all actions are best.
                        prob_best = 1.0 / len(best_actions)
                        prob_other = 0.0

                    # Fill the values in sparse matrix
                    for a in best_actions:
                        row_indices.append(s_idx)
                        col_indices.append(a)
                        data_values.append(prob_best)

                    for a in other_actions:
                        row_indices.append(s_idx)
                        col_indices.append(a)
                        data_values.append(prob_other)

    # Making the sparse policy matrix
    policy_matrix = csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(num_states, num_actions)
    )

    # Each row should sum to 1, so a quick sanity check
    row_sums = np.array(policy_matrix.sum(axis=1)).flatten()
    assert np.allclose(row_sums, 1), "Policy rows do not sum to 1"

    return policy_matrix