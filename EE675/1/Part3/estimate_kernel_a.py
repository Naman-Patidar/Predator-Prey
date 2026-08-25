# Question 3 part a.

import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'Question 1'))

from simulator_a import simulator
from kernel_b import state_to_idx 

def estimate_kernel(N, K, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
        
    num_states = N**4
    num_actions = 5
    
    row_indices = []
    col_indices = []
    data_values = []
    
    # Iterate through all states
    for pred_x in range(1, N+1):
        for pred_y in range(1, N+1):
            for prey_x in range(1, N+1):
                for prey_y in range(1, N+1):
                    pred_loc = (pred_x, pred_y)
                    prey_loc = (prey_x, prey_y)
                    s_idx = state_to_idx(N, pred_loc, prey_loc)
                    
                    # Iterate through all actions
                    for a in range(num_actions):
                        row_idx = s_idx * num_actions + a
                        
                        # To track how many times we landed in each next stare
                        next_state_counts = {}
                        
                        # Sample the simulator k times.
                        for _ in range(K):
                            next_pred, next_prey, _ = simulator(N, pred_loc, prey_loc, a)
                            next_s_idx = state_to_idx(N, next_pred, next_prey)
                            
                            # Increment count for this next state
                            next_state_counts[next_s_idx] = next_state_counts.get(next_s_idx, 0) + 1
                            
                        # Converts counts to probabilities
                        for next_s_idx, count in next_state_counts.items():
                            prob = count / K
                            row_indices.append(row_idx)
                            col_indices.append(next_s_idx)
                            data_values.append(prob)
                            
    P_estimated = csr_matrix((data_values, (row_indices, col_indices)), 
                             shape=(num_states * num_actions, num_states))
    
    return P_estimated