#Question 1 Part b.

import numpy as np
from scipy.sparse import csr_matrix

# Importing your helper function from Part a
from simulator_a import get_valid_next_loc 

def state_to_idx(N, pred_loc, prey_loc):
    #Since predator- N^2, prey- N^2, total states = N^4

    px, py = pred_loc[0] - 1, pred_loc[1] - 1
    rx, ry = prey_loc[0] - 1, prey_loc[1] - 1
    
    # Converting to base N to get a unique index
    return px * (N**3) + py * (N**2) + rx * N + ry

def kernel(N):
    # Generates the |S||A| x |S| sparse state-transition kernel.
    num_states = N**4
    # No of action mapping- 0:stay, 1:up, 2:down, 3:left, 4:right
    num_actions = 5  
    
    # Lists to construct the sparse matrix efficiently
    row_indices = []
    col_indices = []
    data_values = []
    
    all_cells = [(i, j) for i in range(1, N+1) for j in range(1, N+1)]
    
    # Iterate through every possible state
    for pred_x in range(1, N+1):
        for pred_y in range(1, N+1):
            for prey_x in range(1, N+1):
                for prey_y in range(1, N+1):
                    
                    pred_loc = (pred_x, pred_y)
                    prey_loc = (prey_x, prey_y)
                    s_idx = state_to_idx(N, pred_loc, prey_loc)
                    
                    # Iterating through every possible action
                    for a in range(5):
                        # Calculate the specific row index for this (state, action) pair
                        row_idx = s_idx * num_actions + a
                        
                        # Getting the Predator Move
                        next_pred_loc = get_valid_next_loc(N, pred_loc, a)
                        
                        # Get the probabilities based on Catch or No-Catch
                        if next_pred_loc == prey_loc:
                            # CATCH: Prey respawns uniformly
                            prob = 1.0 / (N**2 - 1)
                            for spawn_loc in all_cells:
                                if spawn_loc != next_pred_loc:
                                    next_s_idx = state_to_idx(N, next_pred_loc, spawn_loc)
                                    
                                    row_indices.append(row_idx)
                                    col_indices.append(next_s_idx)
                                    data_values.append(prob)
                        else:
                            # NO CATCH: Prey moves to valid neighbor uniformly
                            prey_actions = [0, 1, 2, 3, 4]
                            valid_prey_locs = []
                            for pa in prey_actions:
                                valid_loc = get_valid_next_loc(N, prey_loc, pa)
                                if valid_loc not in valid_prey_locs:
                                    valid_prey_locs.append(valid_loc)
                            
                            prob = 1.0 / len(valid_prey_locs)
                            for next_prey_loc in valid_prey_locs:
                                next_s_idx = state_to_idx(N, next_pred_loc, next_prey_loc)
                                
                                row_indices.append(row_idx)
                                col_indices.append(next_s_idx)
                                data_values.append(prob)
                                
    # Construct the Compressed Sparse Row (CSR) matrix
    transition_kernel = csr_matrix((data_values, (row_indices, col_indices)), 
                                   shape=(num_states * num_actions, num_states))
    
    return transition_kernel

