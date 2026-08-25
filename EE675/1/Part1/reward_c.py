#Question 1- Part c.

import numpy as np
from scipy.sparse import csr_matrix

#Get the helper function from part a.
from simulator_a import get_valid_next_loc 

def state_to_idx(N, pred_loc, prey_loc):
    #Converting into a 1-D based index array.
    px, py = pred_loc[0] - 1, pred_loc[1] - 1
    rx, ry = prey_loc[0] - 1, prey_loc[1] - 1
    return px * (N**3) + py * (N**2) + rx * N + ry

def reward(N):
    #This function gerenates the reward matrix of size |S||A| x 1. 
    num_states = N**4

    #No of action mappings---  0:stay, 1:up, 2:down, 3:left, 4:right
    num_actions = 5 
    
    row_indices = []
    col_indices = []
    data_values = []
    
    # Iterate through all possible states
    for pred_x in range(1, N+1):
        for pred_y in range(1, N+1):
            for prey_x in range(1, N+1):
                for prey_y in range(1, N+1):
                    
                    pred_loc = (pred_x, pred_y)
                    prey_loc = (prey_x, prey_y)
                    s_idx = state_to_idx(N, pred_loc, prey_loc)
                    
                    # Iterating through all possible actions
                    for a in range(5):
                        # Get where the predator ends up
                        next_pred_loc = get_valid_next_loc(N, pred_loc, a)
                        
                        # If the predator lands on the prey, reward is +1
                        if next_pred_loc == prey_loc:
                            row_indices.append(s_idx)
                            col_indices.append(a)
                            data_values.append(1.0)
                            # We only append 1.0s. The sparse matrix assumes all other pairs are 0.
                            
    # Construct the sparse matrix
    R_matrix = csr_matrix((data_values, (row_indices, col_indices)), 
                          shape=(num_states, num_actions))
    
    return R_matrix

