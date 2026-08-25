# Question 2, part b.

import numpy as np
from scipy.sparse import csr_matrix

def induced_policy(Q):
    """
    Greedily extracts the policy matrix from a given Q-function.
    
    Input:
    Q : Q-function (|S| x |A| dense matrix)
    
    Output:
    pi : greedy policy (|S| x |A| sparse matrix)
    """
    # Find the maximum Q value for each state
    max_Q = np.max(Q, axis=1, keepdims=True)
    
    # Create a boolean mask of where Q equals the max Q.
    # We use np.isclose instead of == to prevent floating point errors
    is_max = np.isclose(Q, max_Q)
    
    # Convert boolean to float (True -> 1.0, False -> 0.0)
    pi_dense = is_max.astype(float)
    
    # Normalize each row so probabilities sum to 1
    # (e.g., if two actions tie, they both get 0.5)
    row_sums = np.sum(pi_dense, axis=1, keepdims=True)
    
    # Avoid division by zero in weird edge cases (though row_sums should never be 0 here)
    row_sums[row_sums == 0] = 1 
    pi_dense /= row_sums
    
    # Convert back to sparse matrix for memory efficiency downstream
    pi_sparse = csr_matrix(pi_dense)
    
    return pi_sparse