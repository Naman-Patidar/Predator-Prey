#Question 1 Part e.

import numpy as np
from scipy.sparse import csr_matrix

def induced_kernel(P, pi):
    """Inputs:
    P : sparse matrix of size |S||A| x |S|
    pi: sparse matrix of size |S| x |A|
    
    Output:
    P_pi : sparse matrix of size |S| x |S|
    """
    num_states = pi.shape[0]
    num_actions = pi.shape[1]
    
    # Converting policy matrix to COO format for extraction of non-zero entries
    pi_coo = pi.tocoo()
    
    # Building the projection matrix of size |S| x (|S||A|)
    # For a state s and action a, the corresponding column in Pi_matrix will be 1 if pi(s, a) > 0
    row_indices = pi_coo.row
    col_indices = pi_coo.row * num_actions + pi_coo.col
    data_values = pi_coo.data
    
    Pi_matrix = csr_matrix((data_values, (row_indices, col_indices)), 
                           shape=(num_states, num_states * num_actions))
    
    # The policy induced kernel is dot product of the projection matrix and the transition matrix
    P_pi = Pi_matrix.dot(P)
    
    return P_pi