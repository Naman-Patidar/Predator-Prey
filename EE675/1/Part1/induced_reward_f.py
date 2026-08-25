#Question 1 Part f.

import numpy as np
from scipy.sparse import csr_matrix

def induced_reward(R, pi):
    """Inputs:
    R : sparse matrix of size |S| x |A|
    pi: sparse matrix of size |S| x |A|
    
    Output:
    R_pi : matrix (or 2D array) of size |S| x 1
    """
    
    # Element wise multiplication of R and pi to get the expected reward for each state-action pair under the policy
    # For each state s and action a, the expected reward is R(s,a)*pi(s,a)
    expected_R_matrix = R.multiply(pi)
    
    # Sum across the actions to get the expected reward for each state under the policy
   
    R_pi = expected_R_matrix.sum(axis=1)
    # This returns a dense column vector of size |S| x 1. 
    return R_pi