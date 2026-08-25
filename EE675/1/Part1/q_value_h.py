#Question 1 Part h.

import numpy as np
from state_value_g import state_value_eval # Using your file from Part g

def q_value_eval(pi, P, R):
    """
    Evaluates the action-value (Q) function for a given policy.
    
    Inputs:
    pi : policy matrix (|S| x |A|)
    P  : transition kernel (|S||A| x |S|)
    R  : reward function (|S| x |A|)
    
    Output:
    Q  : action-value matrix of size (|S| x |A|)
    """
    gamma = 0.99
    num_states = pi.shape[0]
    num_actions = pi.shape[1]
    
    # 1. Get the state value function V^pi (size: |S| x 1)
    # This runs the iterative evaluation from Part (g)
    print("Calculating V for Q-value evaluation...")
    V = state_value_eval(pi, P, R)
    
    # 2. Calculate the expected future value for all (s, a) pairs
    # P is |S||A| x |S|. V is |S| x 1.
    # P.dot(V) results in a |S||A| x 1 column vector
    expected_future_v = P.dot(V)
    
    # 3. Reshape the 1D result into a 2D matrix of size |S| x |A|
    expected_future_v_matrix = expected_future_v.reshape((num_states, num_actions))
    
    # 4. Q(s, a) = R(s, a) + gamma * expected_future_V
    # We convert the sparse R matrix to dense to add it to our dense V matrix.
    # For N=25, this |S| x |A| dense matrix is only about ~15MB, which is safe for RAM.
    Q = R.toarray() + gamma * expected_future_v_matrix
    
    return Q

