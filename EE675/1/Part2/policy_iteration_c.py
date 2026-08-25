# Question 2, part c.

import numpy as np
from scipy.sparse import csr_matrix

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
q1_dir = os.path.join(parent_dir, 'Question 1')
sys.path.append(q1_dir)

from q_value_h import q_value_eval 
from induced_policy_b import induced_policy

def policy_iteration(P, R):
    """Inputs:
    P : sparse transition kernel (|S||A| x |S|)
    R : sparse reward function (|S| x |A|)
    
    Output:
    Q : approximately optimal Q-function (|S| x |A| dense matrix)
    """
    num_states = R.shape[0]
    num_actions = R.shape[1]
    epsilon = 1e-6
    
    # Initialising with a uniform random policy
    pi_dense = np.ones((num_states, num_actions)) / num_actions
    pi = csr_matrix(pi_dense)
    
    V_old = np.zeros((num_states, 1))
    
    iteration = 0

    while True:
        # Policy Evaluation
        Q = q_value_eval(pi, P, R)
        
        # Calculate V^pi from Q and pi to check for convergence
        V_current = np.sum(Q * pi.toarray(), axis=1, keepdims=True)
        
        # Checking for convergence.
        delta = np.max(np.abs(V_current - V_old))
        if delta < epsilon:
            print(f"Policy Iteration converged in {iteration} outer steps.")
            break
            
        V_old = V_current
        
        #Policy Improvement
        pi = induced_policy(Q)
        iteration += 1
        
    return Q