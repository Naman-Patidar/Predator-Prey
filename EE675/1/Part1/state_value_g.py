#Question 1 Part g.

import numpy as np
from induced_kernel_e import induced_kernel
from induced_reward_f import induced_reward

def state_value_eval(pi, P, R):
    """Inputs:
    pi : policy matrix (|S| x |A|)
    P  : transition kernel (|S||A| x |S|)
    R  : reward function (|S| x |A|)
    
    Output:
    V  : state value column vector (|S| x 1)
    """
    # Discount factor, gamma=0.99
    gamma = 0.99
    # Convergence threshold
    theta = 1e-6 
    
    num_states = pi.shape[0]
    
    #Generate the policy-induced kernel and reward
    P_pi = induced_kernel(P, pi)
    R_pi = induced_reward(R, pi)
    
    
    R_pi = np.asarray(R_pi).reshape(num_states, 1)
    
    #Initialize V_0 to zeros
    V = np.zeros((num_states, 1))
    
    #Iterative Policy Evaluation
    iteration = 0
    delta = float('inf')
    
    while delta > theta:
        # Bellmann Update
        V_next = R_pi + gamma * P_pi.dot(V)
        
        # Calculating the maximum change
        delta = np.max(np.abs(V_next - V))
        
        V = V_next
        iteration += 1
        
    print(f"Value evaluation converged in {iteration} iterations.")
    
    return V

