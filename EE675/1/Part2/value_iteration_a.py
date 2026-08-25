# Question 2, part a.
import numpy as np

def value_iteration(P, R):
    """Inputs:
    P : sparse transition kernel (|S||A| x |S|)
    R : sparse reward function (|S| x |A|)
    
    Output:
    Q : approximately optimal Q-function (|S| x |A| dense matrix)
    """
    gamma = 0.99
    epsilon = 1e-6  
    
    num_states = R.shape[0]
    num_actions = R.shape[1]
    
    # Initialize V to zeros
    V = np.zeros((num_states, 1))
    
    #Convert R to a dense 
    R_dense = R.toarray()
    
    iteration = 0
    delta = float('inf')
    
    print("Starting Value Iteration...")
    while delta > epsilon:
        # Calculate expected future value for all (s, a) pairs
        # P is (|S||A|x|S|), V is (|S|x1)->expected_V is (|S||A|x1)
        expected_V = P.dot(V)
        
        # Reshape to (|S|x|A|)
        expected_V_matrix = expected_V.reshape((num_states, num_actions))
        
        # Compute Q(s,a)
        Q = R_dense + gamma * expected_V_matrix
        
        # V_{k+1} is the maximum Q value for each state
        
        V_next = np.max(Q, axis=1, keepdims=True)
        
        # Checking for convergence 
        delta = np.max(np.abs(V_next - V))
        V = V_next
        iteration+=1
        
    print(f"Value Iteration converged in {iteration} iterations.")
    return Q