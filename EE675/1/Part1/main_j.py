#Question 1 Part j.

import time
import matplotlib.pyplot as plt
import numpy as np

# Import all the functions we built
from kernel_b import kernel, state_to_idx
from reward_c import reward
from policy_d import sample_policy
from state_value_g import state_value_eval

def run_experiment():
    # We have new values of N as 4, 6, 8, 10, 12
    N_values = [4, 6, 8, 10, 12]
    state_values = []
    run_times = []
    
    print("Starting Part (j) Evaluation...")
    print("-" * 40)
    
    for N in N_values:
        print(f"Evaluating for N = {N}...")
        start_time = time.time()
        
        # Generate the foundational matrices
        P = kernel(N)
        R = reward(N)
        pi = sample_policy(N)
        
        #Evaluate the state values
        V = state_value_eval(pi, P, R)
        
        #Calculate time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        run_times.append(elapsed_time)
        
        # Extract the value for the specific initial state
        # Initial state: pred at (1,1), prey at (N,N)
        s_idx = state_to_idx(N, (1, 1), (N, N))
        v_initial = V[s_idx, 0] # Extract the scalar value from the column vector
        state_values.append(v_initial)
        
        print(f"  -> State Value for Initial State: {v_initial:.6f}")
        print(f"  -> Time taken: {elapsed_time:.2f} seconds\n")
        
    print("-" * 40)
    print("Experiment Complete! Generating plots...")
    
    #Plot 1: State Values vs N
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, state_values, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('State Value of Initial State vs. Grid Size (N)', fontsize=14)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('State Value V(s_0)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(N_values)
    plt.savefig('state_values_plot.png') # Saves the graph as an image
    plt.show() 
    
    #Plot 2: Run Time vs N
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, run_times, marker='s', linestyle='-', color='r', linewidth=2)
    plt.title('Execution Time vs. Grid Size (N)', fontsize=14)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('Run Time (Seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(N_values)
    plt.savefig('run_time_plot.png')
    plt.show()

if __name__ == "__main__":
    run_experiment()