# Question 2, part d, e.

import time
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
q1_dir = os.path.join(parent_dir, 'Question 1')
sys.path.append(q1_dir)

from kernel_b import kernel            # Kernel function from Question 1
from reward_c import reward                 # Reward from Question 1
from value_iteration_a import value_iteration # From Question 2
from policy_iteration_c import policy_iteration # From Question 2

def run_q2_experiment():
    N_values = [4, 6, 8, 10, 12]
    l1_differences = []
    time_vi = []
    time_pi = []
    
    print("Starting Q2 Evaluation.")
    
    for N in N_values:
        print(f"\n--- Evaluating for N = {N} ---")
        
        # Generate environment
        P = kernel(N)
        R = reward(N)
        
        # Value Iteration
        start_vi = time.time()
        Q_vi = value_iteration(P, R)
        end_vi = time.time()
        time_vi.append(end_vi - start_vi)
        
        # Policy Iteration
        start_pi = time.time()
        Q_pi = policy_iteration(P, R)
        end_pi = time.time()
        time_pi.append(end_pi - start_pi)
        
        # L1 Difference
        # V(s)=max(Q(s,a)) overall all actions.
        V_vi = np.max(Q_vi, axis=1)
        V_pi = np.max(Q_pi, axis=1)
        
        # L1 norm.
        l1_diff = np.sum(np.abs(V_vi - V_pi))
        l1_differences.append(l1_diff)
        
        print(f"L1 Difference: {l1_diff:.6e}")
        print(f"VI Time: {time_vi[-1]:.2f}s | PI Time: {time_pi[-1]:.2f}s")

    
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, l1_differences, marker='o', color='purple', linewidth=2)
    plt.title('L1 Difference between VI and PI Value Functions', fontsize=14)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('L1 Norm Difference', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(N_values)
    plt.savefig('l1_difference_plot.png')
    plt.show()

    
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, time_vi, marker='s', label='Value Iteration', color='blue', linewidth=2)
    plt.plot(N_values, time_pi, marker='^', label='Policy Iteration', color='red', linewidth=2)
    plt.title('Execution Time: Value Iteration vs Policy Iteration', fontsize=14)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('Run Time (Seconds)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(N_values)
    plt.savefig('q2_runtime_plot.png')
    plt.show()

if __name__ == "__main__":
    run_q2_experiment()