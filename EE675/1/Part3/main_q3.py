# Question 3 part b, c.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'Question 1'))
sys.path.append(os.path.join(parent_dir, 'Question 2'))

from kernel_b import kernel
from reward_c import reward
from value_iteration_a import value_iteration
from estimate_kernel_a import estimate_kernel

def run_q3_experiment():
    N = 5
    K_values = [4, 6, 8, 10, 12]
    seeds = [42, 100, 2026, 777, 999] # Random seeds values
    
    print(f"Q3 Evaluation (N={N}) ")
    
    
    print("Computing exact kernel and Q-function baseline")
    P_exact = kernel(N)
    R = reward(N)
    Q_exact = value_iteration(P_exact, R)
    
    mean_l1_diffs = []
    std_l1_diffs = []
    
    for K in K_values:
        print(f"\nEvaluating for K = {K}")
        k_l1_diffs = []
        
        for seed in seeds:
            #Kernel Estimation
            P_est = estimate_kernel(N, K, seed)
            
            # Value iteration with estimated kernel
            Q_est = value_iteration(P_est, R)
            
            # Calculating the L1 difference between the exact Q and the estimated Q
            l1_diff = np.sum(np.abs(Q_exact - Q_est))
            k_l1_diffs.append(l1_diff)
            
        # Calculate statistics for this K
        mean_diff = np.mean(k_l1_diffs)
        std_diff = np.std(k_l1_diffs)
        
        mean_l1_diffs.append(mean_diff)
        std_l1_diffs.append(std_diff)
        
        print(f"Mean L1 Diff: {mean_diff:.2f} | Std Dev: {std_diff:.2f}")

    print("\nGenerating plots")

    # Plotting Mean and Std Dev 
    plt.figure(figsize=(9, 6))
    
    
    plt.plot(K_values, mean_l1_diffs, marker='o', color='teal', linewidth=2, label='Mean L1 Difference')
    
   
    mean_arr = np.array(mean_l1_diffs)
    std_arr = np.array(std_l1_diffs)
    plt.fill_between(K_values, mean_arr - std_arr, mean_arr + std_arr, color='teal', alpha=0.2, label='± 1 Std Dev')

    plt.title(f'Q-Function Error vs. Samples per Action (N={N})', fontsize=14)
    plt.xlabel('Number of Simulator Samples (K)', fontsize=12)
    plt.ylabel('L1 Difference (Exact Q vs. Estimated Q)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(K_values)
    plt.legend()
    plt.savefig('q3_kernel_estimation_plot.png')
    plt.show()

if __name__ == "__main__":
    run_q3_experiment()

