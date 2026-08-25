import torch

import matplotlib
#We use 'Agg' backend to prevent the Tkinter crash
matplotlib.use('Agg') 


import matplotlib.pyplot as plt
from policy_net_a import PolicyNetwork
from gradient_estimate_b import gradient_estimate

from policy_net_a import PolicyNetwork
from gradient_estimate_b import gradient_estimate

def simple_SGA(policy_net, learning_rate):
    """Updates the neural policy parameters using Stochastic Gradient Ascent.
    """
    
    with torch.no_grad():
        for param in policy_net.parameters():
            if param.grad is not None:
                #Subtracting the negative of the gradient is mathematically equivalent to adding the gradient, which is what we want for ascent.
                param.data -= learning_rate * param.grad

def train_sga_model():
    N = 4
    num_iterations = 1000
    
    
    learning_rate = 0.005 
    
    # Initialize the neural network
    policy_net = PolicyNetwork(output_size=9)
    
    rewards_history = []
    
    print("Starting Training with Simple SGA...")
    print("-" * 40)
    
    for i in range(num_iterations):
        # 1. Trajectory and calculate gradients
        policy_net.zero_grad()

        total_reward = gradient_estimate(policy_net, N=N, max_steps=100)
        
        # 2. Stochastic Gradient Ascent step
        simple_SGA(policy_net, learning_rate)
        
        # Track the reward
        rewards_history.append(total_reward)
        
        if (i + 1) % 100 == 0:
            # Calculate the average reward over the last 100 episodes
            avg_reward = sum(rewards_history[-100:]) / 100.0
            print(f"Episode {i+1}/{num_iterations} | Avg Reward (last 100): {avg_reward:.2f}")

    print("-" * 40)
    

    # --- Plotting the Learning Curve ---
    # We use the moving average to smooth out the curve for better visualization
    window = 50
    smoothed_rewards = [sum(rewards_history[max(0, i-window):i+1]) / len(rewards_history[max(0, i-window):i+1]) for i in range(len(rewards_history))]

    plt.figure(figsize=(9, 5))
    plt.plot(rewards_history, color='lightblue', alpha=0.4, label='Raw Reward per Episode')
    plt.plot(smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed Reward (MA {window})')
    plt.title('Learning Curve: Simple Stochastic Gradient Ascent', fontsize=14)
    plt.xlabel('Number of Iterations (Episodes)', fontsize=12)
    plt.ylabel('Total Reward (Catches per 100 steps)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('learning_curve_sga.png')
    #plt.show()

if __name__ == "__main__":
    train_sga_model()
