import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg') # To prevent the Tkinter crash
import matplotlib.pyplot as plt

from policy_net_a import PolicyNetwork
from gradient_estimate_b import gradient_estimate

def train_adam_model():
    N = 4
    num_iterations = 1000
    
    
    learning_rate = 0.005 
    
    # Initialize the neural network
    policy_net = PolicyNetwork(output_size=9)
    
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    rewards_history = []
    
    print("Starting Training with Adam Optimizer...")
    print("-" * 40)
    
    for i in range(num_iterations):
        # 1. Collect trajectory and calculate gradients
        # We zero the gradients before running the backward pass to prevent accumulation from previous iterations.
        optimizer.zero_grad()


        total_reward = gradient_estimate(policy_net, N=N, max_steps=100)
        
        # 2. Perform the Adam optimization step
        optimizer.step()
        
        # Track the reward
        rewards_history.append(total_reward)
        
        if (i + 1) % 100 == 0:
            avg_reward = sum(rewards_history[-100:]) / 100.0
            print(f"Episode {i+1}/{num_iterations} | Avg Reward (last 100): {avg_reward:.2f}")

    print("-" * 40)
    print("Training Complete! Generating Learning Curve...")

    # --- Plotting the Learning Curve ---
    # We use the moving average to smooth out the curve for better visualization
    window = 50
    smoothed_rewards = [sum(rewards_history[max(0, i-window):i+1]) / len(rewards_history[max(0, i-window):i+1]) for i in range(len(rewards_history))]

    plt.figure(figsize=(9, 5))
    plt.plot(rewards_history, color='lightgreen', alpha=0.4, label='Raw Reward per Episode')
    plt.plot(smoothed_rewards, color='darkgreen', linewidth=2, label=f'Smoothed Reward (MA {window})')
    plt.title('Learning Curve: Adam Optimizer', fontsize=14)
    plt.xlabel('Number of Iterations (Episodes)', fontsize=12)
    plt.ylabel('Total Reward (Catches per 100 steps)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    
    plt.savefig('learning_curve_adam.png')

if __name__ == "__main__":
    train_adam_model()