import torch
import numpy as np

# Assuming you have updated your simulator to accept actions 0-8!
from simulator_function import simulator 
from policy_net_a import PolicyNetwork

def gradient_estimate(policy_net, N=4, max_steps=100, gamma=0.99):
    """Runs one episode of the predator-prey game, collects log probabilities and rewards,
    calculates discounted returns, and computes the policy gradient.
    """
    # 1. Initialize the environment
    # Please Note- The indices are 0 based
    pred_loc = (0, 0)
    prey_loc = (N-1, N-1)
    
    log_probs = []
    rewards = []
    
    # 2. Generate a Trajectory 
    for step in range(max_steps):
        state = (pred_loc[0], pred_loc[1], prey_loc[0], prey_loc[1])
        
        # Getting action and its log probability from the neural network
        action, log_prob = policy_net.get_action(state, N)
        
        
        next_pred, next_prey, reward = simulator(N, pred_loc, prey_loc, action)
        
        # Store for gradient calculation
        log_probs.append(log_prob)
        rewards.append(reward)
        
        # Update locations for the next step
        pred_loc = next_pred
        prey_loc = next_prey

    # 3. Calculate Discounted Returns (G_t)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
        
    returns = torch.tensor(returns)
    
    # Normalize returns for better training stability. 
    if returns.std() > 0:
        returns = (returns - returns.mean()) / returns.std()
        
    # 4. Construct the Surrogate Objective (Loss Function)
    policy_loss = []
    for log_prob, G_t in zip(log_probs, returns):
        #We want to maximize the expected return but pytorch minimizes by default, so we minimize the negative of it.
        policy_loss.append(-log_prob * G_t)
        
    # Sum up the losses over the trajectory
    policy_loss = torch.stack(policy_loss).sum()
    
    # 5. Compute the Gradients
    policy_loss.backward()
    
    # Return the total raw reward
    total_reward = sum(rewards)
    return total_reward
