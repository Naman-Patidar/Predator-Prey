EE675 Assignment 2: Reinforcement Learning

Files Included:
- policy_net_a.py: Implements the policy network using the multi layer perceptron (MLP) using pytorch.
- simulator_a.py: It contains the environment and the process of simulation. 
- gradient_estimate_b.py: Runs a full episode and computes gradients using the REINFORCE algorithm.
- simple_sga_c.py: Trains the agent using a manually implemented Stochastic Gradient Ascent (SGA) update rule and generates a learning curve.
- adam_optimizer_d.py: Trains the agent using PyTorch's built-in Adam optimizer and outputs the learning curve.
- Assignment2_Report.pdf: Contains explanations of design choices and working principles.


1. To generate the learning curve for the Simple SGA (Part c):
   python simple_sga_c.py
   (This will run 1000 episodes and generate 'learning_curve_sga.png')

2. To generate the learning curve for the Adam Optimizer (Part d):
   python adam_optimizer_d.py
   (This will run 1000 episodes and generate 'learning_curve_adam.png')
