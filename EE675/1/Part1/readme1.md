Predator-Prey Reinforcement Learning Environment

This project implements a complete Markov Decision Process (MDP) framework for a predator-prey game on an 𝑁×𝑁 grid. The goal is to model the environment, define policies, and evaluate value functions using dynamic programming techniques.

File Structure

Each function is implemented in a separate file as per instructions:

Question 1
│
├── simulator_a.py          # Part (a): Environment simulator
├── kernel_b.py             # Part (b): State-transition kernel
├── reward_c.py             # Part (c): Reward function
├── policy_d.py             # Part (d): Sample policy
├── induced_kernel_e.py     # Part (e): Policy-induced kernel
├── induced_reward_f.py     # Part (f): Policy-induced reward
├── state_value_g.py        # Part (g): State value evaluation
├── q_value_h.py            # Part (h): Q-value evaluation
├── main_j.py               # Part (j): Experiment + plots
│
└── README.md               # This file

Environment Description
Grid size: N×N
Predator initial position: (1,1)
Prey initial position: (N,N)
Actions:
0: Stay
1: Up
2: Down
3: Left
4: Right

Our objective is to catch the prey: Reward= +1, if predator catches the prey
                                    Reqard= 0 , in all other cases
If the prey is caught, it respawns in one of the cell uniformly

Run the file-   main_j.py
