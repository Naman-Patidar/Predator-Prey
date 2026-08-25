import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    
    def __init__(self, input_size=4, hidden_size=64, output_size=9):
        """
        input_size: 4 (predator x, predator y, prey x, prey y)
        output_size: 9 (probabilities for the 9 valid actions including diagonals)
        """
        super(PolicyNetwork, self).__init__()
        
        # We define the neural net layers as-
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Passes the state through the network to get action logits."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # We output raw logits instead of softmax probabilities for numerical stability
        logits = self.fc3(x)
        return logits
        
    def get_action(self, state, N=4):
        
        # Normalize the coordinates to be between 0 and 1
        normalized_state = [s / N for s in state]
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0) 
        
        # Get logits from the network
        logits = self.forward(state_tensor)
        
        # Create a categorical distribution over the logits
        dist = Categorical(logits=logits)
        
        # Sample an action (0 through 8)
        action = dist.sample()
        
        # return the log probability
        return action.item(), dist.log_prob(action)

