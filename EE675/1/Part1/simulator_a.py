#Question 1- Part a.

import numpy as np

# We use the following action mapping:
# 0: stay, 1: up, 2: down, 3: left, 4: right

def get_valid_next_loc(N, loc, action):
    x, y = loc

    if action == 1 and y < N:        # up
        y += 1
    elif action == 2 and y > 1:      # down
        y -= 1
    elif action == 3 and x > 1:      # left
        x -= 1
    elif action == 4 and x < N:      # right
        x += 1
    # else: invalid move → stay in same cell

    return (x, y)


def simulator(N, pred_loc, prey_loc, pred_action):
    # Validate action
    assert pred_action in [0, 1, 2, 3, 4], "Invalid predator action"

    # Moving thw predator
    next_pred_loc = get_valid_next_loc(N, pred_loc, pred_action)

    # Now we check if the prey is caught, and give reward accordingly.
    if next_pred_loc == prey_loc:
        reward = 1

        # Since the prey is caught, we respawn at a random location apart from the predator's new location.
        while True:
            new_prey = (np.random.randint(1, N+1),
                        np.random.randint(1, N+1))
            if new_prey != next_pred_loc:
                next_prey_loc = new_prey
                break

    else:
        reward = 0

        # Prey moves (/or does not move) randomly to one of the valid cells
        prey_actions = [0, 1, 2, 3, 4]

        # First we generate all possible next locations
        possible_locs = set(
            get_valid_next_loc(N, prey_loc, a)
            for a in prey_actions
        )

        # Convert to list for random selection
        possible_locs = list(possible_locs)

        # Choose the next prey location randomly
        next_prey_loc = possible_locs[np.random.choice(len(possible_locs))]

    return next_pred_loc, next_prey_loc, reward