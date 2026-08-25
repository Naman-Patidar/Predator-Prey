import numpy as np

# We are mapping the actions as follows:
# 0: stay, 1: up, 2: down, 3: left, 4: right
# 5: up-left, 6: up-right, 7: down-left, 8: down-right

def get_valid_next_loc(N, loc, action):
    x, y = loc

    if action == 0:
        pass  # stay

    elif action == 1 and y < N - 1:          # up
        y += 1
    elif action == 2 and y > 0:              # down
        y -= 1
    elif action == 3 and x > 0:              # left
        x -= 1
    elif action == 4 and x < N - 1:          # right
        x += 1
    elif action == 5 and x > 0 and y < N - 1:  # up-left
        x -= 1
        y += 1
    elif action == 6 and x < N - 1 and y < N - 1:  # up-right
        x += 1
        y += 1
    elif action == 7 and x > 0 and y > 0:  # down-left
        x -= 1
        y -= 1
    elif action == 8 and x < N - 1 and y > 0:  # down-right
        x += 1
        y -= 1
    # else: invalid move → stay

    return (x, y)


def simulator(N, pred_loc, prey_loc, pred_action):
    assert pred_action in range(9), "Invalid predator action"

    # Move predator
    next_pred_loc = get_valid_next_loc(N, pred_loc, pred_action)

    # Checking the capture
    if next_pred_loc == prey_loc:
        reward = 1

        # Respawn prey at random location != predator
        while True:
            new_prey = (np.random.randint(0, N),
                        np.random.randint(0, N))
            if new_prey != next_pred_loc:
                next_prey_loc = new_prey
                break
    else:
        reward = 0

        # Prey random movement
        prey_actions = list(range(9))

        possible_locs = set(
            get_valid_next_loc(N, prey_loc, a)
            for a in prey_actions
        )

        possible_locs = list(possible_locs)

        next_prey_loc = possible_locs[
            np.random.choice(len(possible_locs))
        ]

    return next_pred_loc, next_prey_loc, reward