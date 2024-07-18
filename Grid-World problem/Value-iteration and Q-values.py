import numpy as np

class GridWorld:
    def __init__(self):
        self.num_rows = 3
        self.num_cols = 4
        self.num_states = self.num_rows * self.num_cols
        self.num_actions = 4
        self.gamma = 0.9
        self.noise = 0.2
        self.iterations = 100
        self.reward_value = 1
        self.punishment_value = -1
        self.grid_rewards = np.zeros(self.num_states)
        self.grid_rewards[3] = self.reward_value  # Reward at grid 3
        self.grid_rewards[7] = self.punishment_value  # Punishment at grid 7
        self.wall_idx = 5
        self.actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Down, Left, Up, Right

        # Initializing grid values
        self.values = np.zeros(self.num_states)
        self.values[self.wall_idx] = float('-inf')  # Wall at grid 5
        self.values[3] = self.grid_rewards[3]  # grid 3
        self.values[7] = self.grid_rewards[7]  # grid 7
        self.policy_values = []

        for _ in range(self.num_states):
          self.policy_values.append("Left")

        self.policy_values[self.wall_idx] = "Wall" # Policy wall at grid 5
        self.policy_values[3] = "+1" # Policy grid 3
        self.policy_values[7] = "-1" # Policy grid 7

        # Initializing Q-values
        self.q_values = np.zeros((self.num_states, self.num_actions))

    def value_iteration(self):
        for itr in range(self.iterations):
            new_values = np.copy(self.values)
            new_policy_values = np.copy(self.policy_values)
            delta = 0
            for s in range(self.num_states):
                if (s == self.wall_idx):  # Skip wall
                  new_values[s] = float('-inf')
                elif (s == 3):
                  new_values[s] = 1
                elif (s == 7):
                  new_values[s] = -1
                else :
                  temp = self.values[s]
                  value = float('-inf')
                  maximum = 0
                  max_action = -1
                  for a in range(self.num_actions): # 0 1 2 3
                    next = self.get_next_state(s, a)
                    if(self.values[next] > maximum):
                      maximum = self.values[next]
                      max_action = a
                  # 3 Right
                  next_state = self.get_next_state(s, a)

                  if (max_action == 0 or max_action == 2):
                    state_max_action = self.get_next_state(s, max_action)
                    state1 = self.get_next_state(s, 1)
                    state2 = self.get_next_state(s, 3) # gamma * ((1 - self.noise) * optimal + (1 - gamma) * state + (1 - gamma) * state )
                    value = self.gamma * ((1 - self.noise) * self.values[state_max_action] + 0.1 * self.values[state1] + 0.1 * self.values[state2])
                    dir = self.know_direction(max_action)
                    new_policy_values[s] = dir

                  if (max_action == 1 or max_action == 3):
                    state_max_action = self.get_next_state(s, max_action)
                    state1 = self.get_next_state(s, 0)
                    state2 = self.get_next_state(s, 2)
                    value = self.gamma * ((1 - self.noise) * self.values[state_max_action] + 0.1 * self.values[state1] + 0.1 * self.values[state2])
                    dir = self.know_direction(max_action)
                    new_policy_values[s] = dir
                  if max_action == -1:
                    value = 0
                  new_values[s] = value
                  delta = max(delta, abs(temp - new_values[s]))
                  # print("delta in loop",delta)
            self.values = new_values
            self.policy_values = new_policy_values
            self.print_grid_values("Value Iteration Results", itr)
            tol=1e-3
            print("tol",tol)
            print("delta",delta)
            if delta < tol:
              break

    def q_value_iteration(self):
        for itr in range(self.iterations):
            new_values = np.copy(self.values)
            new_policy_values = np.copy(self.policy_values)
            delta = 0
            for s in range(self.num_states):
                if (s == self.wall_idx):  # Skip wall
                  new_values[s] = float('-inf')
                  new_policy_values[s] = "WALL"
                  print(" WALL |")
                  print("")
                elif (s == 3):
                  new_values[s] = 1
                  new_policy_values[s] = "+1"
                  print(" +1 |")
                  print("")
                elif (s == 7):
                  new_values[s] = -1
                  new_policy_values[s] = "-1"
                  print(" -1 |")
                  print("")
                else :
                  temp = self.values[s]
                  value = float('-inf')
                  maximum = 0
                  max_action = -1
                  for a in range(self.num_actions): # 0 1 2 3
                    next = self.get_next_state(s, a)
                    if(self.values[next] > maximum):
                      maximum = self.values[next]
                      max_action = a
                  # 3 Right
                  next_state = self.get_next_state(s, a)
                  for a in range(self.num_actions):
                    dir = self.know_direction(a)

                    if (a == 0 or a == 2):
                      state_action = self.get_next_state(s, a)
                      state1 = self.get_next_state(s, 1)
                      state2 = self.get_next_state(s, 3)
                      value = self.gamma * ((1 - self.noise) * self.values[state_action] + 0.1 * self.values[state1] + 0.1 * self.values[state2])
                      print("State ",s," (",dir,") : ",round(value, 2))

                    if (a == 1 or a == 3):
                      state_action = self.get_next_state(s, a)
                      state1 = self.get_next_state(s, 0)
                      state2 = self.get_next_state(s, 2)
                      value = self.gamma * ((1 - self.noise) * self.values[state_action] + 0.1 * self.values[state1] + 0.1 * self.values[state2])
                      print("State ",s," (",dir,") : ",round(value, 2))

                  print("")
                  if (max_action == 0 or max_action == 2):
                    state_max_action = self.get_next_state(s, max_action)
                    state1 = self.get_next_state(s, 1)
                    state2 = self.get_next_state(s, 3)
                    value = self.gamma * ((1 - self.noise) * self.values[state_max_action] + 0.1 * self.values[state1] + 0.1 * self.values[state2])
                    dir = self.know_direction(max_action)
                    new_policy_values[s] = dir


                  if (max_action == 1 or max_action == 3):
                    state_max_action = self.get_next_state(s, max_action)
                    state1 = self.get_next_state(s, 0)
                    state2 = self.get_next_state(s, 2)
                    value = self.gamma * ((1 - self.noise) * self.values[state_max_action] + 0.1 * self.values[state1] + 0.1 * self.values[state2])
                    dir = self.know_direction(max_action)
                    new_policy_values[s] = dir
                  if max_action == -1:
                    value = 0
                  new_values[s] = value
                  delta = max(delta, abs(temp - new_values[s]))
                  # print("delta in loop",delta)
            self.values = new_values
            self.policy_values = new_policy_values
            self.print_grid_values("Value Iteration Results", itr)
            print("")
            tol=1e-3
            print("tol",tol)
            print("delta",delta)
            print("")
            if delta < tol:
              break

    def know_direction(self, a):
        if a == 0:
          dir = "Down"
        if a == 1:
          dir = "Left"
        if a == 2:
          dir = "Up"
        if a == 3:
          dir = "Right"
        return dir

    def get_next_state(self, state, action):
        row, col = divmod(state, self.num_cols)
        next_row = row + self.actions[action][0]  # (1, 0) represents moving down (increase in row by 1, no change in column).
        next_col = col + self.actions[action][1]  # (0, -1) represents moving left (no change in row, decrease in column by 1).
        next_state = next_row * self.num_cols + next_col # (-1, 0) represents moving up (decrease in row by 1, no change in column).
        if ((state==4 and action==1) or (state==8 and action==1)): # (0, 1) represents moving right (no change in row, increase in column by 1).
            next_state = state
        if next_state < 0 or next_state >= self.num_states or next_state == 5:  # Check if next_state is within bounds
            next_state = state
        return next_state

    def print_grid_values(self, title, itr):
        print(f"{title} - Iteration: {itr}:")
        if title == "Value Iteration Results":
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    idx = row * self.num_cols + col
                    if idx == self.wall_idx:
                        print(" WALL ", end="|")
                    elif idx == 3:  # Grid 3 with reward
                        print(f" +{self.reward_value} ", end="|")
                    elif idx == 7:  # Grid 7 with punishment
                        print(f" {self.punishment_value} ", end="|")
                    else:
                        print(f" {self.values[idx]:.2f} ", end="|")
                print()

    def extract_policy(self):
        print(f"The Policy")
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                idx = row * self.num_cols + col
                if idx == self.wall_idx:
                    print(" WALL ", end="|")
                elif idx == 3:  # Grid 3 with reward
                    print(f" +{self.reward_value} ", end="|")
                elif idx == 7:  # Grid 7 with punishment
                    print(f" {self.punishment_value} ", end="|")
                else:
                    print(f" {self.policy_values[idx]} ", end="|")
            print()


if __name__ == "__main__":
    grid_world = GridWorld()
    print("Choose the method:")
    print("1. Value Iteration")
    print("2. Q-Value Iteration")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        grid_world.value_iteration()
    elif choice == '2':
        grid_world.q_value_iteration()
    else:
        print("Invalid choice.")

    print("\nExtracted Policy:")
    grid_world.extract_policy()