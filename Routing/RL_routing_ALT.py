import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. TSP ENVIRONMENT
# ============================================================

class TSPEnv(gym.Env):

    def __init__(self):

        super().__init__()

        # Five cities
        self.cities = np.array([
            [0.0, 0.0],
            [1.0, 3.0],
            [4.0, 4.0],
            [6.0, 1.0],
            [3.0, 0.0],
        ], dtype=np.float32)

        self.n_cities = len(self.cities)

        # Action = choose next city
        self.action_space = spaces.Discrete(self.n_cities)

        # Observation:
        #
        # [current_city_one_hot,
        #  visited_flags,
        #  city_coordinates]
        #
        # current city: 5 values
        # visited:      5 values
        # coordinates:  5 * 2 = 10 values
        #
        # Total = 20
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_cities * 4,),
            dtype=np.float32
        )

    def _get_observation(self):

        # Current city as one-hot vector
        current_one_hot = np.zeros(self.n_cities)
        current_one_hot[self.current_city] = 1.0

        # Which cities have been visited?
        visited = self.visited.astype(np.float32)

        # Flatten coordinates
        coordinates = self.cities.flatten()

        observation = np.concatenate([
            current_one_hot,
            visited,
            coordinates
        ])

        return observation.astype(np.float32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Start at city 0
        self.current_city = 0

        # Mark city 0 as visited
        self.visited = np.zeros(
            self.n_cities,
            dtype=bool
        )

        self.visited[0] = True

        self.total_distance = 0.0

        return self._get_observation(), {}

    def step(self, action):

        # ----------------------------------------------------
        # Invalid action: city has already been visited
        # ----------------------------------------------------

        if self.visited[action]:

            reward = -10.0

            terminated = False

            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                {}
            )

        # ----------------------------------------------------
        # Travel to selected city
        # ----------------------------------------------------

        old_city = self.current_city

        distance = np.linalg.norm(
            self.cities[old_city] -
            self.cities[action]
        )

        self.total_distance += distance

        self.current_city = action
        self.visited[action] = True

        # ----------------------------------------------------
        # Have we visited every city?
        # ----------------------------------------------------

        terminated = np.all(self.visited)

        if terminated:

            # Return to starting city
            return_distance = np.linalg.norm(
                self.cities[self.current_city] -
                self.cities[0]
            )

            self.total_distance += return_distance

            # Reward = negative total tour length
            reward = -self.total_distance

        else:

            # Intermediate reward
            reward = -distance

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            {}
        )


# ============================================================
# 2. NEURAL NETWORK POLICY
# ============================================================

class PolicyNetwork(nn.Module):

    def __init__(self, observation_size, n_actions):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(observation_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, n_actions)
        )

    def forward(self, observation):

        return self.network(observation)


# ============================================================
# 3. CREATE ENVIRONMENT AND POLICY
# ============================================================

env = TSPEnv()

policy = PolicyNetwork(
    observation_size=env.observation_space.shape[0],
    n_actions=env.action_space.n
)

optimizer = optim.Adam(
    policy.parameters(),
    lr=0.001
)


# ============================================================
# 4. TRAIN WITH REINFORCE
# ============================================================

episodes = 5000

for episode in range(episodes):

    observation, info = env.reset()

    done = False

    log_probs = []
    rewards = []

    while not done:

        # Convert observation to PyTorch tensor
        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32
        )

        # Network outputs one score for each city
        logits = policy(observation_tensor)

        # Mask cities that have already been visited
        for city in range(env.n_cities):

            if env.visited[city]:
                logits[city] = -1e9

        # Convert scores to probabilities
        distribution = torch.distributions.Categorical(
            logits=logits
        )

        # Sample next city
        action = distribution.sample()

        # Store log probability
        log_probs.append(
            distribution.log_prob(action)
        )

        # Send action to Gymnasium environment
        observation, reward, terminated, truncated, info = \
            env.step(action.item())

        rewards.append(reward)

        done = terminated or truncated

    # --------------------------------------------------------
    # Calculate returns
    # --------------------------------------------------------

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + 0.99 * G
        returns.insert(0, G)

    returns = torch.tensor(
        returns,
        dtype=torch.float32
    )

    # Normalize returns for more stable training
    if len(returns) > 1:
        returns = (
            returns - returns.mean()
        ) / (returns.std() + 1e-8)

    # --------------------------------------------------------
    # REINFORCE loss
    # --------------------------------------------------------

    loss = 0

    for log_prob, G in zip(log_probs, returns):

        loss += -log_prob * G

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if episode % 500 == 0:

        print(
            f"Episode {episode:4d} | "
            f"Tour distance: "
            f"{env.total_distance:.3f}"
        )


# ============================================================
# 5. TEST THE TRAINED POLICY
# ============================================================

observation, info = env.reset()

done = False

tour = [0]

while not done:

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32
    )

    with torch.no_grad():

        logits = policy(observation_tensor)

    # Mask visited cities
    for city in range(env.n_cities):

        if env.visited[city]:
            logits[city] = -1e9

    # Greedy choice during testing
    action = torch.argmax(logits).item()

    observation, reward, terminated, truncated, info = \
        env.step(action)

    tour.append(action)

    done = terminated or truncated


# Return to starting city for display
tour.append(0)

print()
print("Learned tour:", tour)
print(
    "Total distance:",
    round(env.total_distance, 3)
)