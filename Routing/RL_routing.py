import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================
# 1. TSP DATA
# ============================================================

# Five cities
cities = torch.tensor([
    [0.0, 0.0],  # city 0
    [1.0, 3.0],  # city 1
    [4.0, 4.0],  # city 2
    [6.0, 1.0],  # city 3
    [3.0, 0.0],  # city 4
])

N = len(cities)


# ============================================================
# 2. POLICY NETWORK
# ============================================================

class TSPPolicy(nn.Module):

    def __init__(self, n_cities):
        super().__init__()

        # Input for each city:
        # x coordinate
        # y coordinate
        # visited flag
        #
        # Plus a one-hot vector identifying the current city.
        input_size = 3 + n_cities

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, cities, current_city, visited):

        n = len(cities)

        # Create one feature vector per city
        features = []

        for i in range(n):

            # Coordinates
            x, y = cities[i]

            # Has this city already been visited?
            visited_flag = float(i in visited)

            # One-hot representation of current city
            current_one_hot = [
                float(i == current_city)
                for _ in range(n)
            ]

            feature = torch.cat([
                torch.tensor([x, y, visited_flag]),
                torch.tensor(current_one_hot)
            ])

            features.append(feature)

        features = torch.stack(features)

        # Score each possible next city
        scores = self.network(features).squeeze(-1)

        # Prevent already visited cities from being selected
        for i in visited:
            scores[i] = -1e9

        # Convert scores into probabilities
        probabilities = torch.softmax(scores, dim=0)

        return probabilities


# ============================================================
# 3. DISTANCE FUNCTION
# ============================================================

def distance(a, b):
    return torch.sqrt(
        torch.sum((cities[a] - cities[b]) ** 2)
    )


# ============================================================
# 4. CREATE THE POLICY
# ============================================================

policy = TSPPolicy(N)

optimizer = optim.Adam(
    policy.parameters(),
    lr=0.001
)


# ============================================================
# 5. TRAIN USING REINFORCE
# ============================================================

num_episodes = 5000

for episode in range(num_episodes):

    current_city = 0
    visited = {0}

    log_probabilities = []
    total_distance = 0.0

    # --------------------------------------------------------
    # Construct one complete tour
    # --------------------------------------------------------

    while len(visited) < N:

        probabilities = policy(
            cities,
            current_city,
            visited
        )

        # Sample the next city from the policy
        distribution = torch.distributions.Categorical(
            probabilities
        )

        next_city = distribution.sample()

        # Save log probability for policy-gradient training
        log_probabilities.append(
            distribution.log_prob(next_city)
        )

        next_city = next_city.item()

        # Travel to selected city
        total_distance += distance(
            current_city,
            next_city
        )

        visited.add(next_city)
        current_city = next_city

    # --------------------------------------------------------
    # Return to starting city
    # --------------------------------------------------------

    total_distance += distance(
        current_city,
        0
    )

    # --------------------------------------------------------
    # REINFORCE
    #
    # Reward = negative distance
    # --------------------------------------------------------

    reward = -total_distance

    loss = 0

    for log_prob in log_probabilities:
        loss += -log_prob * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    if episode % 500 == 0:

        print(
            f"Episode {episode:4d} | "
            f"Distance = {total_distance.item():.3f}"
        )


# ============================================================
# 6. TEST THE TRAINED POLICY
# ============================================================

current_city = 0
visited = {0}

tour = [0]
total_distance = 0.0

while len(visited) < N:

    probabilities = policy(
        cities,
        current_city,
        visited
    )

    # During testing, choose the most likely city
    next_city = torch.argmax(probabilities).item()

    total_distance += distance(
        current_city,
        next_city
    )

    tour.append(next_city)
    visited.add(next_city)

    current_city = next_city

# Return to depot
total_distance += distance(
    current_city,
    0
)

tour.append(0)


print()
print("Learned tour:", tour)
print(
    "Total distance:",
    round(total_distance.item(), 3)
)