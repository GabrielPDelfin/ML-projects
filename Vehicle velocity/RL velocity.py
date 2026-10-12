import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# -----------------------------
# Configuration
# -----------------------------
TARGET_SPEED = 15.0
MAX_ACCELERATION = 3.0
DT = 0.1
EPISODE_LENGTH = 100
GAMMA = 0.99
EPISODES = 2000


# -----------------------------
# Custom environment (no Gymnasium)
# -----------------------------
class DeliveryVehicle:
    """Toy simulation: control vehicle speed toward a target speed."""

    def reset(self):
        self.speed = 0.0
        self.steps = 0
        return np.array([self.speed], dtype=np.float32)

    def step(self, action):
        # Convert the policy action into a bounded acceleration.
        # action is a scalar; tanh maps any real value to [-1, 1].
        acceleration = MAX_ACCELERATION * np.tanh(float(action))

        # Simplified vehicle dynamics.
        self.speed = max(0.0, self.speed + acceleration * DT)
        self.steps += 1

        # Reward for reaching the target speed without excessive acceleration.
        reward = -(
            (self.speed - TARGET_SPEED) ** 2
            + 0.1 * acceleration**2
        )

        done = self.steps >= EPISODE_LENGTH
        next_state = np.array([self.speed], dtype=np.float32)
        return next_state, float(reward), done


# -----------------------------
# Policy network
# -----------------------------
class ContinuousPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Output 0: mean; output 1: raw std parameter
        )

    def forward(self, state):
        output = self.network(state)

        mean = output[..., 0]
        # Standard deviation must be positive.
        std = torch.nn.functional.softplus(output[..., 1]) + 1e-5

        return mean, std


def discounted_returns(rewards, gamma):
    returns = []
    G = 0.0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize returns for somewhat more stable policy-gradient updates.
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (
            returns.std(unbiased=False) + 1e-8
        )

    return returns


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    env = DeliveryVehicle()
    policy = ContinuousPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # -----------------------------
    # Training
    # -----------------------------
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)

            mean, std = policy(state_tensor)
            distribution = Normal(mean, std)

            # Sample a continuous action from the Gaussian policy.
            raw_action = distribution.sample()
            action = torch.tanh(raw_action)

            # Save log probability of the sampled pre-tanh action.
            log_prob = distribution.log_prob(raw_action).sum()
            log_probs.append(log_prob)

            state, reward, done = env.step(action.detach().item())
            rewards.append(reward)

        returns = discounted_returns(rewards, GAMMA)

        # REINFORCE: encourage actions followed by higher returns.
        loss = torch.stack(
            [-log_prob * G for log_prob, G in zip(log_probs, returns)]
        ).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if episode % 200 == 0:
            print(
                f"Episode {episode:4d} | "
                f"episode reward: {sum(rewards):9.2f} | "
                f"final speed: {state[0]:5.2f} m/s"
            )

    # -----------------------------
    # Test with deterministic actions
    # -----------------------------
    state = env.reset()
    done = False
    speed_history = [float(state[0])]

    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        with torch.no_grad():
            mean, std = policy(state_tensor)
            action = torch.tanh(mean)  # Use the policy mean during testing.

        state, reward, done = env.step(action.item())
        speed_history.append(float(state[0]))

    print("\nTest episode")
    print(f"Initial speed: {speed_history[0]:.2f} m/s")
    print(f"Final speed:   {speed_history[-1]:.2f} m/s")
    print(f"Target speed:  {TARGET_SPEED:.2f} m/s")


if __name__ == "__main__":
    main()
