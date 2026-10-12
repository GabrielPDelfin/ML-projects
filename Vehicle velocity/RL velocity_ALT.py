import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
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
# Gymnasium environment
# -----------------------------
class DeliveryVehicleEnv(gym.Env):
    """Toy environment: control vehicle speed toward a target speed."""

    def __init__(self):
        super().__init__()

        # The policy's action is bounded between -1 and 1.
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation contains the current speed in m/s.
        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self.speed = 0.0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.speed = 0.0
        self.steps = 0
        observation = np.array([self.speed], dtype=np.float32)
        return observation, {}

    def step(self, action):
        # Map normalized action [-1, 1] to acceleration [-3, 3] m/s^2.
        normalized_action = float(np.clip(action[0], -1.0, 1.0))
        acceleration = MAX_ACCELERATION * normalized_action

        # Simplified vehicle dynamics.
        self.speed = max(0.0, self.speed + acceleration * DT)
        self.steps += 1

        # Reward higher when close to target speed and using less acceleration.
        reward = -(
            (self.speed - TARGET_SPEED) ** 2
            + 0.1 * acceleration**2
        )

        terminated = False
        truncated = self.steps >= EPISODE_LENGTH

        observation = np.array([self.speed], dtype=np.float32)
        info = {"acceleration": acceleration}
        return observation, float(reward), terminated, truncated, info


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

    # Normalize for somewhat more stable policy-gradient updates.
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (
            returns.std(unbiased=False) + 1e-8
        )

    return returns


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    env = DeliveryVehicleEnv()
    policy = ContinuousPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # -----------------------------
    # Training
    # -----------------------------
    for episode in range(1, EPISODES + 1):
        state, info = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)

            mean, std = policy(state_tensor)
            distribution = Normal(mean, std)

            # Sample an unconstrained Gaussian action, then squash it
            # into the environment's valid action range [-1, 1].
            raw_action = distribution.sample()
            action = torch.tanh(raw_action)

            # REINFORCE uses the log probability of the sampled
            # pre-tanh action. Tanh is a deterministic transform
            # of that sample.
            log_prob = distribution.log_prob(raw_action).sum()
            log_probs.append(log_prob)

            state, reward, terminated, truncated, info = env.step(
                action.detach().cpu().numpy().astype(np.float32)
            )
            rewards.append(reward)
            done = terminated or truncated

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
    state, info = env.reset()
    done = False
    speed_history = [float(state[0])]

    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        with torch.no_grad():
            mean, std = policy(state_tensor)
            action = torch.tanh(mean)  # Use the policy's mean at test time.

        state, reward, terminated, truncated, info = env.step(
            action.cpu().numpy().astype(np.float32)
        )
        speed_history.append(float(state[0]))
        done = terminated or truncated

    print("\nTest episode")
    print(f"Initial speed: {speed_history[0]:.2f} m/s")
    print(f"Final speed:   {speed_history[-1]:.2f} m/s")
    print(f"Target speed:  {TARGET_SPEED:.2f} m/s")

    env.close()


if __name__ == "__main__":
    main()
