# The following codes are provided by the project.

# Ornstein–Uhlenbeck Noise
# We'll use a specific noise process that has some desired properties, called the Ornstein–Uhlenbeck process. It essentially generates random samples from a Gaussian (Normal) distribution, but each sample affects the next one such that two consecutive samples are more likely to be closer together than further apart. In this sense, the process in Markovian in nature.
# Why is this relevant to us? We could just sample from Gaussian distribution, couldn't we? Yes, but remember that we want to use this process to add some noise to our actions, in order to encourage exploratory behavior. And since our actions translate to force and torque being applied to a quadcopter, we want consecutive actions to not vary wildly. Otherwise, we may not actually get anywhere! Imagine flicking a controller up-down, left-right randomly!
# Besides the temporally correlated nature of samples, the other nice thing about the OU process is that it tends to settle down close to the specified mean over time. When used to generate noise, we can specify a mean of zero, and that will have the effect of reducing exploration as we make progress on learning the task.
# Here is a sample implementation of the Ornstein-Uhlenbeck process that you can use.
import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state