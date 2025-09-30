import numpy as np
import matplotlib.pyplot as plt


class ConstantReward:
    """Reward function that always returns 1."""
    def __init__(self):
        pass

    def get_reward(self, state):
        return 1.0

    def compute_reward(self, points):
        """Vectorized reward over a grid of points."""
        return np.ones(points.shape[:-1])


class DistanceReward:
    """
    Reward that increases as the state gets closer to `goal`.

    Parameters
    ----------
    reward_max : float
        Maximum reward at the goal.
    goal : np.ndarray
        Target position (expects shape (2,) for 2D).
    offsets : np.ndarray
        Unused (kept for interface compatibility).
    scale : float
        Unused (kept for interface compatibility).
    """
    def __init__(self, reward_max=1.0, goal=np.array([1.0, 1.0]),
                 offsets=np.array([1.0, 1.0]), scale=0.1):
        self.reward_max = reward_max
        self.goal = goal
        # Longest distance in a unit square (for 2D) is sqrt(2); generalizes to sqrt(dim)
        self.make_positive = np.sqrt(goal.shape[-1])
        self.reward_scale = reward_max / self.make_positive

    def get_reward(self, inputs):
        """
        Return a positive reward that is highest at the goal and
        decreases linearly with Euclidean distance.
        """
        return self.reward_scale * (self.make_positive - np.linalg.norm(self.goal - inputs, axis=-1))

    def compute_reward(self, points):
        """Vectorized wrapper over get_reward for a grid of points."""
        return self.get_reward(points)

    def visualize(self, start=0, end=10, points=10):
        """
        Plot filled contours of the reward field (2D only).

        Parameters
        ----------
        start, end : float
            Range for both x and y axes.
        points : int
            Number of samples per axis.
        """
        x = np.linspace(start, end, points)
        y = np.linspace(start, end, points)
        ncontours = 20

        X, Y = np.meshgrid(x, y)
        grid = np.concatenate((np.expand_dims(X, 2), np.expand_dims(Y, 2)), axis=2)
        Z = self.get_reward(grid)

        plt.contourf(X, Y, Z, ncontours, cmap='viridis_r')
        plt.axis("equal")
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    Rew = DistanceReward(goal=np.array([1.0, 1.0]))
    Rew.visualize(end=1)
