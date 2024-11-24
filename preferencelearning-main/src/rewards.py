import numpy as np
import matplotlib.pyplot as plt


# reward function which always returns 1
class ConstantReward:
    def __init__(self):
        pass

    def get_reward(self, state):
        return 1.0

    def compute_reward(self, points):
        return np.ones(points.shape[:-1])


class DistanceReward(object):
    def __init__(self, reward_max=1.0, goal=np.array([1.0, 1.0]), offsets=np.array([1.0, 1.0]), scale=0.1):
        """
        offsets: offset to be subtracted after scaling
        """
        # TODO remove scale and offsets
        # self.scale = scale
        self.reward_max = reward_max
        self.goal = goal
        self.make_positive = np.sqrt(goal.shape[-1]) # Assuming env diagonal is longest distance in the map
        self.reward_scale = reward_max / self.make_positive
        # self.offsets = offsets

    def get_reward(self, inputs):
        # reward must be negative as distance from goal is positive
        # Need to use actual state coordinates not normalized as reward is directly linked to observation.
        return self.reward_scale * (self.make_positive - np.linalg.norm(self.goal-inputs,axis=-1))
    
    def compute_reward(self, points):
        return self.get_reward(points)

    def visualize(self, start=0, end=10, points=10):
        # Problem with not creating new graph but using same graph to visualize, can spoil gradients!
        # TODO works only for 2D lyapunov functions
        x = np.linspace(start, end, points)
        y = np.linspace(start, end, points)
        ncontours = 20

        X, Y = np.meshgrid(x, y)
        Y0 = np.concatenate((np.expand_dims(X, 2), np.expand_dims(Y, 2)), 2)
        Z = self.get_reward(Y0)
        plt.contourf(X, Y, Z, ncontours, cmap='viridis_r');
        plt.axis("equal")
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    Rew = DistanceReward(goal=np.array([1.0, 1.0]))
    Rew.visualize(end=1)