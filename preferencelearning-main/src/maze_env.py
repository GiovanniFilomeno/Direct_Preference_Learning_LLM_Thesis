import matplotlib
from gymnasium import Env
from gymnasium import spaces
from rewards import ConstantReward, DistanceReward
from maze import Maze
import random
import numpy as np
import matplotlib.pyplot as plt
import time

random.seed(1)


def line(x1, x2, y1, y2, alpha, scale, ax):
    # Plots a line between scaled x1,y1 and x2,y2 with opacity alpha.
    return ax.plot(np.array([x1 * scale, x2 * scale]), np.array([y1 * scale, y2 * scale]), 'k-', alpha)


def intersect(p1, p2, q1, q2):
    # Checks if line between p1,p2 intersects one between q1,q2
    cp1 = np.cross(p2 - p1, q1 - p2)
    cp2 = np.cross(p2 - p1, q2 - p2)
    cp3 = np.cross(q2 - q1, p1 - q2)
    cp4 = np.cross(q2 - q1, p2 - q2)
    t1 = np.dot(cp1, cp2)
    t2 = np.dot(cp3, cp4)
    if t1 < 0 and t2 < 0:
        # If both end points of both lines are on either sides of the other line
        # the lines certainly intersect
        return True
    elif t1 > 0 or t2 > 0:
        # If either of the lines end points are on the same side of the other
        # they definitely do not intersect
        return False
    else:
        # Now for the equal to zero cases, as we cannot use cross product to test.
        # We know one of the end points is collinear to the other line.
        # i.e. 1 triplet out of 4 choices is collinear
        # Thus checking all 4 pairs along with corresponding cross product
        # zero cross product => collinear, along with -ve dot product => intersects!.
        if (np.dot(p1 - q1, p2 - q1) < 0 and cp1 == 0) or (np.dot(p1 - q2, p2 - q2) < 0 and cp2 == 0) \
                or (np.dot(q1 - p1, q2 - p1) < 0 and cp3 == 0) or (np.dot(q1 - p2, q2 - p2) < 0 and cp4 == 0):
            return True
        else:
            # Fails if two of the points perfectly coincide, but probably that's not going to happen frequently.
            return False
        

def generate_world(sz_fac, maze):
    # Generates a list of lines corresponding to the environment boundaries for intersection checking.
    maps = []
    for i in range(maze.nx):
        for j in range(maze.ny):
            # Flipping north and south as cells are numbered downwards
            if maze.cell_at(i, j).walls['S']:
                maps.append(np.array([i, i + 1, j + 1, j + 1]) * sz_fac)
            if maze.cell_at(i, j).walls['N']:
                maps.append(np.array([i, i + 1, j, j]) * sz_fac)
            if maze.cell_at(i, j).walls['E']:
                maps.append(np.array([i + 1, i + 1, j, j + 1]) * sz_fac)
            if maze.cell_at(i, j).walls['W']:
                maps.append(np.array([i, i, j, j + 1]) * sz_fac)
    return maps


def draw_map(sz_fac, maze, ax=None, alpha=0.5, prints=None):
    # function to plot environment map
    # ax contains pre-drawn plot, all lines are plotted on ax
    if not ax:
        ax = plt.gca()
    maps = []
    for i in range(maze.nx):
        for j in range(maze.ny):
            # Flipping north and south as cells are numbered downwards
            if maze.cell_at(i, j).walls['S']:
                maps.append(line(i, i + 1, j + 1, j + 1, alpha, sz_fac, ax))
            if maze.cell_at(i, j).walls['N']:
                maps.append(line(i, i + 1, j, j, alpha, sz_fac, ax))
            if maze.cell_at(i, j).walls['E']:
                maps.append(line(i + 1, i + 1, j, j + 1, alpha, sz_fac, ax))
            if maze.cell_at(i, j).walls['W']:
                maps.append(line(i, i, j, j + 1, alpha, sz_fac, ax))
    return maps


class MazeEnv(Env):
    def __init__(self, sz=3, maze=None, start=np.array([0.1, 0.1]), goal=np.array([1.0, 1.0]),
                 reward="distance", log=False, eval=False, dt=0.03, horizon=5, wall_penalty=10, slide=1, image_freq=20):

        """
        sz: The number of rows and columns in the maze
        maze: Pre-made custom maze, if it exists
        start: Agent start point, should be inside the unit square.
        goal: Agent goal point, should be inside the unit square.
        reward_fn: External reward function to use.
        log: If episode data must be logged to wandb
        dt: time step per episode step.
        horizon: length of episode.
        wall_penalty: penalty for bumping into walls
        slide: if 1, activates slide feature, which causes walls to be slippery and allow motion along tangent.

        """

        # Maze dimensions (ncols, nrows)
        nx, ny = sz, sz
        self.sz = sz
        # Maze entry position
        ix, iy = 0, 0
        # Map size factor, maze is upscaled by this factor
        # self.map_sz_fac = map_sz_fac
        # self.temp_rescale = 1 / (self.map_sz_fac * self.sz)

        # deprecated
        self.scale = np.array([0, 0])
        self.zero = np.array([0, 0])

        # self.knowledge = np.array([0, 0])
        # self.old_return = 0

        self.traj = []  # Stores last state of each episode
        self.episode = []  # Stores entire episode
        self.cur_return = 0  # Accumulates current episode return
        self.log = log  # Whether to log above stats to wandb along with plot
        self.eval = eval  # Whether to log eval metrics

        # parameters for gym Env
        self.action_shape = 2
        self.observation_shape = 2
        self.action_space = spaces.Box(
            low=-np.full(self.action_shape, 1.0, dtype=np.float32),
            high=np.full(self.action_shape, 1.0, dtype=np.float32), shape=(2,),
            dtype=np.float32
        )
        # TODO Possible source of error: observation low and high are set to 0 and 1 (magic numbers)
        self.observation_space = spaces.Box(
            low=np.full(self.observation_shape, 0.0, dtype=np.float32),
            high=np.full(self.observation_shape, 1.0, dtype=np.float32), shape=(2,),
            dtype=np.float32
        )

        self.maze = Maze(nx, ny, ix, iy)
        self.dt = dt  # Should be sufficient to reach goal within horizon (including noise)

        # Time step counter and episode horizon
        self.episode_counter = 0
        self.image_freq = image_freq
        self.counter = 0
        self.horizon = horizon
        npoints = 25

        if maze:
            self.maze = maze
        else:
            self.maze.make_maze()

        self.worldlines = generate_world(1 / self.sz, self.maze)
        self.state = np.array(start)
        self.goal = np.array(goal)
        self.wall_penalty = wall_penalty
        self.slide = slide

        if reward == "distance":
            self.reward_fn = DistanceReward(goal=self.goal)
        else:
            self.reward_fn = ConstantReward()

        # Since reward is directly linked to obs, need to pre-compute full array according to observational limits
        x = np.linspace(0, 1, npoints)
        y = np.linspace(0, 1, npoints)
        self.X, self.Y = np.meshgrid(x, y)  # Needed for plotting contour
        self.points = np.stack((self.X, self.Y), axis=-1)
        self.Z = None  # To store reward fn output once. Reward function may be changed in main code after init,
        # hence not init here
        self.cb = None  # To store colorbar reference, so that it can be removed everytime before redoing image

    def reset(self, state=np.array([0.1, 0.1])):
        # Resets env to state
        # Logs scatter map with final position
        # TODO need to move logging to logcallback
        # print("Resetting env!\n",self.state,self.cur_return)
        self.episode_counter += 1
        if self.log and not self.eval:
            # Only for training envs. Will not run in test envs. (eval is default false for envs)
            self.traj.append(self.state)

            if self.episode_counter % self.image_freq == 0:
                maps = draw_map(1 / self.sz, self.maze)  # maze map already has sz so passing remaining factor

                if self.Z is None:
                    self.Z = self.reward_fn.compute_reward(self.points)  # negating reward to standardize plot
                maps.append(plt.contourf(self.X, self.Y, self.Z, 30, cmap='viridis_r'))
                if self.cb is not None:
                    self.cb.remove()
                self.cb = plt.colorbar()
                plt.scatter(np.array(self.traj)[:, 0], np.array(self.traj)[:, 1], c=np.linspace(0, 1, len(self.traj)),
                            cmap="hot")
                plt.axis("equal")

            #     wandb.log({"state": self.state, "return": self.cur_return,
            #                "goal_distance": np.sqrt(np.sum((self.state - (self.goal)) ** 2)),
            #                "trajectory": wandb.Image(plt), "episode": self.episode})
            #     plt.close()
            # else:
            #     wandb.log({"state": self.state, "return": self.cur_return,
            #                "goal_distance": np.sqrt(np.sum((self.state - (self.goal)) ** 2)),
            #                "episode": self.episode})

        self.state = state
        self.counter = 0
        self.cur_return = 0
        self.episode = []
        # Not adding first state
        # self.episode.append(self.state)
        return self.state

    def collision(self, new_pose):
        # True if new_pose crosses worldlines.
        for i in range(len(self.worldlines)):
            if intersect(self.state, new_pose, self.worldlines[i][[0, 2]], self.worldlines[i][[1, 3]]):
                return True
        return False

    def update_trackers(self, state,  penalty=0, done=False, infos={"A": 1.0}):
        # function to update current state, return, episode list and log evaluation stats if needed.
        # dummy infos
        self.prev_state = self.state
        self.state = state
        reward = self.reward_fn.get_reward(self.state) - penalty  # reward and env both use [0,1] domain.
        self.cur_return += reward
        self.episode.append(state)

        # observation, reward, terminated, truncated, info
        # TODO: Fix Truncated output for gymnasium
        return state, reward, done, False, infos

    def step(self, action):
        # action must be a vector. Does not support batching
        # Automatically clips action to avoid bad inputs
        action = np.clip(action, -1, 1)
        self.counter += 1  # incrementing step counter
        infos = {"A": 1.0}  # dummy infos
        done = False
        if self.counter >= self.horizon:
            done = True

        # TODO remove this from here, it should auto reset based on this condition
        if self.counter > self.horizon:
            print("Error: Env stepping after reset!!")
            # If env is not reset, you will have episodes > 101 time steps.
            return self.update_trackers(self.state, done=done, infos=infos)

        new_pose = self.state + (action[0]) * self.dt * (
            np.array([np.cos(action[1] * np.pi), np.sin(action[1] * np.pi)]))

        # Checking collisions
        for i in range(len(self.worldlines)):
            if intersect(self.state, new_pose, self.worldlines[i][[0, 2]], self.worldlines[i][[1, 3]]):
                if self.slide:
                    # print(new_pose, self.state)
                    newx = new_pose.copy()
                    newy = new_pose.copy()
                    # TODO assumes only horizontal or vertical walls
                    newx[1] = self.state[1]  # movement only along x
                    newy[0] = self.state[0]  # movement only along y
                    if not self.collision(newx):
                        return self.update_trackers(newx, penalty=self.wall_penalty, done=done,
                                                    infos=infos)
                    elif not self.collision(newy):
                        return self.update_trackers(newy, penalty=self.wall_penalty, done=done,
                                                    infos=infos)

                # if new position crosses walls + cannot slide, refuse movement
                return self.update_trackers(self.state, penalty=self.wall_penalty, done=done, infos=infos)

        return self.update_trackers(new_pose, done=done, infos=infos)

    def render(self, mode="human"):
        raise NotImplementedError

    def background(self, ax):
        maps = draw_map(1 / self.sz, self.maze, ax=ax)  # maze map already has sz so passing remaining factor
        if self.Z is None:
            self.Z = self.reward_fn.compute_reward(self.points)
        maps.append(ax.contourf(self.X, self.Y, self.Z, 30, cmap='viridis_r'))
        # maps.append(plt.colorbar())
        return maps