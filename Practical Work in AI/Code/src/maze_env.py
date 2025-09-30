import matplotlib
from gymnasium import Env
from gymnasium import spaces
from rewards import ConstantReward, DistanceReward
from maze import Maze
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from collections import deque

random.seed(1)


def cross2d(a, b):
    """Return the 2D scalar cross product: a.x*b.y - a.y*b.x."""
    return a[0]*b[1] - a[1]*b[0]


def line(x1, x2, y1, y2, alpha, scale, ax):
    """Plot a line between scaled (x1,y1) and (x2,y2) with opacity alpha."""
    return ax.plot(np.array([x1 * scale, x2 * scale]),
                   np.array([y1 * scale, y2 * scale]),
                   'k-', alpha=alpha)


def intersect(p1, p2, q1, q2, *, strict=False, eps=1e-9):
    """
    Return True if segments p1–p2 and q1–q2 intersect.
    With strict=False, intersection at a shared endpoint DOES count.
    """
    def cross(a, b):   # 2-D
        return a[0]*b[1] - a[1]*b[0]

    r, s = p2 - p1, q2 - q1
    rxs  = cross(r, s)
    q_p  = q1 - p1
    qpxr = cross(q_p, r)

    if abs(rxs) < eps and abs(qpxr) < eps:          # collinear
        t0 = np.dot(q1 - p1, r) / np.dot(r, r)
        t1 = np.dot(q2 - p1, r) / np.dot(r, r)
        t0, t1 = min(t0, t1), max(t0, t1)
        if strict:
            return t0 < 1 - eps and t1 > eps        # true overlap
        else:
            return t0 < 1 + eps and t1 > -eps       # include endpoints
    if abs(rxs) < eps:                              # parallel
        return False

    t = cross(q_p, s) / rxs
    u = cross(q_p, r) / rxs
    if strict:
        return (eps < t < 1 - eps) and (eps < u < 1 - eps)
    else:
        return (0 - eps <= t <= 1 + eps) and (0 - eps <= u <= 1 + eps)


def point_segment_distance(p, a, b):
    """Euclidean distance from point p to segment a–b."""
    ab = b - a
    t  = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


# ------------------------------------------------------------------
# Helper: distance from (x,y) to the nearest wall
# ------------------------------------------------------------------
def dist_to_wall_exact(env, x, y):
    p = np.array([x, y])
    dmin = np.inf
    for seg in env.worldlines:           # [x1,x2,y1,y2]
        a = np.array([seg[0], seg[2]])
        b = np.array([seg[1], seg[3]])
        dmin = min(dmin, point_segment_distance(p, a, b))
    return dmin            # [0..1]


def compute_dead_end_mask(maze, exits=None):
    """
    Return a boolean matrix (ny, nx):
        True  -> cell is in a dead end
        False -> cell is on a 'good' path (reaches border or goal)
    """
    nx, ny = maze.nx, maze.ny
    if exits is None:
        exits = set()
    # 1) build adjacency list + degree
    neigh = {}
    deg   = np.zeros((ny, nx), dtype=int)

    for y in range(ny):
        for x in range(nx):
            cell = maze.cell_at(x, y)
            nbs  = []
            if not cell.walls["N"]: nbs.append((x,   y-1))
            if not cell.walls["S"]: nbs.append((x,   y+1))
            if not cell.walls["W"]: nbs.append((x-1, y))
            if not cell.walls["E"]: nbs.append((x+1, y))
            neigh[(x, y)] = nbs
            deg[y, x]     = len(nbs)

    # border cells -> 'good' exits
    for x in range(nx):
        exits.add((x, 0))
        exits.add((x, ny-1))
    for y in range(ny):
        exits.add((0, y))
        exits.add((nx-1, y))

    # 2) prune non-exit leaves
    dead = np.zeros((ny, nx), dtype=bool)
    Q = deque([(j, i)  # (x, y)
               for (i, j), d in np.ndenumerate(deg)
               if d == 1 and (j, i) not in exits])

    while Q:
        x, y = Q.popleft()
        if dead[y, x]:
            continue
        dead[y, x] = True
        # decrease degree of neighbors
        for nx_, ny_ in neigh[(x, y)]:
            if dead[ny_, nx_]:
                continue
            deg[ny_, nx_] -= 1
            if deg[ny_, nx_] == 1 and (nx_, ny_) not in exits:
                Q.append((nx_, ny_))

    return dead


# ---------------------------------------------------------------
# Mask of horizontal corridors closed both above and below
# ---------------------------------------------------------------
def compute_horizontal_corridor_mask(maze):
    nx, ny = maze.nx, maze.ny
    mask = np.zeros((ny, nx), dtype=bool)

    for y in range(ny):
        x = 0
        while x < nx:
            cell = maze.cell_at(x, y)
            # start only if the cell is closed above and below
            if cell.walls["N"] and cell.walls["S"]:
                # find the continuous segment (N=S=True)
                x_start = x
                while x < nx and maze.cell_at(x, y).walls["N"] and maze.cell_at(x, y).walls["S"]:
                    x += 1
                x_end = x - 1  # inclusive

                # if both ends have W/E walls → corridor is truly closed
                left_wall  = maze.cell_at(x_start, y).walls["W"]
                right_wall = maze.cell_at(x_end,   y).walls["E"]
                if left_wall and right_wall:
                    mask[y, x_start:x_end+1] = True
            else:
                x += 1
    return mask


# ---------------------------------------------------------------
# Helper: matrix of traversable degree for each cell
# ---------------------------------------------------------------
def compute_degree_matrix(maze):
    nx, ny = maze.nx, maze.ny
    deg = np.zeros((ny, nx), dtype=np.int8)

    def open_both(c1, c2, w1, w2):
        return (not c1.walls[w1]) and (not c2.walls[w2])

    for y in range(ny):
        for x in range(nx):
            c = maze.cell_at(x, y)
            d = 0
            # north
            if y > 0 and open_both(c, maze.cell_at(x, y-1), "N", "S"):
                d += 1
            # south
            if y < ny-1 and open_both(c, maze.cell_at(x, y+1), "S", "N"):
                d += 1
            # west
            if x > 0 and open_both(c, maze.cell_at(x-1, y), "W", "E"):
                d += 1
            # east
            if x < nx-1 and open_both(c, maze.cell_at(x+1, y), "E", "W"):
                d += 1
            deg[y, x] = d
    return deg


def generate_world(sz_fac, maze):
    """Generate line segments for maze walls, scaled by sz_fac, for intersection checks."""
    maps = []
    for i in range(maze.nx):
        for j in range(maze.ny):
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
    """
    Draw the maze map.
    Fill fully-blocked cells (all walls True) in black.
    Return the axes used for drawing.
    """
    if ax is None:
        ax = plt.gca()

    # Fully closed cells => solid black rectangle
    for j in range(maze.ny):
        for i in range(maze.nx):
            cell = maze.cell_at(i, j)
            if all(cell.walls.values()):
                rect = plt.Rectangle(
                    (i * sz_fac, j * sz_fac),
                    sz_fac, sz_fac,
                    color='black', alpha=1.0
                )
                ax.add_patch(rect)

    # Draw walls
    for i in range(maze.nx):
        for j in range(maze.ny):
            if maze.cell_at(i, j).walls['S']:
                line(i, i + 1, j + 1, j + 1, alpha, sz_fac, ax)
            if maze.cell_at(i, j).walls['N']:
                line(i, i + 1, j, j, alpha, sz_fac, ax)
            if maze.cell_at(i, j).walls['E']:
                line(i + 1, i + 1, j, j + 1, alpha, sz_fac, ax)
            if maze.cell_at(i, j).walls['W']:
                line(i, i, j, j + 1, alpha, sz_fac, ax)

    return ax


# Define the Policy Network with configurable depth, hidden size, and dropout
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, dropout_prob=0.0):
        super(PolicyNetwork, self).__init__()

        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MazeEnv(Env):
    def __init__(self, sz=3, maze=None, start=np.array([0.1, 0.1]),
                 goal=np.array([1.0, 1.0]),
                 reward="distance", log=False, eval=False,
                 dt=0.03, horizon=5, wall_penalty=10, slide=1, image_freq=20,
                 use_dpo=False, dpo_model_path="best_dpo_policy.pth", hidden_dim=32, num_layers=3, dropout_prob=0.0):

        nx, ny = sz, sz
        self.sz = sz
        ix, iy = 0, 0

        self.traj = []
        self.episode = []
        self.cur_return = 0
        self.log = log
        self.eval = eval

        self.action_shape = 2
        self.observation_shape = 2
        self.action_space = spaces.Box(
            low=-np.full(self.action_shape, 1.0, dtype=np.float32),
            high=np.full(self.action_shape, 1.0, dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.full(self.observation_shape, 0.0, dtype=np.float32),
            high=np.full(self.observation_shape, 1.0, dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.maze = Maze(nx, ny, ix, iy)
        self.dt = dt

        self.episode_counter = 0
        self.image_freq = image_freq
        self.counter = 0
        self.horizon = horizon

        npoints = 50
        self.maze.make_maze_fail()

        self.worldlines = generate_world(1 / self.sz, self.maze)
        self.state = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.wall_penalty = wall_penalty
        self.slide = slide

        if reward == "distance":
            self.reward_fn = DistanceReward(goal=self.goal)
        else:
            self.reward_fn = ConstantReward()

        x = np.linspace(0, 1, npoints)
        y = np.linspace(0, 1, npoints)
        self.X, self.Y = np.meshgrid(x, y)
        self.points = np.stack((self.X, self.Y), axis=-1)
        self.Z = None
        self.cb = None

        self.use_dpo = use_dpo

        if self.use_dpo:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            # Load policy network
            self.policy_net = PolicyNetwork(hidden_dim=hidden_dim,
                                            num_layers=num_layers,
                                            dropout_prob=dropout_prob).to(self.device)
            self.policy_net.load_state_dict(torch.load(dpo_model_path,
                                                       map_location=self.device))
            self.policy_net.eval()
            # Load normalization stats used during training
            stats = np.load("../tests/norm_stats.npz")
            self._mean = torch.tensor(stats["mean"], dtype=torch.float32,
                                      device=self.device)
            self._std  = torch.tensor(stats["std"] + 1e-8, dtype=torch.float32,
                                      device=self.device)

    def evaluate_state_with_dpo(self, state):
        """
        Evaluate the current state with the DPO network using
        precomputed self._mean / self._std for normalization.
        """
        if not torch.is_tensor(self._mean):
            self._mean = torch.as_tensor(self._mean, dtype=torch.float32,
                                         device=self.device)
            self._std  = torch.as_tensor(self._std,  dtype=torch.float32,
                                         device=self.device)

        x_std = (torch.as_tensor(state, dtype=torch.float32,
                                 device=self.device) - self._mean) / self._std
        with torch.no_grad():
            return self.policy_net(x_std).item()

    def sample_open_state_continuous(self):
        """Sample a continuous position inside any non-fully-closed cell, avoiding walls."""
        open_cells = []
        for y_idx in range(self.maze.ny):
            for x_idx in range(self.maze.nx):
                cell = self.maze.cell_at(x_idx, y_idx)
                if not all(cell.walls.values()):  # exclude fully closed cells
                    open_cells.append((x_idx, y_idx))

        sz_fac = 1 / self.sz
        min_offset = 0.001
        max_offset = 0.999

        # keep sampling until a valid point is found
        while True:
            x_idx, y_idx = random.choice(open_cells)
            x = (x_idx + np.random.uniform(min_offset, max_offset)) * sz_fac
            y = (y_idx + np.random.uniform(min_offset, max_offset)) * sz_fac

            # reject points too close to a wall
            if not self.point_collision(x, y):
                return np.array([x, y], dtype=float)

    def sample_open_state(self):
        return self.sample_open_state_continuous()

    def reset(self, state=None):
        print(f"Resetting environment. Previous state: {self.state}, Counter: {self.counter}")

        if state is None:
            print("Starting defined")
            state = self.sample_open_state()

        self.state = state
        self.episode_counter += 1

        if self.log and not self.eval:
            self.traj.append(self.state)
            if self.episode_counter % self.image_freq == 0:
                ax = draw_map(1 / self.sz, self.maze)
                if self.Z is None:
                    self.Z = self.reward_fn.compute_reward(self.points)
                cf = plt.contourf(self.X, self.Y, self.Z, 30, cmap='viridis_r')
                if self.cb is not None:
                    self.cb.remove()
                self.cb = plt.colorbar()
                plt.scatter(np.array(self.traj)[:, 0],
                            np.array(self.traj)[:, 1],
                            c=np.linspace(0, 1, len(self.traj)),
                            cmap="hot")
                plt.axis("equal")

        self.counter = 0
        self.cur_return = 0
        self.episode = []
        return self.state

    def collision(self, new_pose):
        """Return True if segment state→new_pose crosses any wall."""
        for i in range(len(self.worldlines)):
            # each entry is [x1, x2, y1, y2]
            if intersect(self.state, new_pose,
                         self.worldlines[i][[0, 2]],
                         self.worldlines[i][[1, 3]]):
                return True
        return False

    def segment_collision(self, p_from, p_to):
        """Return True if segment p_from→p_to intersects a wall."""
        for seg in self.worldlines:          # [x1,x2,y1,y2], already scaled
            if intersect(p_from, p_to,
                         seg[[0, 2]],        # q1
                         seg[[1, 3]], strict=True):  # q2
                return True
        return False

    def point_collision(self, x, y, epsilon=0.001):
        """
        Return True if the point (x,y) is invalid because:
        - it's outside the maze, or
        - it's inside a fully walled cell, or
        - it's within 'epsilon' of a partial wall inside an otherwise open cell.
        """
        cx = int(x * self.sz)
        cy = int(y * self.sz)

        # Out of bounds
        if cx < 0 or cx >= self.maze.nx or cy < 0 or cy >= self.maze.ny:
            return True

        # Cell lookup
        cell = self.maze.cell_at(cx, cy)

        # Fully closed cell
        if all(cell.walls.values()):
            return True

        local_x = x * self.sz - cx
        local_y = y * self.sz - cy

        # Proximity to walls within epsilon
        if cell.walls['N'] and local_y >= 1 - epsilon:
            return True
        if cell.walls['S'] and local_y <= epsilon:
            return True
        if cell.walls['E'] and local_x >= 1 - epsilon:
            return True
        if cell.walls['W'] and local_x <= epsilon:
            return True

        return False

    def update_trackers(self, state, penalty=0, done=False, infos=None):
        """Update state, compute reward (DPO or hand-crafted), and log episode info."""
        if infos is None:
            infos = {}
        self.prev_state = self.state
        self.state = state
        if self.use_dpo:
            new_val  = self.evaluate_state_with_dpo(self.state)
            prev_val = getattr(self, "_prev_dpo_val", new_val)
            reward   = (new_val - prev_val) - penalty
            # add a smooth positive shaping towards the goal
            d = np.linalg.norm(self.state - self.goal)
            reward += 5 * np.exp(-d / 0.02)
            self._prev_dpo_val = new_val
        else:
            reward = self.reward_fn.get_reward(self.state) - penalty
        self.cur_return += reward
        self.episode.append(state)
        return state, reward, done, False, infos

    def step(self, action, epsilon_goal=0.03):
        """One environment step with (magnitude, direction) action in polar form."""
        print(f"Step called. Counter: {self.counter}, Horizon: {self.horizon}")
        action = np.clip(action, -1, 1)
        self.counter += 1
        done = False
        infos = {}

        if self.counter >= self.horizon:
            done = True

        if np.linalg.norm(self.state - self.goal) < epsilon_goal:
            done = True

        new_pose = self.state + action[0] * self.dt * \
                   np.array([np.cos(action[1] * np.pi),
                             np.sin(action[1] * np.pi)])

        if self.collision(new_pose):
            if self.slide:
                newx = new_pose.copy()
                newy = new_pose.copy()
                newx[1] = self.state[1]
                newy[0] = self.state[0]
                if not self.collision(newx):
                    return self.update_trackers(newx, penalty=self.wall_penalty, done=done, infos=infos)
                elif not self.collision(newy):
                    return self.update_trackers(newy, penalty=self.wall_penalty, done=done, infos=infos)
            return self.update_trackers(self.state, penalty=self.wall_penalty, done=done, infos=infos)
        else:
            return self.update_trackers(new_pose, done=done, infos=infos)

    def render(self, mode="human"):
        raise NotImplementedError

    def background(self, ax):
        """Draw maze and reward field background onto the provided Axes."""
        ax_map = draw_map(1 / self.sz, self.maze, ax=ax)
        if self.Z is None:
            self.Z = self.reward_fn.compute_reward(self.points)
        cf = ax.contourf(self.X, self.Y, self.Z, 30, cmap='viridis_r')
        return [ax_map, cf]
