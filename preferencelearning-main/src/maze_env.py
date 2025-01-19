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


def cross2d(a, b):
    """Ritorna lo scalare della cross 2D: a.x*b.y - a.y*b.x."""
    return a[0]*b[1] - a[1]*b[0]

def line(x1, x2, y1, y2, alpha, scale, ax):
    # Plots a line between scaled x1,y1 and x2,y2 with opacity alpha.
    return ax.plot(np.array([x1 * scale, x2 * scale]),
                   np.array([y1 * scale, y2 * scale]), 
                   'k-', alpha=alpha)

def intersect(p1, p2, q1, q2):
    """
    Checks if line segment p1->p2 intersects line segment q1->q2 (2D).
    Usa la cross in 2D (scalare) al posto di np.cross con padding.
    """
    # cross fra (p2-p1) e (q1-p2) etc., tutti scalari
    cp1 = cross2d(p2 - p1, q1 - p2)
    cp2 = cross2d(p2 - p1, q2 - p2)
    cp3 = cross2d(q2 - q1, p1 - q2)
    cp4 = cross2d(q2 - q1, p2 - q2)

    # t1 = segno di cp1 * cp2
    # t2 = segno di cp3 * cp4
    t1 = cp1 * cp2
    t2 = cp3 * cp4

    if t1 < 0 and t2 < 0:
        # se i segmenti 'strisciano' uno sull'altro, si intersecano
        return True
    elif t1 > 0 or t2 > 0:
        # Non si intersecano
        return False
    else:
        # Ora gestiamo il caso in cui cp1, cp2, cp3, cp4 = 0 => collinearità/overlap
        # Per semplicità, usiamo la logica del codice esistente:
        # np.dot(...) < 0 e cp1 == 0, ecc. 
        # Stavolta cp1, cp2, cp3, cp4 sono semplici float
        dot1 = np.dot(p1 - q1, p2 - q1)
        dot2 = np.dot(p1 - q2, p2 - q2)
        dot3 = np.dot(q1 - p1, q2 - p1)
        dot4 = np.dot(q1 - p2, q2 - p2)
        # Controllo collinearità
        if (dot1 < 0 and cp1 == 0) or \
           (dot2 < 0 and cp2 == 0) or \
           (dot3 < 0 and cp3 == 0) or \
           (dot4 < 0 and cp4 == 0):
            return True
        else:
            return False
        

def generate_world(sz_fac, maze):
    # Generates a list of lines corresponding to the environment boundaries for intersection checking.
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
    Disegna la mappa del maze.
    Colora in nero le celle 'bloccate' (tutte le pareti True).
    Ritorna l'axes su cui ha disegnato.
    """
    if ax is None:
        ax = plt.gca()

    # Celle completamente chiuse => rettangolo nero
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

    # Disegno i muri
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


from gymnasium import Env


class MazeEnv(Env):
    def __init__(self, sz=3, maze=None, start=np.array([0.1, 0.1]),
                 goal=np.array([1.0, 1.0]),
                 reward="distance", log=False, eval=False,
                 dt=0.03, horizon=5, wall_penalty=10, slide=1, image_freq=20):

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

        npoints = 25
        # Se hai passato un Maze esterno, usa quello:
        # if maze:
            # self.maze = maze
        # else:
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

    def sample_open_state(self):
        open_cells = []
        for j in range(self.maze.ny):
            for i in range(self.maze.nx):
                cell = self.maze.cell_at(i, j)
                if not all(cell.walls.values()):
                    open_cells.append((i, j))
        i, j = random.choice(open_cells)
        sz_fac = 1 / self.sz
        x = (i + 0.5) * sz_fac
        y = (j + 0.5) * sz_fac
        return np.array([x, y], dtype=float)

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
        # True if new_pose crosses worldlines
        for i in range(len(self.worldlines)):
            # each entry is [x1, x2, y1, y2]
            if intersect(self.state, new_pose,
                         self.worldlines[i][[0, 2]],
                         self.worldlines[i][[1, 3]]):
                return True
        return False

    def update_trackers(self, state, penalty=0, done=False, infos=None):
        if infos is None:
            infos = {}
        self.prev_state = self.state
        self.state = state
        reward = self.reward_fn.get_reward(self.state) - penalty
        self.cur_return += reward
        self.episode.append(state)
        return state, reward, done, False, infos

    def step(self, action):
        print(f"Step called. Counter: {self.counter}, Horizon: {self.horizon}")
        action = np.clip(action, -1, 1)
        self.counter += 1
        done = False
        infos = {}

        if self.counter >= self.horizon:
            done = True

        # Se sfori la horizon
        if self.counter > self.horizon:
            print("Error: Env stepping after reset!!")
            return self.update_trackers(self.state, done=done, infos=infos)

        # Nuova posa
        new_pose = self.state + action[0] * self.dt * \
                   np.array([np.cos(action[1] * np.pi),
                             np.sin(action[1] * np.pi)])

        # Collisione?
        if self.collision(new_pose):
            # se slide attivo, prova a muoverti solo in x oppure in y
            if self.slide:
                newx = new_pose.copy()
                newy = new_pose.copy()
                newx[1] = self.state[1]  # movimento solo su x
                newy[0] = self.state[0]  # movimento solo su y
                if not self.collision(newx):
                    return self.update_trackers(newx, penalty=self.wall_penalty, done=done, infos=infos)
                elif not self.collision(newy):
                    return self.update_trackers(newy, penalty=self.wall_penalty, done=done, infos=infos)

            # Se non si può scivolare, rimani fermo e penalizza
            return self.update_trackers(self.state, penalty=self.wall_penalty, done=done, infos=infos)
        else:
            return self.update_trackers(new_pose, done=done, infos=infos)

    def render(self, mode="human"):
        raise NotImplementedError

    def background(self, ax):
        ax_map = draw_map(1 / self.sz, self.maze, ax=ax)
        if self.Z is None:
            self.Z = self.reward_fn.compute_reward(self.points)
        cf = ax.contourf(self.X, self.Y, self.Z, 30, cmap='viridis_r')
        return [ax_map, cf]
