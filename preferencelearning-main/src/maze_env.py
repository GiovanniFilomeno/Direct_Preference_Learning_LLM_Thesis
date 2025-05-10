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
        
def point_segment_distance(p, a, b):
    ab = b - a
    t  = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

# ------------------------------------------------------------------
# helper: distanza dal muro più vicino
# ------------------------------------------------------------------
def dist_to_wall_exact(env, x, y):
    p = np.array([x, y])
    dmin = np.inf
    for seg in env.worldlines:           # [x1,x2,y1,y2]
        a = np.array([seg[0], seg[2]])
        b = np.array([seg[1], seg[3]])
        dmin = min(dmin, point_segment_distance(p, a, b))
    return dmin            # [0..1]
        
# maze_env.py  (fuori dalla classe)

def compute_dead_end_mask(maze, exits=None):
    """
    Ritorna una matrice booleana (ny, nx):
        True  -> cella in un vicolo cieco
        False -> cella su percorso 'buono' (raggiunge bordo o goal)
    """
    nx, ny = maze.nx, maze.ny
    if exits is None:
        exits = set()
    # ---------------------------------------
    # 1) costruiamo lista adiacenze + grado
    # ---------------------------------------
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

    # celle di bordo -> uscite “buone”
    for x in range(nx):
        exits.add((x, 0))
        exits.add((x, ny-1))
    for y in range(ny):
        exits.add((0, y))
        exits.add((nx-1, y))

    # ---------------------------------------
    # 2) potatura foglie non-uscite
    # ---------------------------------------
    dead = np.zeros((ny, nx), dtype=bool)
    Q = deque([(j, i)          # (x, y)
           for (i, j), d in np.ndenumerate(deg) if d == 1 and (j, i) not in exits])


    while Q:
        x, y = Q.popleft()
        if dead[y, x]:
            continue
        dead[y, x] = True
        # riduci il grado dei vicini
        for nx_, ny_ in neigh[(x, y)]:
            if dead[ny_, nx_]:               # già potata
                continue
            deg[ny_, nx_] -= 1
            if deg[ny_, nx_] == 1 and (nx_, ny_) not in exits:
                Q.append((nx_, ny_))

    return dead

# ---------------------------------------------------------------
# mask di corridoi orizzontali chiusi sopra e sotto
# ---------------------------------------------------------------
def compute_horizontal_corridor_mask(maze):
    nx, ny = maze.nx, maze.ny
    mask = np.zeros((ny, nx), dtype=bool)

    for y in range(ny):
        x = 0
        while x < nx:
            cell = maze.cell_at(x, y)
            # inizia solo se quella cella è chiusa sopra e sotto
            if cell.walls["N"] and cell.walls["S"]:
                # individua tutto il segmento continuativo (N=S=True)
                x_start = x
                while x < nx and maze.cell_at(x, y).walls["N"] and maze.cell_at(x, y).walls["S"]:
                    x += 1
                x_end = x - 1          # ultima inclusa

                # se alle estremità troviamo muri W/E → è davvero chiuso
                left_wall  = maze.cell_at(x_start, y).walls["W"]
                right_wall = maze.cell_at(x_end,   y).walls["E"]
                if left_wall and right_wall:
                    mask[y, x_start:x_end+1] = True
            else:
                x += 1
    return mask

# ---------------------------------------------------------------
# helper: matrice del grado "traversabile"
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
            # nord
            if y > 0 and open_both(c, maze.cell_at(x, y-1), "N", "S"):
                d += 1
            # sud
            if y < ny-1 and open_both(c, maze.cell_at(x, y+1), "S", "N"):
                d += 1
            # ovest
            if x > 0 and open_both(c, maze.cell_at(x-1, y), "W", "E"):
                d += 1
            # est
            if x < nx-1 and open_both(c, maze.cell_at(x+1, y), "E", "W"):
                d += 1
            deg[y, x] = d
    return deg


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

# Define the Policy Network class for DPO
# Define the Policy Network with configurable depth, hidden size, and dropout
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, dropout_prob=0.0):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Add dropout if dropout_prob > 0
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
            self.policy_net = PolicyNetwork(hidden_dim=hidden_dim, 
                                            num_layers=num_layers, dropout_prob=dropout_prob).to(self.device)
            self.policy_net.load_state_dict(torch.load(dpo_model_path, map_location=self.device))
            self.policy_net.eval()

    def evaluate_state_with_dpo(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

        # -----------------------------------------------------------
        #  --- costanti usate nel training ---
        MEAN = torch.tensor([0.5, 0.5],  dtype=torch.float32, device=self.device)
        STD  = torch.tensor([0.289, 0.289], dtype=torch.float32, device=self.device)

        def _std(x: torch.Tensor) -> torch.Tensor:
            """centra e scala come nel training"""
            return (x - MEAN) / STD

        state_tensor = _std(state_tensor)  
        # -----------------------------------------------------------
        return self.policy_net(state_tensor).item()
    
    # Funzione per generare uno stato aperto continuo
    def sample_open_state_continuous(self):
        open_cells = []
        for y_idx in range(self.maze.ny): 
            for x_idx in range(self.maze.nx):
                cell = self.maze.cell_at(x_idx, y_idx) 
                if not all(cell.walls.values()):  # Escludi le celle completamente chiuse
                    open_cells.append((x_idx, y_idx))
        
        sz_fac = 1 / self.sz
        min_offset = 0.001
        max_offset = 0.999
        
        while True:  # Continua finché non trovi un punto valido
            x_idx, y_idx = random.choice(open_cells)
            x = (x_idx + np.random.uniform(min_offset, max_offset)) * sz_fac
            y = (y_idx + np.random.uniform(min_offset, max_offset)) * sz_fac

            # Se il punto cade su un muro, scartalo e riprova
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
        # True if new_pose crosses worldlines
        for i in range(len(self.worldlines)):
            # each entry is [x1, x2, y1, y2]
            if intersect(self.state, new_pose,
                         self.worldlines[i][[0, 2]],
                         self.worldlines[i][[1, 3]]):
                return True
        return False
    
    def point_collision(self, x, y, epsilon=0.001):
        """
        Return True if the point (x,y) is invalid because:
        - It's out of the maze, or
        - It's in a fully walled cell, or
        - It's within 'epsilon' of a partial wall inside an otherwise open cell.
        """
        cx = int(x * self.sz)
        cy = int(y * self.sz)

        # Out of bounds => collision
        if cx < 0 or cx >= self.maze.nx or cy < 0 or cy >= self.maze.ny:
            return True

        # Identify the cell
        cell = self.maze.cell_at(cx, cy)

        # If the cell is fully closed, that’s definitely a collision
        if all(cell.walls.values()):
            return True

        # --- Now check partial walls. ---
        # We'll compute the point's local coordinates in [0..1] within this cell.
        #   local_x = 0.0 means left edge of the cell
        #   local_x = 1.0 means right edge of the cell
        # (Similarly for local_y = 0..1.)
        local_x = x * self.sz - cx
        local_y = y * self.sz - cy

        # If there's a 'N' (north) wall => top edge is blocked => local_y=1 is not allowed
        if cell.walls['N']:
            if local_y >= 1 - epsilon:
                return True

        # If there's a 'S' (south) wall => bottom edge blocked => local_y=0 is not allowed
        if cell.walls['S']:
            if local_y <= epsilon:
                return True

        # If there's an 'E' (east) wall => right edge => local_x=1 is blocked
        if cell.walls['E']:
            if local_x >= 1 - epsilon:
                return True

        # If there's a 'W' (west) wall => left edge => local_x=0 is blocked
        if cell.walls['W']:
            if local_x <= epsilon:
                return True

        # If we get here, it's not out of bounds, not fully closed,
        # and not within epsilon of any partial wall => no collision
        return False


    def update_trackers(self, state, penalty=0, done=False, infos=None):
        if infos is None:
            infos = {}
        self.prev_state = self.state
        self.state = state
        if self.use_dpo:
            reward = self.evaluate_state_with_dpo(self.state) - penalty
        else:
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
        ax_map = draw_map(1 / self.sz, self.maze, ax=ax)
        if self.Z is None:
            self.Z = self.reward_fn.compute_reward(self.points)
        cf = ax.contourf(self.X, self.Y, self.Z, 30, cmap='viridis_r')
        return [ax_map, cf]
