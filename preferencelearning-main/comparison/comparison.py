#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparison.py — pipeline completa esperimenti A/B/C/D:
- generazione punti
- feature & punteggi
- sampling preferenze
- training DPO
- rollouts & visualizzazioni

Cartella output: comparison/outputs/<test_id>/

Dipendenze: numpy, pandas, torch, matplotlib, tqdm, scipy
Repo paths: importa da ../src (MazeEnv, Maze, PolicyNetwork, utilities)
"""

import os
import sys
import math
import time
import json
import random
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# SciPy: usata solo per la euristica euclidea in A*
try:
    from scipy.spatial import distance as scipy_distance
except Exception:
    scipy_distance = None  # fallback

# ---------------------------------------------------------------------
# Importa dal progetto
# ---------------------------------------------------------------------
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR  = os.path.join(REPO_DIR, "src")
TESTS_DIR = os.path.join(REPO_DIR, "tests")
assert os.path.isdir(SRC_DIR), f"src non trovato in {SRC_DIR}"
sys.path.insert(0, SRC_DIR)

from maze_env import MazeEnv, dist_to_wall_exact, compute_degree_matrix, compute_horizontal_corridor_mask, PolicyNetwork
from maze import Maze

# ---------------------------------------------------------------------
# Utility di riproducibilità
# ---------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

# ---------------------------------------------------------------------
# Configurazioni esperimenti
# ---------------------------------------------------------------------
@dataclass
class DatasetConfig:
    # distribuzione dei punti
    n_random: int = 4000
    target_ratio_degree_nonzero: float = 0.5   # per "balanced"
    n_corridor_per_row: int = 20               # densità punti sui corridoi
    n_extra_goal: int = 50                     # punti extra vicino al goal
    add_wall_band: bool = False                # punti vicino ai muri
    grid_uniform: bool = False                 # griglia uniforme
    grid_nx: int = 64                          # risoluzione griglia

@dataclass
class ScoreWeights:
    # total_score = w_path*path_distance + w_goal*dist_goal - w_deg*degree - w_wall*dist_from_wall + w_gate*dist_gate
    w_path: float = 0.3
    w_wall: float = 0.2
    w_goal: float = 0.2
    w_deg:  float = 0.8
    w_gate: float = 0.1

@dataclass
class PairingConfig:
    n_pairs_total: int = 200_000
    frac_same_row: float = 0.4
    hard_ratio: float = 0.7
    thresh_delta_score: float = 0.05
    min_delta_deg: float = 0.001
    max_trials: int = 50_000_000

@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 150
    lr: float = 1e-3
    hidden_dim: int = 256
    num_layers: int = 4
    dropout_prob: float = 0.05
    early_stopping_patience: int = 10

@dataclass
class SolveConfig:
    dt: float = 0.15
    horizon_dpo: int = 180
    horizon_base: int = 180
    epsilon_goal: float = 0.106  # come nel tuo notebook
    plot_points: bool = False

@dataclass
class Experiment:
    test_id: str
    dataset: DatasetConfig
    weights: ScoreWeights
    pairing: PairingConfig
    train: TrainConfig
    solve: SolveConfig

# ---------------------------------------------------------------------
# Helpers per feature non presenti in src
# ---------------------------------------------------------------------
def _can_move(env: MazeEnv, cx: int, cy: int, nx: int, ny: int) -> bool:
    """Controlla limiti e muri per movimento da cella (cx,cy) -> (nx,ny)."""
    if nx < 0 or nx >= env.maze.nx or ny < 0 or ny >= env.maze.ny:
        return False
    # Non entrare in cella completamente chiusa
    if all(env.maze.cell_at(nx, ny).walls.values()):
        return False

    current_cell = env.maze.cell_at(cx, cy)
    next_cell    = env.maze.cell_at(nx, ny)
    dx, dy = nx - cx, ny - cy
    if dx == 0 and dy == -1:   # N
        if current_cell.walls["N"] or next_cell.walls["S"]:
            return False
    elif dx == 0 and dy == 1:  # S
        if current_cell.walls["S"] or next_cell.walls["N"]:
            return False
    elif dx == 1 and dy == 0:  # E
        if current_cell.walls["E"] or next_cell.walls["W"]:
            return False
    elif dx == -1 and dy == 0: # W
        if current_cell.walls["W"] or next_cell.walls["E"]:
            return False
    return True

from collections import deque

def _shortest_path_cells(env, s):
    cx = min(max(int(s[0]*env.sz), 0), env.maze.nx-1)
    cy = min(max(int(s[1]*env.sz), 0), env.maze.ny-1)
    gx = min(max(int(env.goal[0]*env.sz), 0), env.maze.nx-1)
    gy = min(max(int(env.goal[1]*env.sz), 0), env.maze.ny-1)

    def _can_move(cx, cy, nx, ny):
        if nx < 0 or nx >= env.maze.nx or ny < 0 or ny >= env.maze.ny: return False
        if all(env.maze.cell_at(nx, ny).walls.values()): return False
        c, n = env.maze.cell_at(cx, cy), env.maze.cell_at(nx, ny)
        dx, dy = nx - cx, ny - cy
        if   dx==0 and dy==-1 and (c.walls["N"] or n.walls["S"]): return False
        if   dx==0 and dy== 1 and (c.walls["S"] or n.walls["N"]): return False
        if   dx== 1 and dy==0  and (c.walls["E"] or n.walls["W"]): return False
        if   dx==-1 and dy==0  and (c.walls["W"] or n.walls["E"]): return False
        return True

    dist = { (cx,cy): 0 }
    Q = deque([(cx,cy)])
    while Q:
        ux, uy = Q.popleft()
        if (ux,uy) == (gx,gy): break
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            vx, vy = ux+dx, uy+dy
            if not _can_move(ux, uy, vx, vy): continue
            if (vx,vy) not in dist:
                dist[(vx,vy)] = dist[(ux,uy)] + 1
                Q.append((vx,vy))
    return dist.get((gx,gy), np.inf)

def _gate_x_for_row(env, y: float) -> float:
    row = min(max(int(y * env.sz), 0), env.maze.ny - 1)
    block = row // 3  # serpentina con step=3
    return 1.0 if (block % 2 == 0) else 0.0

def _dist_to_gate_state(env, s: np.ndarray) -> float:
    return abs(s[0] - _gate_x_for_row(env, s[1]))

def _shortest_path_distance(env: MazeEnv, x: float, y: float) -> float:
    """Costo in passi di cella dal punto continuo (x,y) alla cella goal via BFS/A* leggera."""
    cx, cy = min(int(x * env.sz), env.maze.nx - 1), min(int(y * env.sz), env.maze.ny - 1)
    gx, gy = min(int(env.goal[0]*env.sz), env.maze.nx - 1), min(int(env.goal[1]*env.sz), env.maze.ny - 1)

    from queue import PriorityQueue
    frontier = PriorityQueue()
    frontier.put((0.0, (cx, cy)))
    cost = {(cx, cy): 0.0}

    while not frontier.empty():
        _, (ux, uy) = frontier.get()
        if (ux, uy) == (gx, gy):
            break
        cur = cost[(ux, uy)]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            vx, vy = ux+dx, uy+dy
            if not _can_move(env, ux, uy, vx, vy):
                continue
            new_c = cur + 1.0
            if (vx, vy) not in cost or new_c < cost[(vx, vy)]:
                cost[(vx, vy)] = new_c
                # euristica euclidea
                if scipy_distance is not None:
                    h = scipy_distance.euclidean((vx, vy), (gx, gy))
                else:
                    h = math.hypot(vx-gx, vy-gy)
                frontier.put((new_c + h, (vx, vy)))
    return cost.get((gx, gy), np.inf)

# ---------------------------------------------------------------------
# Generazione posizioni
# ---------------------------------------------------------------------
def _generate_positions_random(env: MazeEnv, n: int) -> np.ndarray:
    pts = []
    for _ in range(n):
        pts.append(tuple(env.sample_open_state_continuous()))
    return np.asarray(pts, dtype=float)

def _generate_positions_balanced_degree(env: MazeEnv, n: int, target_ratio_nonzero: float, deg_mat: np.ndarray) -> np.ndarray:
    pts_zero, pts_nonzero = [], []
    while len(pts_zero) + len(pts_nonzero) < n:
        s = tuple(env.sample_open_state_continuous())
        cx, cy = min(int(s[0]*env.sz), env.maze.nx-1), min(int(s[1]*env.sz), env.maze.ny-1)
        deg = int(deg_mat[cy, cx])
        (pts_nonzero if deg > 0 else pts_zero).append(s)
        nz_needed = int(n * target_ratio_nonzero)
        z_needed  = n - nz_needed
        if len(pts_nonzero) >= nz_needed and len(pts_zero) >= z_needed:
            break
    pts = pts_nonzero[:nz_needed] + pts_zero[:z_needed]
    random.shuffle(pts)
    return np.asarray(pts, dtype=float)

def _generate_positions_corridors(env: MazeEnv, n_per_row: int) -> np.ndarray:
    """Punti equispaziati nelle righe aperte (corrispondenti a serpentina con step=3)."""
    pts = []
    for row in range(0, env.maze.ny, 3):
        y_c = (row + 0.5) / env.sz
        xs = np.linspace(0.05, 0.95, n_per_row)
        for x in xs:
            if not env.point_collision(x, y_c):
                pts.append((x, y_c))
    return np.asarray(pts, dtype=float)

def _generate_positions_goal(env: MazeEnv, n: int) -> np.ndarray:
    pts = []
    for _ in range(n):
        x = np.random.uniform(0.90, 0.99)
        y = np.random.uniform(0.90, 0.99)
        if not env.point_collision(x, y):
            pts.append((x, y))
    return np.asarray(pts, dtype=float)

def _generate_positions_wall_band(env: MazeEnv, n: int) -> np.ndarray:
    """Cintura vicino a muri: x o y in fasce 0-0.08 e 0.92-1.0."""
    pts = []
    bands = [(0.005, 0.08), (0.92, 0.995)]
    for _ in range(n):
        if random.random() < 0.5:
            # vertical band
            xb = random.choice(bands)
            x = np.random.uniform(*xb)
            y = np.random.uniform(0.02, 0.98)
        else:
            yb = random.choice(bands)
            y = np.random.uniform(*yb)
            x = np.random.uniform(0.02, 0.98)
        if not env.point_collision(x, y):
            pts.append((x, y))
    return np.asarray(pts, dtype=float)

def _generate_positions_grid(env: MazeEnv, nx: int) -> np.ndarray:
    xs = np.linspace(0.01, 0.99, nx)
    ys = np.linspace(0.01, 0.99, nx)
    pts = []
    for y in ys:
        for x in xs:
            if not env.point_collision(x, y):
                pts.append((x, y))
    return np.asarray(pts, dtype=float)

# ---------------------------------------------------------------------
# Feature & punteggi
# ---------------------------------------------------------------------
def _compute_scores(env: MazeEnv, positions: np.ndarray, deg_mat: np.ndarray) -> pd.DataFrame:
    rows = []
    goal = np.asarray(env.goal, dtype=float)

    # pre-calcolo maschera corridoi
    corr_mask = compute_horizontal_corridor_mask(env.maze)

    for (x, y) in positions:
        d_goal = np.linalg.norm(goal - np.array([x, y]))
        d_wall = dist_to_wall_exact(env, x, y)
        cx, cy = min(int(x * env.sz), env.maze.nx-1), min(int(y * env.sz), env.maze.ny-1)
        degree = int(deg_mat[cy, cx])
        path_d = _shortest_path_distance(env, x, y)

        rows.append((x, y, d_goal, d_wall, degree, path_d))

    df = pd.DataFrame(rows, columns=["x", "y", "distance_to_goal", "distance_from_wall", "degree", "path_distance"])

    # normalizzazioni (robuste)
    for col in ["distance_to_goal", "distance_from_wall", "path_distance"]:
        m = df[col].replace(np.inf, np.nan).max()
        if pd.isna(m) or m == 0:
            df[col] = 0.0
        else:
            df[col] = df[col].replace(np.inf, m).div(m)

    # degree normalizzato a [0,1]
    if df["degree"].max() > 0:
        df["degree"] = df["degree"] / df["degree"].max()
    else:
        df["degree"] = 0.0

    # distanza dal "varco" (serpentina step=3, varco lato alternato)
    varco_col = (df["y"] * env.sz).floordiv(3)  # blocco di 3 righe
    direction = (varco_col % 2 == 0)           # True => si va a dx
    col_varco = np.where(direction, 1.0, 0.0)   # 1 (dx) o 0 (sx) normalizzate
    df["distance_to_gate"] = np.abs(df["x"] - col_varco)
    m = df["distance_to_gate"].max()
    df["distance_to_gate"] = df["distance_to_gate"].div(m if m > 0 else 1.0)

    return df

def _apply_total_score(df: pd.DataFrame, w: ScoreWeights) -> pd.DataFrame:
    df = df.copy()
    df["total_score"] = (
        w.w_path * df["path_distance"] +
        w.w_goal * df["distance_to_goal"] -
        w.w_deg  * df["degree"] -
        w.w_wall * df["distance_from_wall"] +
        w.w_gate * df["distance_to_gate"]
    )
    return df

# ---------------------------------------------------------------------
# Sampling coppie preferenza
# ---------------------------------------------------------------------
def _sample_same_row_pairs(df: pd.DataFrame, n_pairs: int, min_dx: float = 0.25) -> List[Tuple[float,float,float,float,int]]:
    pairs = []
    grouped = df.groupby(pd.cut(df.y, bins=np.arange(0, 1.01, 0.1), include_lowest=True))
    buckets = list(grouped)
    while len(pairs) < n_pairs:
        _, g = random.choice(buckets)
        if len(g) < 2:
            continue
        a, b = g.sample(2).itertuples()
        if abs(a.x - b.x) < min_dx:
            continue
        better, worse = (a, b) if a.total_score < b.total_score else (b, a)
        pairs.append((better.x, better.y, worse.x, worse.y, 1))
    return pairs

def _sample_pairs(df: pd.DataFrame,
                  num_pairs: int,
                  hard_ratio: float,
                  thresh: float,
                  min_delta_deg: float,
                  max_trials: int) -> List[Tuple[float,float,float,float,int]]:
    idx = df.index.to_numpy()
    scores = df["total_score"].to_numpy()
    degrees = df["degree"].to_numpy()
    pairs = set()
    n_hard = int(num_pairs * hard_ratio)
    trials = 0
    pbar = tqdm(total=num_pairs, desc="sampling pairs")
    while len(pairs) < num_pairs and trials < max_trials:
        trials += 1
        i, j = np.random.choice(idx, 2, replace=False)
        si, sj = scores[i], scores[j]
        if si == sj:
            continue

        better, worse = (i, j) if si < sj else (j, i)
        delta = abs(si - sj)
        want_hard = len(pairs) < n_hard
        if (want_hard and delta >= thresh) or (not want_hard and delta < thresh):
            continue

        if abs(degrees[better] - degrees[worse]) < min_delta_deg:
            continue

        pair = (df.at[better, "x"], df.at[better, "y"],
                df.at[worse,  "x"], df.at[worse,  "y"], 1)
        if pair not in pairs:
            pairs.add(pair)
            pbar.update(1)
    pbar.close()
    if len(pairs) < num_pairs:
        print(f"⚠️ solo {len(pairs)} coppie generate dopo {trials} tentativi (min_delta_deg troppo alto?).")
    return list(pairs)

# ---------------------------------------------------------------------
# Training DPO
# ---------------------------------------------------------------------
def _train_dpo(preferences_path: str, out_model_path: str, cfg: TrainConfig, device: Optional[str] = None) -> Dict:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print(f"[DPO] device = {device}")

    df_pref = pd.read_parquet(preferences_path)

    # split
    train_df = df_pref.sample(frac=0.90, random_state=42)
    temp_df  = df_pref.drop(train_df.index)
    val_df   = temp_df.sample(frac=0.50, random_state=42)
    test_df  = temp_df.drop(val_df.index)

    # --- CORREZIONE DATA LEAKAGE ---
    # Calcoliamo SEMPRE le statistiche sul dataset corrente (train_df o df_pref)
    # Non controlliamo se il file esiste, lo sovrascriviamo per specificità del test.
    
    norm_stats_path = os.path.join(TESTS_DIR, "norm_stats.npz")
    
    print(f"[DPO] Calcolo nuove statistiche di normalizzazione per questo test...")
    # Calcola su tutto il df_pref o solo train_df (meglio train_df per rigore assoluto)
    # Qui usiamo df_pref per coerenza col tuo codice precedente
    mean = np.array([df_pref[["x_better","y_better"]].stack().mean(),
                     df_pref[["x_better","y_better"]].stack().mean()], dtype=np.float32)
    std  = np.array([df_pref[["x_better","y_better"]].stack().std(),
                     df_pref[["x_better","y_better"]].stack().std()], dtype=np.float32) + 1e-8
    
    np.savez(norm_stats_path, mean=mean, std=std)
    print(f"[DPO] Salvato {norm_stats_path} (mean={mean[0]:.3f}, std={std[0]:.3f})")
    
    # ... poi definisci la classe PreferenceDataset ...

    class PreferenceDataset(Dataset):
        def __init__(self, df):
            norm = np.load(norm_stats_path)
            mean = norm["mean"].astype(np.float32)
            std  = norm["std"].astype(np.float32) + 1e-8
            self.xb = ((df[["x_better","y_better"]].values - mean) / std).astype(np.float32)
            self.xw = ((df[["x_worse", "y_worse"] ].values - mean) / std).astype(np.float32)
            self.y  = df["preference"].values.astype(np.float32)

        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            return (torch.tensor(self.xb[i], dtype=torch.float32),
                    torch.tensor(self.xw[i], dtype=torch.float32),
                    torch.tensor(self.y[i],  dtype=torch.float32))

    train_ds = PreferenceDataset(train_df)
    val_ds   = PreferenceDataset(val_df)
    test_ds  = PreferenceDataset(test_df)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    model = PolicyNetwork(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, dropout_prob=cfg.dropout_prob).to(device)

    def dpo_loss(model, xb, xw, m=0.25):
        return torch.clamp(m - (model(xb) - model(xw)), min=0).mean()

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)

    best_val = float("inf")
    no_imp = 0
    train_losses, val_losses = [], []

    for epoch in range(cfg.epochs):
        # train
        model.train()
        tr_loss = 0.0
        for xb, xw, _ in train_dl:
            xb, xw = xb.to(device), xw.to(device)
            opt.zero_grad()
            loss = dpo_loss(model, xb, xw)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_dl))
        train_losses.append(tr_loss)

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, xw, _ in val_dl:
                xb, xw = xb.to(device), xw.to(device)
                va_loss += dpo_loss(model, xb, xw).item()
        va_loss /= max(1, len(val_dl))
        val_losses.append(va_loss)

        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), out_model_path)
            no_imp = 0
            print(f"[DPO] epoch {epoch+1}: new best val {va_loss:.4f} -> saved {out_model_path}")
        else:
            no_imp += 1

        print(f"[DPO] epoch {epoch+1}/{cfg.epochs} | train {tr_loss:.4f} | val {va_loss:.4f}")
        if no_imp >= cfg.early_stopping_patience:
            print(f"[DPO] early stopping at epoch {epoch+1}")
            break

    # test con best model
    model.load_state_dict(torch.load(out_model_path, map_location=device))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, xw, _ in test_dl:
            xb, xw = xb.to(device), xw.to(device)
            rb, rw = model(xb), model(xw)
            correct += (rb > rw).sum().item()
            total   += xb.size(0)
    test_acc = correct / max(1, total)
    print(f"[DPO] test accuracy = {test_acc:.4f}")

    # salva losses
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val": best_val,
        "test_accuracy": test_acc
    }

# ---------------------------------------------------------------------
# Policy & rollout (adattate dai tuoi notebook)
# ---------------------------------------------------------------------
def _candidate_actions(env: MazeEnv) -> np.ndarray:
    angles = np.linspace(-1, 1, 32, endpoint=False, dtype=np.float32)
    # velocità più fini per “scorrere” lungo i muri
    fine = [0.002, 0.005, 0.008]
    speeds = fine + [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    cell = 1.0 / env.sz
    speeds = [0.5*cell/env.dt, 0.8*cell/env.dt, cell/env.dt] + speeds
    a, s = np.meshgrid(angles, speeds)
    return np.stack([s.ravel(), a.ravel()], axis=1)

def _is_legal(env: MazeEnv, s_from: np.ndarray, s_to: np.ndarray, n: int = 4) -> bool:
    """False se QUALSIASI punto intermedio tocca un muro; e se il segmento interseca mura (strict)."""
    alphas = np.linspace(0., 1., n)
    for a in alphas:
        p = (1-a)*s_from + a*s_to
        if env.point_collision(p[0], p[1]):
            return False
    return not env.segment_collision(s_from, s_to)

def _policy_dist_safe(env: MazeEnv, tol: float = 1e-6) -> np.ndarray:
    acts = _candidate_actions(env)
    cur_d = np.linalg.norm(env.goal - env.state)
    best, best_d = None, cur_d
    for a in acts:
        dx = a[0]*np.cos(a[1]*np.pi)*env.dt
        dy = a[0]*np.sin(a[1]*np.pi)*env.dt
        s1 = env.state + np.array([dx, dy])
        if not _is_legal(env, env.state, s1):  # evita collisioni
            continue
        d = np.linalg.norm(env.goal - s1)
        if d < best_d - tol:
            best_d, best = d, a
    return best if best is not None else np.zeros(2, dtype=np.float32)

def _to_std_tensor(x: np.ndarray, env_dpo: MazeEnv, device) -> "torch.Tensor":
    import torch
    # MazeEnv carica self._mean / self._std dal file tests/norm_stats.npz
    mean_t = env_dpo._mean if torch.is_tensor(env_dpo._mean) else torch.as_tensor(env_dpo._mean, dtype=torch.float32, device=device)
    std_t  = env_dpo._std  if torch.is_tensor(env_dpo._std)  else torch.as_tensor(env_dpo._std,  dtype=torch.float32, device=device)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    return (x_t - mean_t) / std_t

def _policy_baseline_lexico(env, tol=1e-6):
    acts = _candidate_actions(env)
    next_states = env.state + np.stack(
        [acts[:,0]*np.cos(acts[:,1]*np.pi)*env.dt,
         acts[:,0]*np.sin(acts[:,1]*np.pi)*env.dt], axis=1
    )
    # mosse lecite secondo lo step
    legal = [i for i in range(len(acts)) if not env.collision(next_states[i])]
    if not legal:
        return np.zeros(2, dtype=np.float32)

    d0 = float(np.linalg.norm(env.goal - env.state))
    paths = np.array([_shortest_path_cells(env, next_states[i]) for i in legal])
    dists = np.array([np.linalg.norm(env.goal - next_states[i]) for i in legal])

    # 1) minimizza path
    pmin = np.min(paths)
    cand = np.where(paths <= pmin + 1e-9)[0]
    if len(cand) > 1:
        # 2) a parità di path, minimizza distanza euclidea
        idx = cand[np.argmin(dists[cand])]
    else:
        idx = cand[0]

    # se nessun miglioramento di path è possibile (pmin == path attuale),
    # usa fallback gate-aware evitando ping-pong
    p0 = _shortest_path_cells(env, env.state)
    if pmin >= p0 - 1e-9:
        best, best_g = None, np.inf
        for j in legal:
            g = _dist_to_gate_state(env, next_states[j])
            if g < best_g:
                best_g, best = g, j
        if best is not None:
            idx = best

    return acts[legal[idx]]


def _policy_dpo_two_step_safe(env: MazeEnv, device, tol_val=1e-3, tol_dist=1e-3):
    import torch
    acts = _candidate_actions(env)
    cur_d = float(np.linalg.norm(env.goal - env.state))
    p0 = _shortest_path_cells(env, env.state)

    next_states_1 = env.state + np.stack(
        [acts[:,0]*np.cos(acts[:,1]*np.pi)*env.dt,
         acts[:,0]*np.sin(acts[:,1]*np.pi)*env.dt], axis=1
    )
    legal1 = [i for i in range(len(acts)) if not env.collision(next_states_1[i])]
    if not legal1:
        return np.zeros(2, dtype=np.float32)

    # tieni sia la 'key' per il confronto (score, -path, -dist)
    # sia i dati "reali" (score, path, dist) per la decisione
    best_two, best_two_key, best_two_data = None, None, None
    best_one, best_one_key, best_one_data = None, None, None

    with torch.no_grad():
        for i in legal1:
            s1 = next_states_1[i]
            # secondo passo
            s2_all = s1 + np.stack(
                [acts[:,0]*np.cos(acts[:,1]*np.pi)*env.dt,
                 acts[:,0]*np.sin(acts[:,1]*np.pi)*env.dt], axis=1
            )
            scores = env.policy_net(_to_std_tensor(s2_all, env, device)).cpu().numpy().squeeze()

            # scegli il s2 lecito con score max; tie‑break: path min → dist min
            order = np.argsort(scores)[::-1]
            chosen = None
            data_tup = None
            key_tup  = None
            for k in order:
                s2 = s2_all[k]
                # if env.collision(s2):
                #     continue
                if not _is_legal(env, s1, s2):
                    continue
                p2 = _shortest_path_cells(env, s2)
                d2 = float(np.linalg.norm(env.goal - s2))
                sc = float(scores[k])
                chosen  = acts[i]
                data_tup = (sc, p2, d2)
                key_tup  = (sc, -p2, -d2)  # max su score, min su path/dist
                break
            if chosen is None:
                continue

            if best_two_key is None or key_tup > best_two_key:
                best_two, best_two_key, best_two_data = chosen, key_tup, data_tup

            # record a 1 passo (per fallback)
            p1 = _shortest_path_cells(env, s1)
            d1 = float(np.linalg.norm(env.goal - s1))
            data1 = (data_tup[0], p1, d1)
            key1  = (data_tup[0], -p1, -d1)
            if best_one_key is None or key1 > best_one_key:
                best_one, best_one_key, best_one_data = acts[i], key1, data1

    # decisione: usa i valori reali (non negati)
    if best_two is not None:
        sc2, p2, d2 = best_two_data
        if (p2 < p0 - 1e-9) or (d2 < cur_d - tol_dist):
            return best_two

    if best_one is not None:
        sc1, p1, d1 = best_one_data
        if (p1 < p0 - 1e-9) or (d1 < cur_d - tol_dist):
            return best_one

    # fallback gate‑aware
    best, best_g = None, np.inf
    for i in legal1:
        s1 = next_states_1[i]
        g  = _dist_to_gate_state(env, s1)
        if g < best_g:
            best_g, best = g, i
    return acts[best] if best is not None else np.zeros(2, dtype=np.float32)


def _rollout(env, policy_fn, max_steps, epsilon_goal, device=None):
    traj, dists, paths = [env.state.copy()], [float(np.linalg.norm(env.goal - env.state))], [_shortest_path_cells(env, env.state)]
    goal_step = -1
    for step in range(max_steps):
        a = policy_fn(env)
        s_next, _, _, _, _ = env.step(action=a, epsilon_goal=epsilon_goal)
        traj.append(s_next.copy())
        dists.append(float(np.linalg.norm(env.goal - s_next)))
        paths.append(_shortest_path_cells(env, s_next))
        if dists[-1] < epsilon_goal and goal_step < 0:
            goal_step = step + 1
            break
    return np.array(traj), dists, goal_step, paths

# ---------------------------------------------------------------------
# Costruzione dataset + training + solving per un esperimento
# ---------------------------------------------------------------------
def run_experiment(exp: Experiment, outputs_dir: str):
    set_seed(42)
    out_dir = os.path.join(outputs_dir, exp.test_id)
    os.makedirs(out_dir, exist_ok=True)

    # 1) ambiente & maze serpentina
    maze = Maze(10, 10, 0, 0)
    env  = MazeEnv(sz=10, maze=maze, start=np.array([0.05, 0.05]),
                   goal=np.array([0.95, 0.95]), reward="distance",
                   log=False, eval=False, dt=0.1, horizon=100,
                   wall_penalty=10, slide=1, image_freq=100)
    deg_mat = compute_degree_matrix(env.maze)

    # 2) genera posizioni
    pts = []

    # random
    if exp.dataset.n_random > 0:
        pts.append(_generate_positions_random(env, exp.dataset.n_random))

    # balanced degree
    if exp.dataset.target_ratio_degree_nonzero is not None and exp.dataset.target_ratio_degree_nonzero >= 0.0:
        n_bal = int(0)  # default 0; se vuoi aggiungerne includi qui
        if n_bal > 0:
            pts.append(_generate_positions_balanced_degree(env, n_bal, exp.dataset.target_ratio_degree_nonzero, deg_mat))

    # corridoi
    if exp.dataset.n_corridor_per_row and exp.dataset.n_corridor_per_row > 0:
        pts.append(_generate_positions_corridors(env, exp.dataset.n_corridor_per_row))

    # goal extra
    if exp.dataset.n_extra_goal and exp.dataset.n_extra_goal > 0:
        pts.append(_generate_positions_goal(env, exp.dataset.n_extra_goal))

    # wall band
    if exp.dataset.add_wall_band:
        pts.append(_generate_positions_wall_band(env, max(200, exp.dataset.n_random // 4)))

    # grid uniforme
    if exp.dataset.grid_uniform:
        pts.append(_generate_positions_grid(env, exp.dataset.grid_nx))

    positions = np.vstack(pts) if len(pts) > 0 else _generate_positions_random(env, 4000)

    # 3) feature & punteggi
    df = _compute_scores(env, positions, deg_mat)
    df = _apply_total_score(df, exp.weights)

    # salva posizioni
    pos_path = os.path.join(out_dir, "positions.parquet")
    df.to_parquet(pos_path, index=False)
    print(f"[{exp.test_id}] salvato {pos_path} ({len(df)} punti)")

    # 4) coppie preferenza (mix same-row + mix controllato hard/easy)
    n_same = int(exp.pairing.n_pairs_total * exp.pairing.frac_same_row)
    pairs_row = _sample_same_row_pairs(df, n_same)
    pairs_mix = _sample_pairs(df,
                              exp.pairing.n_pairs_total - len(pairs_row),
                              hard_ratio=exp.pairing.hard_ratio,
                              thresh=exp.pairing.thresh_delta_score,
                              min_delta_deg=exp.pairing.min_delta_deg,
                              max_trials=exp.pairing.max_trials)
    pairs = pairs_row + pairs_mix

    pref_path = os.path.join(out_dir, "preferences.parquet")
    pd.DataFrame(pairs, columns=["x_better","y_better","x_worse","y_worse","preference"]).to_parquet(pref_path, index=False)
    print(f"[{exp.test_id}] salvato {pref_path} ({len(pairs)} coppie)")

    # 5) training DPO
    model_path = os.path.join(out_dir, "best_dpo_policy.pth")
    hist = _train_dpo(pref_path, model_path, exp.train)

    # salvataggio curva loss
    fig = plt.figure(figsize=(7,4))
    plt.plot(hist["train_losses"], label="train")
    plt.plot(hist["val_losses"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"DPO Losses [{exp.test_id}]")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_losses.png"), dpi=150)
    plt.close(fig)

    # 6) solving / rollouts
    # DPO env
    dpo_env = MazeEnv(sz=10, maze=maze, start=np.array([0.05,0.05]),
                      goal=np.array([0.95,0.95]), reward="distance",
                      dt=exp.solve.dt, horizon=exp.solve.horizon_dpo, slide=1,
                      use_dpo=True, dpo_model_path=model_path,
                      hidden_dim=exp.train.hidden_dim, num_layers=exp.train.num_layers, dropout_prob=exp.train.dropout_prob)
    device = "mps" if (hasattr(__import__('torch').backends, 'mps') and __import__('torch').backends.mps.is_available()) else "cpu"

    # -- subito dopo dpo_env = MazeEnv(... use_dpo=True, ...)
    import torch
    stats_path = os.path.join(TESTS_DIR, "norm_stats.npz")  # <repo>/tests/norm_stats.npz
    if os.path.isfile(stats_path):
        _norm = np.load(stats_path)
        dpo_env._mean = torch.tensor(_norm["mean"], dtype=torch.float32, device=dpo_env.device)
        dpo_env._std  = torch.tensor(_norm["std"] + 1e-8, dtype=torch.float32, device=dpo_env.device)
    else:
        print(f"⚠️  Norm stats non trovate in {stats_path} — la DPO userà ciò che ha caricato MazeEnv.")


    # baseline env (stesso maze)
    base_env = MazeEnv(sz=10, maze=maze, start=np.array([0.05,0.05]),
                       goal=np.array([0.95,0.95]), reward="distance",
                       dt=exp.solve.dt, horizon=exp.solve.horizon_base, slide=1, use_dpo=False)

    # traj_dpo, dist_dpo, iter_dpo = _rollout(dpo_env, lambda e: _policy_dpo_two_step_safe(e, device=device),
    #                                         max_steps=exp.solve.horizon_dpo, epsilon_goal=exp.solve.epsilon_goal)
    # traj_base, dist_base, iter_base = _rollout(base_env, _policy_dist_safe,
    #                                            max_steps=exp.solve.horizon_base, epsilon_goal=exp.solve.epsilon_goal)

    # DPO
    traj_dpo, dist_dpo, iter_dpo, path_dpo = _rollout(
        dpo_env, lambda e: _policy_dpo_two_step_safe(e, device=device),
        max_steps=exp.solve.horizon_dpo, epsilon_goal=exp.solve.epsilon_goal
    )
    # Baseline
    traj_base, dist_base, iter_base, path_base = _rollout(
        base_env, _policy_baseline_lexico,
        max_steps=exp.solve.horizon_base, epsilon_goal=exp.solve.epsilon_goal
    )


    # 7) plotting
    fig, axes = plt.subplots(1, 2, figsize=(13,5))

    axes[0].plot(path_dpo,  label="Preference Policy (path)", lw=1)
    axes[0].plot(path_base, label="Baseline (path)", lw=1, ls=":")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("euclidean distance to goal")
    axes[0].set_title("Distance-to-goal per step", fontsize=14)
    axes[0].legend(); axes[0].grid(True)

    # mappa + traiettorie
    from maze_env import draw_map as _draw_map
    _draw_map(1/dpo_env.sz, dpo_env.maze, ax=axes[1], alpha=1.0)

    if len(traj_base) > 0:
        axes[1].scatter(traj_base[:,0], traj_base[:,1], color="red", s=12, label="Baseline")
    else:
        axes[1].scatter([],[], color="red", s=12, label="Baseline")

    if len(traj_dpo) > 0:
        axes[1].scatter(traj_dpo[:,0], traj_dpo[:,1], color="blue", s=12, label="Preference Policy")
    else:
        axes[1].scatter([],[], color="blue", s=12, label="Preference Policy")

    axes[1].invert_xaxis(); axes[1].set_aspect("equal")
    axes[1].set_title("Trajectories", fontsize=14)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    # 8) salva report JSON
    report = {
        "test_id": exp.test_id,
        "n_points": len(df),
        "n_pairs": len(pairs),
        "training": {
            "best_val": float(hist["best_val"]),
            "test_accuracy": float(hist["test_accuracy"])
        },
        "solve": {
            "dpo_steps_to_goal": int(iter_dpo),
            "baseline_steps_to_goal": int(iter_base)
        },
        "paths": {
            "positions": pos_path,
            "preferences": pref_path,
            "model": model_path,
            "loss_plot": os.path.join(out_dir, "training_losses.png"),
            "comparison_plot": fig_path
        }
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{exp.test_id}] finito. Report in {os.path.join(out_dir, 'report.json')}")

# ---------------------------------------------------------------------
# Suite di esperimenti: A*, B*, C*, D*
# ---------------------------------------------------------------------
def build_experiments() -> Dict[str, Experiment]:
    base_dataset = DatasetConfig(
        n_random=4000,
        target_ratio_degree_nonzero=0.5,
        n_corridor_per_row=20,
        n_extra_goal=50,
        add_wall_band=False,
        grid_uniform=False
    )
    base_weights = ScoreWeights(0.3, 0.2, 0.2, 0.8, 0.1)
    base_pairing = PairingConfig(200_000, 0.4, 0.7, 0.05, 0.001, 50_000_000)
    base_train   = TrainConfig(128, 150, 1e-3, 256, 4, 0.05, 10)
    # base_solve   = SolveConfig(0.15, 180, 180, 0.106)
    base_solve = SolveConfig(dt=0.15, horizon_dpo=240, horizon_base=240, epsilon_goal=0.12)


    E: Dict[str, Experiment] = {}

    # --- A: distribuzione punti ---
    E["A1_random"] = Experiment("A1_random",
        dataset=DatasetConfig(n_random=6000, target_ratio_degree_nonzero=0.0, n_corridor_per_row=0, n_extra_goal=0, add_wall_band=False, grid_uniform=False),
        weights=base_weights, pairing=base_pairing, train=base_train, solve=base_solve)

    E["A2_balanced_degree"] = Experiment("A2_balanced_degree",
        dataset=DatasetConfig(n_random=2000, target_ratio_degree_nonzero=0.7, n_corridor_per_row=10, n_extra_goal=30, add_wall_band=False, grid_uniform=False),
        weights=base_weights, pairing=base_pairing, train=base_train, solve=base_solve)

    E["A3_corridor_heavy"] = Experiment("A3_corridor_heavy",
        dataset=DatasetConfig(n_random=2000, target_ratio_degree_nonzero=0.3, n_corridor_per_row=200, n_extra_goal=10, add_wall_band=False, grid_uniform=False),
        weights=base_weights, pairing=base_pairing, train=base_train, solve=base_solve)

    E["A4_goal_biased"] = Experiment("A4_goal_biased",
        dataset=DatasetConfig(n_random=2000, target_ratio_degree_nonzero=0.5, n_corridor_per_row=10, n_extra_goal=400, add_wall_band=False, grid_uniform=False),
        weights=base_weights, pairing=base_pairing, train=base_train, solve=base_solve)

    E["A5_wall_band"] = Experiment("A5_wall_band",
        dataset=DatasetConfig(n_random=2000, target_ratio_degree_nonzero=0.5, n_corridor_per_row=10, n_extra_goal=20, add_wall_band=True, grid_uniform=False),
        weights=base_weights, pairing=base_pairing, train=base_train, solve=base_solve)

    E["A6_uniform_grid"] = Experiment("A6_uniform_grid",
        dataset=DatasetConfig(n_random=0, target_ratio_degree_nonzero=0.0, n_corridor_per_row=0, n_extra_goal=0, add_wall_band=False, grid_uniform=True, grid_nx=72),
        weights=base_weights, pairing=base_pairing, train=base_train, solve=base_solve)

    # --- B: pesi delle preferenze ---
    E["B1_deg_strong"] = Experiment("B1_deg_strong",
        dataset=base_dataset, weights=ScoreWeights(w_path=0.1, w_wall=0.1, w_goal=0.1, w_deg=1.5, w_gate=0.05),
        pairing=base_pairing, train=base_train, solve=base_solve)

    E["B2_goal_strong"] = Experiment("B2_goal_strong",
        dataset=base_dataset, weights=ScoreWeights(w_path=0.2, w_wall=0.1, w_goal=1.2, w_deg=0.4, w_gate=0.05),
        pairing=base_pairing, train=base_train, solve=base_solve)

    E["B3_wide_space"] = Experiment("B3_wide_space",
        dataset=base_dataset, weights=ScoreWeights(w_path=0.2, w_wall=1.2, w_goal=0.2, w_deg=0.6, w_gate=0.1),
        pairing=base_pairing, train=base_train, solve=base_solve)

    E["B4_path_strong"] = Experiment("B4_path_strong",
        dataset=base_dataset, weights=ScoreWeights(w_path=1.1, w_wall=0.2, w_goal=0.2, w_deg=0.6, w_gate=0.1),
        pairing=base_pairing, train=base_train, solve=base_solve)

    E["B5_gate_strong"] = Experiment("B5_gate_strong",
        dataset=base_dataset, weights=ScoreWeights(w_path=0.2, w_wall=0.2, w_goal=0.2, w_deg=0.4, w_gate=1.0),
        pairing=base_pairing, train=base_train, solve=base_solve)

    # --- C: difficoltà / copertura delle coppie ---
    E["C1_easy_pairs"] = Experiment("C1_easy_pairs",
        dataset=base_dataset, weights=base_weights,
        pairing=PairingConfig(n_pairs_total=120_000, frac_same_row=0.4, hard_ratio=0.3, thresh_delta_score=0.05, min_delta_deg=0.0005, max_trials=20_000_000),
        train=base_train, solve=base_solve)

    E["C2_hard_pairs"] = Experiment("C2_hard_pairs",
        dataset=base_dataset, weights=base_weights,
        pairing=PairingConfig(n_pairs_total=200_000, frac_same_row=0.4, hard_ratio=0.9, thresh_delta_score=0.02, min_delta_deg=0.001, max_trials=50_000_000),
        train=base_train, solve=base_solve)

    E["C3_very_hard_diverse_deg"] = Experiment("C3_very_hard_diverse_deg",
        dataset=base_dataset, weights=base_weights,
        pairing=PairingConfig(n_pairs_total=200_000, frac_same_row=0.2, hard_ratio=0.9, thresh_delta_score=0.01, min_delta_deg=0.02, max_trials=60_000_000),
        train=base_train, solve=base_solve)

    # --- D: iperparametri di training ---
    E["D1_small_net"] = Experiment("D1_small_net",
        dataset=base_dataset, weights=base_weights, pairing=base_pairing,
        train=TrainConfig(batch_size=128, epochs=150, lr=1e-3, hidden_dim=64, num_layers=2, dropout_prob=0.0, early_stopping_patience=10),
        solve=base_solve)

    E["D2_dropout_heavy"] = Experiment("D2_dropout_heavy",
        dataset=base_dataset, weights=base_weights, pairing=base_pairing,
        train=TrainConfig(batch_size=128, epochs=150, lr=1e-3, hidden_dim=256, num_layers=4, dropout_prob=0.2, early_stopping_patience=10),
        solve=base_solve)

    E["D3_more_epochs"] = Experiment("D3_more_epochs",
        dataset=base_dataset, weights=base_weights, pairing=base_pairing,
        train=TrainConfig(batch_size=128, epochs=300, lr=1e-3, hidden_dim=256, num_layers=4, dropout_prob=0.05, early_stopping_patience=20),
        solve=base_solve)

    return E

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline comparativa preferenze/DPO")
    parser.add_argument("--tests", type=str, default="A1_random",
                        help="Lista test_id separati da virgola oppure 'ALL' per tutti")
    parser.add_argument("--outputs", type=str, default=os.path.join(REPO_DIR, "comparison", "outputs"),
                        help="Cartella di output")
    args = parser.parse_args()

    exps = build_experiments()
    if args.tests.upper() == "ALL":
        to_run = list(exps.keys())
    else:
        to_run = [t.strip() for t in args.tests.split(",") if t.strip() in exps]
        if not to_run:
            print(f"Nessun test valido tra {args.tests}. Test disponibili: {list(exps.keys())}")
            return

    os.makedirs(args.outputs, exist_ok=True)
    print(f"Esecuzione test: {to_run}")
    for tid in to_run:
        t0 = time.time()
        run_experiment(exps[tid], outputs_dir=args.outputs)
        print(f"== [{tid}] completato in {time.time() - t0:.1f}s ==")

if __name__ == "__main__":
    main()
