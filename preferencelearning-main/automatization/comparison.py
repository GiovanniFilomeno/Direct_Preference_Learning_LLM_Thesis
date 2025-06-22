import numpy as np, torch, matplotlib.pyplot as plt
import os
import sys

current_dir = os.getcwd()
module_path = os.path.abspath(os.path.join(current_dir, "../src"))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np, torch, matplotlib.pyplot as plt
from maze import Maze
from maze_env import MazeEnv, draw_map
from maze_env import PolicyNetwork

def make_comparison ():

    horizon = 100
    horizon_base = 100
    dpo_path = 'best_dpo_policy.pth'

    def best_improving_action(env, acts, ref_metric, tol=1e-6):
        """
        Ritorna la prima azione che riduce di almeno `tol`
        il valore di ref_metric(s') rispetto a quello corrente.
        Se nessuna azione migliora, restituisce None.
        """
        best_a, best_val = None, ref_metric  # migliore finora (deve essere < ref)
        for a in acts:
            dx = a[0]*np.cos(a[1]*np.pi)*env.dt
            dy = a[0]*np.sin(a[1]*np.pi)*env.dt
            s1 = env.state + np.array([dx, dy])

            # qui usiamo la distanza al goal come metrica di riferimento
            val = np.linalg.norm(env.goal - s1)

            # opzionale: scarta subito se la mossa collide
            if env.point_collision(s1[0], s1[1]):
                continue

            if val < best_val - tol:
                best_val, best_a = val, a

        return best_a

    # ------------------------------------------------------------------
    # helper: rollout con politica arbitraria f(state) -> action
    # ------------------------------------------------------------------
    def rollout(env, policy_fn, max_steps=500):
        if max_steps is None:
            max_steps = env.horizon
        env.reset(state=start.copy())
        traj, dists = [env.state.copy()], [np.linalg.norm(env.goal - env.state)]

        goal_step = -1  # default = non raggiunto

        for step in range(max_steps):
            a = policy_fn(env)
            state, _, done, _, _ = env.step(a)
            traj.append(state.copy())
            dists.append(np.linalg.norm(env.goal - state))
            if done:
                if np.linalg.norm(env.goal - state) < 0.03:
                    print(f"✔️ Obiettivo raggiunto in {step+1} passi")
                    goal_step = step + 1
                # else:
                #     print(f"❌ Horizon raggiunto senza arrivare al goal")
                break

        final_dist = np.linalg.norm(env.goal - traj[-1])
        return np.array(traj), dists, goal_step, final_dist

    def policy_dist_safe(env, tol=1e-6):
        acts = candidate_actions()
        cur_d = np.linalg.norm(env.goal - env.state)
        a = best_improving_action(env, acts, cur_d, tol=tol)
        return a if a is not None else np.zeros(2, dtype=np.float32)  # resta fermo


    def to_std(x: torch.Tensor) -> torch.Tensor:
        return (x - env_dpo._mean) / env_dpo._std

    # ------------------------------------------------------------------
    # 1. ambiente + DPO-policy caricata
    # ------------------------------------------------------------------
    hidden_dim = 256
    num_layers = 4
    dropout_prob = 0.05

    start = np.array([0.05, 0.05])
    maze  = Maze(10,10,0,0)
    env_dpo = MazeEnv(sz=10, maze=maze, start=start, goal=np.array([0.95,0.95]),
                    reward="distance", dt=0.15, horizon=horizon, slide=1,
                    use_dpo=True, dpo_model_path=dpo_path, hidden_dim=hidden_dim, num_layers=num_layers, dropout_prob=dropout_prob)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    #  --- costanti usate nel training ---
    norm = np.load("norm_stats.npz")
    MEAN = norm["mean"].astype(np.float32)     # shape (2,)
    STD  = norm["std"].astype(np.float32) + 1e-8   # per evitare div/0

    def get_norm_tensors(device):
        if not hasattr(get_norm_tensors, "_cache"):
            get_norm_tensors._cache = {}
        if device not in get_norm_tensors._cache:
            get_norm_tensors._cache[device] = (
                torch.tensor(MEAN, device=device),   # <<== torch sul device giusto
                torch.tensor(STD,  device=device)
            )
        return get_norm_tensors._cache[device]

    # ---- normalizzazione per l'inferenza ----
    def _std(x: torch.Tensor) -> torch.Tensor:
        mean_t, std_t = get_norm_tensors(x.device)   # mai numpy qui
        return (x - mean_t) / std_t

    def is_legal(env, s_from, s_to, n=4):
        """Vero se il segmento s_from→s_to non tocca muri."""
        alphas = np.linspace(0., 1., n)
        for a in alphas:
            p = (1-a)*s_from + a*s_to
            if env.collision(p):          # oppure env.point_collision(*p)
                return False
        return True

    # ------------------------------------------------------------------
    # 2. definisci le due politiche
    # ------------------------------------------------------------------
    angles = np.linspace(-1, 1, 32, endpoint=False, dtype=np.float32)
    speeds = [0.05, 0.1, 0.2, 0.4, 0.8]                    

    def candidate_actions():
        a, s = np.meshgrid(angles, speeds)
        return np.stack([s.ravel(), a.ravel()], axis=1)


    def best_action_two_steps(env, beam_speeds=None):
        """
        Primo passo scelto guardando a due step di profondità.
        Se nessuna coppia (s1, s2) è percorribile, cade su
        “miglior primo passo fra quelli leciti”.
        """
        acts = candidate_actions()                         # (N, 2)

        # ---------- pre‑calcola i possibili s1 ----------
        next_states_1 = env.state + np.stack(
            [acts[:, 0] * np.cos(acts[:, 1] * np.pi) * env.dt,
            acts[:, 0] * np.sin(acts[:, 1] * np.pi) * env.dt],
            axis=1
        )

        best_two_score   = -np.inf   # miglior VALORE con 2 passi leciti
        best_two_action  = None      # → azione da restituire se esiste
        best_one_score   = -np.inf   # miglior VALORE con solo il 1° passo
        best_one_action  = None

        for i, s1 in enumerate(next_states_1):
            # 1° passo deve essere lecito
            if not is_legal(env, env.state, s1):
                continue

            # -------------------------------------------------------
            # valuta *tutti* i secondi passi a partire da s1
            # -------------------------------------------------------
            s2_all = s1 + np.stack(
                [acts[:, 0] * np.cos(acts[:, 1] * np.pi) * env.dt,
                acts[:, 0] * np.sin(acts[:, 1] * np.pi) * env.dt],
                axis=1
            )

            with torch.no_grad():
                scores = env.policy_net(
                    _std(torch.tensor(s2_all, dtype=torch.float32,
                                    device=env.device))
                ).cpu().numpy()              # shape (N,)

            # valore migliore raggiungibile in questo ramo
            idx_best2   = scores.argmax()
            two_score   = float(scores[idx_best2])
            s2_best     = s2_all[idx_best2]

            # --- (a) aggiorna il record 2‑passi SOLO se s2 è lecito ---
            if is_legal(env, s1, s2_best) and two_score > best_two_score:
                best_two_score  = two_score
                best_two_action = acts[i]

            # --- (b) aggiorna SEMPRE il record 1‑passo ---------------
            if two_score > best_one_score:
                best_one_score  = two_score
                best_one_action = acts[i]

        # -------------------------------------------------------------
        # restituisci:
        #   • l’azione che apre la MIGLIOR coppia lecita, se esiste
        #   • altrimenti la MIGLIORE azione di primo passo lecita
        # -------------------------------------------------------------
        if best_two_action is not None:
            return best_two_action
        if best_one_action is not None:
            return best_one_action

        # fallback (tutti i primi passi erano illegali – caso rarissimo)
        # print(f"‑‑ step {env.counter:3d} → action {best_two_action or best_one_action}")
        return np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. esegui i due roll-out
    # ------------------------------------------------------------------
    # traj_dpo, dist_dpo = rollout(env_dpo, policy_dpo_safe, max_steps=horizon)

    traj_dpo, dist_dpo, iter_dpo, dist_end_dpo = rollout(env_dpo, best_action_two_steps, max_steps=horizon)

    # traj_dpo, dist_dpo = rollout(env_dpo, policy_dpo, max_steps=horizon)

    #  per il baseline usiamo **una nuova copia** dell’ambiente (stesso maze!)
    env_base = MazeEnv(sz=10, maze=maze, start=start, goal=np.array([0.95,0.95]),
                    reward="distance", dt=0.15, horizon=horizon_base, slide=1, use_dpo=False)

    traj_base, dist_base, iter_base, dist_end_base = rollout(env_base, policy_dist_safe, max_steps=horizon_base)


    return iter_dpo, iter_base, dist_end_dpo, dist_end_base
