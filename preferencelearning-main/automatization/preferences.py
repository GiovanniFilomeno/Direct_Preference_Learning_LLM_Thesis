from tqdm import tqdm
import pandas as pd
import numpy as np
import random

def generate_preferences(total_couples=200_000, weights_vec=[0.0, 0.0, 0.0, 0.0, 0.0], hard_ratio = 0.7):
    df = pd.read_parquet("positions.parquet")

    # --------------------------------------------------------
    # 3) pesi
    # --------------------------------------------------------
    w_path, w_wall, w_goal, w_deg, w_gate = weights_vec[0], weights_vec[1], weights_vec[2], weights_vec[3], weights_vec[4]

    # --------------------------------------------------------
    # 4) score (minore è migliore)
    # --------------------------------------------------------
    df["total_score"] = (
        w_path * df.path_distance        #  + penalità: cammino lungo
        + w_goal * df.distance_to_goal        #  + penalità: lontano dal goal
        - w_deg  * df.degree         #  + penalità: vicolo cieco
        - w_wall * df.distance_from_wall        #  – bonus: punti “larghi”, lontani dai muri
        + w_gate * df.distance_to_gate
    )

    def sample_pairs(
            df: pd.DataFrame,
            num_pairs: int        = 200_000,
            hard_ratio: float     = 0.7,
            thresh: float         = 0.05,
            min_delta_deg: float  = 0.001,     # <- NOVITÀ: differenza minima di degree
            max_trials: int       = 50_000_000   # per evitare loop infiniti
        ):

        idx       = df.index.to_numpy()
        scores    = df["total_score"].to_numpy()
        degrees   = df["degree"].to_numpy()          # cache per velocità
        pairs     = set()
        n_hard    = int(num_pairs * hard_ratio)

        pbar = tqdm(total=num_pairs, desc="sampling pairs")
        trials = 0
        while len(pairs) < num_pairs and trials < max_trials:
            trials += 1

            i, j = np.random.choice(idx, 2, replace=False)
            si, sj = scores[i], scores[j]

            if si == sj:          # identico score ⇒ salta subito
                continue

            better, worse = (i, j) if si < sj else (j, i)   # score minore = migliore
            delta_score   = abs(si - sj)
            want_hard     = len(pairs) < n_hard
            if  (want_hard  and delta_score >= thresh) or \
                (not want_hard and delta_score <  thresh):
                continue        # scorretto per la fascia che stiamo riempiendo

            # -----------------------------------------  
            # 2) vincolo sullo scarto di degree
            # -----------------------------------------
            if abs(degrees[better] - degrees[worse]) < min_delta_deg:
                continue        # differenza troppo piccola → poco informativa

            # -----------------------------------------  
            # 3) aggiungi la coppia
            # -----------------------------------------
            pair = (df.at[better,"x"], df.at[better,"y"],
                    df.at[worse, "x"], df.at[worse, "y"], 1)

            if pair not in pairs:
                pairs.add(pair)
                pbar.update(1)

        pbar.close()

        if len(pairs) < num_pairs:
            print(f"⚠️  solo {len(pairs)} coppie generate dopo {trials} tentativi "
                f"(probabilmente min_delta_deg è troppo alto).")

        return list(pairs)

    def sample_same_row_pairs(df, n_pairs, min_dx=0.25):
        pairs = []
        grouped = df.groupby(pd.cut(df.y, bins=np.arange(0,1.01,0.1)))
        while len(pairs) < n_pairs:
            _, g = random.choice(list(grouped))
            if len(g) < 2: continue
            a, b = g.sample(2).itertuples()
            if abs(a.x - b.x) < min_dx: continue
            better, worse = (a,b) if a.total_score < b.total_score else (b,a)
            pairs.append((better.x,better.y,worse.x,worse.y,1))
        return pairs


    n_total   = total_couples
    pairs_row = sample_same_row_pairs(df, int(n_total*0.4))
    pairs_mix = sample_pairs(df, n_total - len(pairs_row), hard_ratio=hard_ratio)
    pairs = pairs_row + pairs_mix

    pd.DataFrame(pairs,
        columns=["x_better","y_better","x_worse","y_worse","preference"]
    ).to_parquet("preferences.parquet", index=False)