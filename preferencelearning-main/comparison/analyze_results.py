#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_final_plots.py
Versione 'Blindata' per la tesi.
1. Forza backend 'Agg' (risolve file vuoti/bianchi).
2. Usa Imputazione a 240 steps per i timeout (risolve grafici vuoti).
3. Genera le 3 figure chiave.
"""
import matplotlib
matplotlib.use('Agg') # <--- FONDAMENTALE: Impedisce errori di display

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configura i percorsi
RESULTS_DIR = "./thesis_gold_run"
OUTPUT_DIR = "./thesis_figures_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stile grafico accademico
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

def load_data(root_dir):
    print(f"--> Cerco dati in: {os.path.abspath(root_dir)}")
    pattern = os.path.join(root_dir, "seed=*", "*", "report.json")
    files = glob.glob(pattern)
    
    if not files:
        print("❌ ERRORE: Nessun file report.json trovato!")
        return pd.DataFrame()

    print(f"✅ Trovati {len(files)} file report. Elaborazione...")
    
    data = []
    for f in files:
        try:
            with open(f, 'r') as fp: rep = json.load(fp)
            tid = rep['test_id']
            group = tid.split('_')[0]
            steps = rep['solve']['dpo_steps_to_goal']
            horizon = 240
            
            # LOGICA DI IMPUTAZIONE:
            # Se steps = -1 o >= 240 -> FALLIMENTO
            if steps == -1 or steps >= horizon:
                success = 0
                steps_eff = np.nan    # Efficienza: Ignora (NaN)
                steps_steer = 240.0   # Steering: Usa 240 (Timeout visibile)
            else:
                success = 1
                steps_eff = float(steps)
                steps_steer = float(steps)
                
            data.append({
                'seed': rep.get('seed', 0),
                'test_id': tid,
                'group': group[0], 
                'steps_eff': steps_eff,
                'steps_steer': steps_steer,
                'success': success
            })
        except Exception as e: 
            pass
            
    df = pd.DataFrame(data)
    print(f"✅ DataFrame caricato: {len(df)} righe totali.")
    return df

def clean_labels(ax):
    """Pulisce le etichette dell'asse X"""
    labels = [item.get_text() for item in ax.get_xticklabels()]
    new_labels = []
    for l in labels:
        parts = l.split('_')
        if len(parts) > 1: 
            txt = " ".join(parts[1:]).title()
            txt = txt.replace("Corridor Heavy", "Corridor").replace("Balanced Degree", "Balanced")
            txt = txt.replace("Efficient Safety", "Hybrid").replace("Goal Strong", "Greedy")
            txt = txt.replace("Wide Space", "Safe").replace("Path Strong", "Imitation")
            new_labels.append(txt)
        else: new_labels.append(l)
    ax.set_xticklabels(new_labels, rotation=30, ha='right')

def plot_rq1_efficiency(df):
    """Fig 1: Efficienza (solo successi)"""
    print("\n--- Generazione Fig 1 (Efficiency) ---")
    subset = df[df['group'] == 'A'].dropna(subset=['steps_eff'])
    
    if len(subset) == 0:
        print("⚠️ Nessun successo nel Gruppo A.")
        return

    plt.figure(figsize=(10, 6))
    order = sorted(subset['test_id'].unique())
    
    # Barre
    sns.barplot(data=subset, x='test_id', y='steps_eff', order=order, 
                palette="Blues_d", errorbar=None, alpha=0.6)
    # Punti singoli
    sns.stripplot(data=subset, x='test_id', y='steps_eff', order=order,
                  color='black', size=5, jitter=0.15)
    
    plt.axhline(29, color='red', ls='--', lw=2, label='A* Baseline')
    plt.ylabel("Steps (Success Only)")
    plt.xlabel("")
    clean_labels(plt.gca())
    plt.title("RQ1: Efficiency (Group A)")
    plt.legend()
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Fig1_Efficiency.png")
    plt.savefig(outfile)
    print(f"Salvato: {outfile}")

def plot_rq2_steering(df):
    """Fig 2: Steering (include timeout a 240)"""
    print("\n--- Generazione Fig 2 (Steering) ---")
    targets = ["B2_goal_strong", "B3.1_efficient_safety", "B3_wide_space", "B4_path_strong"]
    targets = [t for t in targets if t in df['test_id'].unique()]
    
    if not targets:
        print("⚠️ Dati Gruppo B mancanti.")
        return

    subset = df[df['test_id'].isin(targets)].copy()
    
    plt.figure(figsize=(9, 6))
    
    # Linea di trend
    sns.pointplot(data=subset, x='test_id', y='steps_steer', order=targets,
                  color='firebrick', capsize=0.1, linestyles='-', errorbar=None)
    
    # Punti singoli (con jitter per vedere sovrapposizioni a 240)
    sns.stripplot(data=subset, x='test_id', y='steps_steer', order=targets,
                  color='black', alpha=0.5, size=6, jitter=0.1)
    
    plt.ylabel("Steps (Timeout = 240)")
    plt.xlabel("Intent")
    clean_labels(plt.gca())
    
    plt.axhline(29, color='gray', ls=':', label='Greedy Limit')
    plt.axhline(240, color='red', ls='--', alpha=0.5, label='Timeout')
    
    plt.title("RQ2: Steering Trade-off")
    plt.legend(loc='center left')
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "Fig2_Steering.png")
    plt.savefig(outfile)
    print(f"Salvato: {outfile}")

def plot_rq3_robustness(df):
    """Fig 3: Success Rate"""
    print("\n--- Generazione Fig 3 (Robustness) ---")
    targets = ["A1_random", "H1_quantized_score", "F1_noise_10pct", "F2_noise_25pct", "E3_cutoff_80"]
    targets = [t for t in targets if t in df['test_id'].unique()]
    
    subset = df[df['test_id'].isin(targets)].copy()
    agg = subset.groupby('test_id', observed=True)['success'].mean().reset_index()
    agg = agg.sort_values('test_id')
    
    plt.figure(figsize=(10, 5))
    colors = ["#2ecc71" if x > 0.8 else "#c0392b" for x in agg['success']]
    ax = sns.barplot(data=agg, x='test_id', y='success', palette=colors)
    
    plt.ylim(0, 1.1)
    plt.ylabel("Success Rate")
    plt.xlabel("")
    clean_labels(plt.gca())
    plt.title("RQ3: Robustness")
    
    for p, val in zip(ax.patches, agg['success']):
        h = p.get_height()
        if not np.isnan(h):
            ax.text(p.get_x()+p.get_width()/2., h+0.02, f"{val*100:.0f}%", ha='center', weight='bold')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "Fig3_Robustness.png")
    plt.savefig(outfile)
    print(f"Salvato: {outfile}")

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Cartella {RESULTS_DIR} non trovata.")
        return
    
    df = load_data(RESULTS_DIR)
    if df.empty:
        print("❌ Nessun dato caricato. Controlla i percorsi.")
        return

    plot_rq1_efficiency(df)
    plot_rq2_steering(df)
    plot_rq3_robustness(df)
    print(f"\n✅ FINITO. I grafici sono in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()