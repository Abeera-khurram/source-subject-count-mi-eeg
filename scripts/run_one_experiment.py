import sys, os, argparse, torch
R = os.environ.get('MLEMSDA_ROOT', '/content/drive/MyDrive/MLEMSDA_BCI2a')
sys.path.insert(0, f"{R}/src")
from experiment import load_config, load_subsets, run_one

ap = argparse.ArgumentParser()
ap.add_argument('--target', required=True)          # e.g. A01
ap.add_argument('--k', type=int, required=True)
ap.add_argument('--rank', type=int, default=0)      # which subset of this (target,k): 0 = first
ap.add_argument('--results', default=f"{R}/results/raw/results_raw.csv")
ap.add_argument('--config', default=f"{R}/configs/experiment_config.yaml")
a = ap.parse_args()
cfg = load_config(a.config)
df = load_subsets(f"{R}/configs/source_subsets.csv")
row = df[(df.target == a.target) & (df.k == a.k) & (df['rank'] == a.rank)].iloc[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
run_one(row, cfg, R, device, a.results)
