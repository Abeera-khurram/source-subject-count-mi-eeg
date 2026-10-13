import sys, os, argparse, torch, pandas as pd
R = os.environ.get('MLEMSDA_ROOT', '/content/drive/MyDrive/MLEMSDA_BCI2a')
sys.path.insert(0, f"{R}/src")
from experiment import load_config, load_subsets, run_one, make_run_id

ap = argparse.ArgumentParser()
ap.add_argument('--ks', type=int, nargs='+', default=[8, 1, 2, 3, 4, 5, 6, 7])
ap.add_argument('--max_subsets', type=int, default=10)   # SAME number for every k (k=8 has only 1)
ap.add_argument('--targets', nargs='+', default=[f"A0{i}" for i in range(1, 10)])
ap.add_argument('--results', default=f"{R}/results/raw/results_raw.csv")
ap.add_argument('--config', default=f"{R}/configs/experiment_config.yaml")
a = ap.parse_args()
cfg = load_config(a.config)
df = load_subsets(f"{R}/configs/source_subsets.csv")
df = df[df.k.isin(a.ks) & df.target.isin(a.targets) & (df['rank'] < a.max_subsets)]
# order: subset rank first, so if time runs out every k has the same number of subsets done
df = df.sort_values(['rank', 'k', 'target']).reset_index(drop=True)
done = set()
if os.path.exists(a.results):
    old = pd.read_csv(a.results)
    done = set(old.loc[old.status == 'completed', 'run_id'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
todo = [r for _, r in df.iterrows() if make_run_id(r) not in done]
print(f"Total planned: {len(df)} | already completed: {len(df) - len(todo)} | to run now: {len(todo)}", flush=True)
for i, r in enumerate(todo):
    print(f"\n=== [{i + 1}/{len(todo)}] ===", flush=True)
    run_one(r, cfg, R, device, a.results)
