import sys, os, argparse, torch, pandas as pd
R = os.environ.get('MLEMSDA_ROOT', '/content/drive/MyDrive/MLEMSDA_BCI2a')
sys.path.insert(0, f"{R}/src")
from experiment import load_config, load_subsets
from experiment_deepconv_only import run_one_deepconv

ap = argparse.ArgumentParser()
ap.add_argument('--subsets_file', default=f"{R}/configs/final_subsets.csv")
ap.add_argument('--targets', nargs='+', default=[f"A0{i}" for i in range(1, 10)])
ap.add_argument('--results', default=f"{R}/results/raw/deepconv_only_ks.csv")
ap.add_argument('--config', default=f"{R}/configs/experiment_config.yaml")
a = ap.parse_args()

cfg = load_config(a.config)
df = load_subsets(a.subsets_file)
df = df[df.target.isin(a.targets)]
df = df.sort_values(['k', 'subset_id', 'target']).reset_index(drop=True)

done = set()
if os.path.exists(a.results):
    old = pd.read_csv(a.results)
    done = set(old.loc[old.status == 'completed', 'run_id'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
todo = [r for _, r in df.iterrows() if f"{r['target']}_k{int(r['k'])}_s{r['subset_id']}" not in done]
print(f"Total planned: {len(df)} | already completed: {len(df) - len(todo)} | to run now: {len(todo)}", flush=True)
for i, r in enumerate(todo):
    print(f"\n=== [{i + 1}/{len(todo)}] ===", flush=True)
    run_one_deepconv(r, cfg, R, device, a.results)
