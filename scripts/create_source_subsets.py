import sys, math
import numpy as np, pandas as pd

OUT = sys.argv[1]
N_SUBSETS = 10
SEEDS = [1000 + i for i in range(1, N_SUBSETS + 1)]   # 1001 ... 1010
SUBJECTS = [f"A0{i}" for i in range(1, 10)]

rows = []
for t_idx, target in enumerate(SUBJECTS, start=1):
    pool = [s for s in SUBJECTS if s != target]
    for k in range(1, 9):
        if k == 8:
            rows.append({"target_subject": target, "k": 8, "subset_id": 1,
                         "source_selection_seed": None,
                         "source_subjects": " ".join(pool), "is_repeat": False})
            continue
        seen = set()
        n_possible = math.comb(len(pool), k)
        for sid, seed in enumerate(SEEDS, start=1):
            rng = np.random.RandomState([seed, t_idx, k])
            while True:
                pick = tuple(sorted(str(x) for x in rng.choice(pool, size=k, replace=False)))
                if pick not in seen or len(seen) >= n_possible:
                    break
            rows.append({"target_subject": target, "k": k, "subset_id": sid,
                         "source_selection_seed": seed,
                         "source_subjects": " ".join(pick), "is_repeat": pick in seen})
            seen.add(pick)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("Total configurations:", len(df), "(expected 639)")
print("Rows per k:", df.groupby("k").size().to_dict())
print("Repeated subsets per k:", df[df.is_repeat].groupby("k").size().to_dict())
print("Saved:", OUT)
