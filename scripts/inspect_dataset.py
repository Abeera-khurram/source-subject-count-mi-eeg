import sys, os
import numpy as np, pandas as pd, scipy.io as sio

DATA_DIR, OUT = sys.argv[1], sys.argv[2]
rows = []
for s in range(1, 10):
    for sess in ["T", "E"]:
        fname = f"A0{s}{sess}.mat"
        m = sio.loadmat(os.path.join(DATA_DIR, fname), struct_as_record=False, squeeze_me=True)
        n_trials, mi_runs, art_trials = 0, 0, 0
        chans, fs, labels = set(), set(), set()
        for run in m["data"]:
            chans.add(int(run.X.shape[1]))
            fs.add(int(run.fs))
            tr = np.atleast_1d(run.trial)
            y = np.atleast_1d(run.y)
            art = np.atleast_1d(run.artifacts)
            if tr.size > 0:
                mi_runs += 1
                n_trials += int(tr.size)
                labels.update(y.tolist())
                art_trials += int(np.sum(art != 0))
        rows.append({"Subject": f"A0{s}", "Session": sess,
                     "Runs with trials": mi_runs, "Trials": n_trials,
                     "Artifact-flagged trials": art_trials,
                     "Channels": sorted(chans), "Sampling rate (Hz)": sorted(fs),
                     "Classes": sorted(labels)})
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.to_string(index=False))
print("Saved:", OUT)
