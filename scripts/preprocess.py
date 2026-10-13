import os, sys, json
import numpy as np, pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, resample_poly

# ---------------- Settings (all recorded in the output metadata) ----------------
PREPROC_VERSION = "v1"
CHANNEL_IDX   = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13]   # 0-based columns of the 22 EEG channels
CHANNEL_NAMES = ["FC3","FC1","FCz","FC2","FC4","C3","C1","Cz","C2","C4","CP3"]
KEEP_CLASSES  = {1: 0, 2: 1}     # left hand -> 0, right hand -> 1
BAND_HZ       = (8.0, 30.0)
FILTER_ORDER  = 4
CUE_OFFSET_S  = 2.0              # window start, seconds after trial start
WINDOW_S      = 4.0              # window length in seconds
TARGET_FS     = 500
N_SAMPLES_OUT = 2000             # model requirement
SUBJECTS = [f"A0{i}" for i in range(1, 10)]

DATA_DIR, OUT_DIR, REPORT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT_DIR, exist_ok=True)

def process_file(path):
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    Xs, ys, arts = [], [], []
    n_all = 0
    for run in m["data"]:
        trial = np.atleast_1d(run.trial)
        if trial.size == 0:
            continue                                   # runs without MI trials
        fs = int(run.fs)
        sig = np.asarray(run.X, dtype=np.float64)[:, CHANNEL_IDX]
        sos = butter(FILTER_ORDER, BAND_HZ, btype="bandpass", fs=fs, output="sos")
        sig = sosfiltfilt(sos, sig, axis=0)            # filter the continuous run, then cut trials
        y = np.atleast_1d(run.y).astype(int)
        art = np.atleast_1d(run.artifacts).astype(int)
        w = int(round(WINDOW_S * fs))
        for t0, lab, a in zip(trial, y, art):
            n_all += 1
            if lab not in KEEP_CLASSES:
                continue
            s = int(t0) + int(round(CUE_OFFSET_S * fs))
            seg = sig[s:s + w]
            if seg.shape[0] != w:
                raise ValueError(f"{path}: window outside the recording")
            seg = resample_poly(seg, TARGET_FS, fs, axis=0)      # (2000, 11)
            assert seg.shape == (N_SAMPLES_OUT, len(CHANNEL_IDX)), seg.shape
            Xs.append(seg.T.astype(np.float32))                  # (11, 2000)
            ys.append(KEEP_CLASSES[int(lab)])
            arts.append(int(a != 0))
    return np.stack(Xs), np.array(ys, dtype=np.int64), np.array(arts, dtype=np.int64), n_all

rows = []
for s in SUBJECTS:
    Xl, yl, al, sl = [], [], [], []
    n_all_total = 0
    for sess in ["T", "E"]:
        X, y, a, n_all = process_file(os.path.join(DATA_DIR, f"{s}{sess}.mat"))
        n_all_total += n_all
        Xl.append(X); yl.append(y); al.append(a); sl += [sess] * len(y)
    X = np.concatenate(Xl); y = np.concatenate(yl); a = np.concatenate(al)
    np.savez_compressed(os.path.join(OUT_DIR, f"{s}.npz"), X=X, y=y, artifact=a, session=np.array(sl))
    rows.append({"Subject": s, "Trials in files (4 classes)": n_all_total,
                 "Trials kept (left+right)": len(y), "Left (0)": int((y == 0).sum()),
                 "Right (1)": int((y == 1).sum()), "Artifact-flagged among kept": int(a.sum()),
                 "Shape": str(X.shape[1:]), "Removed trials": 0})
    print(f"{s}: kept {len(y)} trials, shape {X.shape}, artifacts flagged {int(a.sum())}")

pd.DataFrame(rows).to_csv(REPORT, index=False)
meta = dict(preprocessing_version=PREPROC_VERSION, channel_idx=CHANNEL_IDX, channel_names=CHANNEL_NAMES,
            classes="1=left->0, 2=right->1", band_hz=BAND_HZ, filter="butterworth bandpass, zero-phase (sosfiltfilt)",
            filter_order=FILTER_ORDER, window_start_s=CUE_OFFSET_S, window_length_s=WINDOW_S,
            resample_to_hz=TARGET_FS, samples_per_trial=N_SAMPLES_OUT, artifact_trials="kept, flagged in file",
            normalization="not applied here (computed from source subjects at training time)")
json.dump(meta, open(os.path.join(OUT_DIR, "preprocessing_meta.json"), "w"), indent=2)
print("Saved report:", REPORT)
