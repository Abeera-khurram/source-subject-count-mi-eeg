"""Load preprocessed BCI IV-2a subjects and build the 2-class (left vs right hand) tensors."""
import os, re, glob
import numpy as np
import torch

SUBJECTS = [f"A0{i}" for i in range(1, 10)]
_CACHE = {}
_NUM = re.compile(r"\d+")


def norm_subject(v):
    n = int(_NUM.findall(str(v))[0])
    return "A%02d" % n


def _read_arrays(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    if ext in ('.h5', '.hdf5'):
        import h5py
        out = {}
        with h5py.File(path, 'r') as h:
            h.visititems(lambda n, o: out.__setitem__(n, o[()]) if hasattr(o, 'shape') else None)
        return out
    raise ValueError(f"Unsupported file type: {path}")


def load_subject(processed_dir, subj):
    subj = norm_subject(subj)
    if subj in _CACHE:
        return _CACHE[subj]
    files = [f for f in glob.glob(os.path.join(processed_dir, '*'))
             if re.search(subj, os.path.basename(f), re.I)
             and f.lower().endswith(('.npz', '.h5', '.hdf5'))]
    if len(files) != 1:
        raise FileNotFoundError(f"{subj}: expected exactly 1 processed file in {processed_dir}, found {files}. "
                                f"Directory has: {sorted(os.listdir(processed_dir))}")
    arrays = _read_arrays(files[0])
    X = [v for v in arrays.values() if getattr(v, 'ndim', 0) == 3]
    if len(X) != 1:
        raise ValueError(f"{files[0]}: need exactly one 3-D array, keys/shapes: "
                         f"{ {k: getattr(v, 'shape', None) for k, v in arrays.items()} }")
    X = X[0]
    ykey = [k for k in arrays if k.lower() in ('y', 'labels', 'label')]
    if len(ykey) != 1:
        raise ValueError(f"{files[0]}: cannot find label key, keys/shapes: "
                         f"{ {k: getattr(v, 'shape', None) for k, v in arrays.items()} }")
    y = arrays[ykey[0]]
    if getattr(y, 'ndim', 0) != 1 or len(y) != X.shape[0]:
        raise ValueError(f"{files[0]}: label vector shape {getattr(y, 'shape', None)} != ({X.shape[0]},)")
    if X.shape[1:] != (11, 2000):
        raise ValueError(f"{files[0]}: expected shape (n, 11, 2000), got {X.shape}")
    _CACHE[subj] = (X.astype(np.float32), np.asarray(y).astype(np.int64))
    return _CACHE[subj]


def to_two_class(X, y):
    """Keep the two smallest label values (left, right hand: 1,2 or 0,1) -> re-encode as 0 and 1."""
    classes = np.unique(y)
    if len(classes) not in (2, 4):
        raise ValueError(f"Unexpected label set: {classes}")
    keep = classes[:2]
    mask = np.isin(y, keep)
    return X[mask], (y[mask] == keep[1]).astype(np.int64), keep


def load_domain(processed_dir, subjects):
    """Pooled 2-class data as torch tensors: X (N,1,2000,11), y (N,)."""
    Xs, ys = [], []
    for s in subjects:
        X, y = load_subject(processed_dir, s)
        X2, y2, _ = to_two_class(X, y)
        Xs.append(X2); ys.append(y2)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    X = np.expand_dims(X.transpose(0, 2, 1), 1)
    return torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(y)
