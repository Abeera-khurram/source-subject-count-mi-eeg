import os, re, sys, time, json, platform, datetime, random, traceback
import numpy as np, pandas as pd, torch, yaml, higher
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, cohen_kappa_score, precision_recall_fscore_support,
                             confusion_matrix)
from backbone import deep
from data_utils import load_domain, norm_subject

COLUMNS = ['run_id', 'target_subject', 'k', 'subset_id', 'source_subjects', 'source_selection_seed',
           'training_seed', 'preprocessing_version', 'model_version', 'learning_rate', 'optimizer',
           'batch_size', 'epochs', 'n_source_subjects', 'n_source_trials', 'n_target_trials',
           'accuracy', 'kappa', 'macro_f1', 'precision_macro', 'recall_macro', 'training_time_seconds',
           'checkpoint_path', 'log_path', 'status', 'error_message',
           'datetime', 'inference_time_seconds', 'pretrain_epochs_run', 'per_class_precision',
           'per_class_recall', 'per_class_f1', 'confusion_matrix', 'meta_params', 'python', 'torch',
           'cuda', 'gpu', 'os']


def load_config(path):
    return yaml.safe_load(open(path))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------- Stage 1: DeepConvNet pretraining (source subjects only) ----------------
def pretrain(X, y, cfg, device, seed, log):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    n_val = max(int(round(cfg['pre_val_frac'] * len(y))), 2)
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xva, yva = X[ti].to(device), y[ti].to(device), X[vi].to(device), y[vi].to(device)
    model = deep().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['pre_lr'], weight_decay=cfg['pre_wd'])
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg['pre_bs'], shuffle=True)
    best, best_state, wait, ep_run = float('inf'), None, 0, 0
    for epoch in range(cfg['pre_max_epochs']):
        ep_run = epoch + 1
        model.train()
        for xb, yb in loader:
            out = F.log_softmax(model(xb), dim=1).view(xb.size(0), -1)
            loss = F.nll_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = F.log_softmax(model(Xva), dim=1).view(Xva.size(0), -1)
            vl = F.nll_loss(out, yva).item()
        if vl < best - cfg['pre_delta']:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % 10 == 0:
            log(f"[pretrain] epoch {epoch} val_loss {vl:.4f} best {best:.4f}")
        if wait >= cfg['pre_patience']:
            log(f"[pretrain] early stop at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model, ep_run


# ---------------- Stage 2: MLEMSDA meta-learning (as in released loocv_meta_deep.py) ----------------
def make_tasks(Xg, yg, cfg, seed):
    g = torch.Generator().manual_seed(seed)
    n, dev = len(yg), Xg.device
    tasks = []
    for _ in range(cfg['num_tasks']):
        perm = torch.randperm(n, generator=g).to(dev)
        q, s = perm[:cfg['query_size']], perm[cfg['query_size']:cfg['query_size'] + cfg['support_max']]
        tasks.append((Xg[s], yg[s], Xg[q], yg[q]))
    return tasks


def meta_train(model, X, y, cfg, device, seed, log):
    set_seed(seed + 1)
    Xg, yg = X.to(device), y.to(device)
    tasks = make_tasks(Xg, yg, cfg, seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['meta_lr'], weight_decay=cfg['meta_wd'],
                                momentum=cfg['meta_momentum'])
    inner_opt = torch.optim.SGD(model.parameters(), lr=cfg['inner_lr'], weight_decay=cfg['meta_wd'],
                                momentum=cfg['meta_momentum'])
    ft_loader = DataLoader(TensorDataset(Xg, yg), batch_size=cfg['ft_bs'], shuffle=True)
    best_loss, wait = float('inf'), 0
    for epoch in range(cfg['meta_epochs']):
        model.train()
        for m in model.modules():                      # FIX: BatchNorm ke running stats freeze karo meta-training ke dauran
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        qlosses = []
        for Xs, ys, Xq, yq in tasks:
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                for xb, yb in DataLoader(TensorDataset(Xs, ys), batch_size=cfg['support_bs'], shuffle=True):
                    logits = fnet(xb).view(xb.size(0), -1)   # FIXED: raw logits into cross_entropy (double-softmax removed)
                    diffopt.step(F.cross_entropy(logits, yb))
                logits_q = fnet(Xq).view(Xq.size(0), -1)
                qloss = F.cross_entropy(logits_q, yq)
                optimizer.zero_grad(); qloss.backward(); optimizer.step()
                qlosses.append(qloss.item())
        for xb, yb in ft_loader:                                      # fine-tuning on SOURCE data
            out = F.log_softmax(model(xb), dim=1).view(xb.size(0), -1)
            loss = F.nll_loss(out, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        cur_loss = np.mean(qlosses)
        if epoch % 10 == 0:
            log(f"[meta] epoch {epoch} query_loss {cur_loss:.4f}")
        if cur_loss < best_loss - cfg.get('meta_delta', 0.001):        # SPEEDUP: jab tak sudhar ho raha hai chalte raho
            best_loss, wait = cur_loss, 0
        else:
            wait += 1
        if wait >= cfg.get('meta_patience', 20):                       # SPEEDUP: sudhar rukne par khud ruk jao
            log(f"[meta] early stop at epoch {epoch} (no improvement {cfg.get('meta_patience', 20)} epochs)")
            break
    return model


@torch.no_grad()
def predict(model, X, device, bs=64):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        out = F.softmax(model(xb), dim=1).view(xb.size(0), -1)
        preds.append(out.argmax(1).cpu())
    return torch.cat(preds).numpy()


def compute_metrics(yt, yp):
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    return dict(accuracy=accuracy_score(yt, yp), kappa=cohen_kappa_score(yt, yp), macro_f1=float(f.mean()),
                precision_macro=float(p.mean()), recall_macro=float(r.mean()),
                per_class_precision=json.dumps(p.round(4).tolist()), per_class_recall=json.dumps(r.round(4).tolist()),
                per_class_f1=json.dumps(f.round(4).tolist()),
                confusion_matrix=json.dumps(confusion_matrix(yt, yp, labels=[0, 1]).tolist()))


# ---------------- subsets file (configs/source_subsets.csv) ----------------
def _col(df, exact, contains=()):
    low = {c.lower(): c for c in df.columns}
    for e in exact:
        if e in low: return low[e]
    for cn in contains:
        for lc, c in low.items():
            if cn in lc: return c
    raise KeyError(f"column not found ({exact}) in {list(df.columns)}")


def load_subsets(path):
    df = pd.read_csv(path)
    c = dict(target=_col(df, ('target', 'target_subject'), ('target',)),
             k=_col(df, ('k', 'n_sources', 'num_sources'), ()),
             subset=_col(df, ('subset_id', 'subset'), ('subset',)),
             sources=_col(df, ('source_subjects', 'sources'), ('subjects', 'source')),
             seed=_col(df, ('source_selection_seed', 'seed'), ('seed',)))
    out = pd.DataFrame({'target': df[c['target']].map(norm_subject), 'k': df[c['k']].astype(int),
                        'subset_id': df[c['subset']],
                        'sources': df[c['sources']].map(lambda v: [f"A{int(n):02d}" for n in re.findall(r'\d+', str(v))]),
                        'sel_seed': df[c['seed']]})
    out['rank'] = out.groupby(['target', 'k']).cumcount()
    return out


def make_run_id(r):
    return f"{r['target']}_k{int(r['k'])}_s{r['subset_id']}"


# ---------------- one run ----------------
def run_one(row, cfg, R, device, results_csv, processed_dir=None):
    processed_dir = processed_dir or f"{R}/data/processed"
    run_id = make_run_id(row)
    os.makedirs(f"{R}/logs", exist_ok=True); os.makedirs(f"{R}/checkpoints", exist_ok=True)
    log_path, ckpt = f"{R}/logs/{run_id}.log", f"{R}/checkpoints/{run_id}.pt"
    lf = open(log_path, 'w')

    def log(msg):
        line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); lf.write(line + '\n'); lf.flush()

    rec = dict(run_id=run_id, target_subject=row['target'], k=int(row['k']), subset_id=row['subset_id'],
               source_subjects=' '.join(row['sources']), source_selection_seed=row['sel_seed'],
               training_seed=cfg['training_seed'], preprocessing_version=cfg['preprocessing_version'],
               model_version=cfg['model_version'],
               learning_rate=f"pretrain {cfg['pre_lr']} / meta {cfg['meta_lr']} inner {cfg['inner_lr']}",
               optimizer='AdamW (pretrain) + SGD (meta)',
               batch_size=f"{cfg['pre_bs']} / {cfg['support_bs']} / {cfg['ft_bs']}",
               epochs=f"pretrain<= {cfg['pre_max_epochs']} (early stop) / meta {cfg['meta_epochs']}",
               n_source_subjects=len(row['sources']), log_path=log_path, checkpoint_path=ckpt,
               meta_params=json.dumps({k: cfg[k] for k in ('num_tasks', 'query_size', 'support_max', 'support_bs', 'ft_bs')}),
               datetime=datetime.datetime.now().isoformat(timespec='seconds'), python=platform.python_version(),
               torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', os=platform.platform())
    model = None
    try:
        assert row['target'] not in row['sources'], "LEAKAGE: target in source list"
        Xs, ys = load_domain(processed_dir, row['sources'])
        Xt, yt = load_domain(processed_dir, [row['target']])
        rec.update(n_source_trials=len(ys), n_target_trials=len(yt))
        log(f"RUN {run_id} sources={row['sources']} n_src={len(ys)} n_tgt={len(yt)} device={device}")
        t0 = time.time()
        model, ep_run = pretrain(Xs, ys, cfg, device, cfg['training_seed'], log)
        model = meta_train(model, Xs, ys, cfg, device, cfg['training_seed'], log)
        rec['training_time_seconds'] = round(time.time() - t0, 1)
        rec['pretrain_epochs_run'] = ep_run
        t1 = time.time()
        pred = predict(model, Xt, device)                 # target used ONCE, only here
        rec['inference_time_seconds'] = round(time.time() - t1, 3)
        rec.update(compute_metrics(yt.numpy(), pred))
        torch.save(model.state_dict(), ckpt)
        rec.update(status='completed', error_message='')
        log(f"DONE acc={rec['accuracy']:.4f} kappa={rec['kappa']:.4f} time={rec['training_time_seconds']}s")
    except Exception as e:
        log(traceback.format_exc())
        rec.update(status='failed', error_message=repr(e)[:500])
    finally:
        lf.close()
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    pd.DataFrame([rec]).reindex(columns=COLUMNS).to_csv(results_csv, mode='a', index=False,
                                                        header=not os.path.exists(results_csv))
    return rec
