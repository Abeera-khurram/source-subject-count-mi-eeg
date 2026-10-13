import os, time, json, platform, datetime, traceback
import torch
import pandas as pd
from backbone import deep
from experiment import meta_train, predict, compute_metrics
from data_utils import load_domain

COLUMNS = ['run_id', 'target_subject', 'k', 'subset_id', 'source_subjects', 'source_selection_seed',
           'training_seed', 'preprocessing_version', 'model_version', 'n_source_subjects',
           'n_source_trials', 'n_target_trials', 'accuracy', 'kappa', 'macro_f1',
           'precision_macro', 'recall_macro', 'meta_training_time_seconds', 'used_checkpoint',
           'checkpoint_path', 'log_path', 'status', 'error_message', 'datetime',
           'per_class_precision', 'per_class_recall', 'per_class_f1', 'confusion_matrix',
           'python', 'torch', 'cuda', 'gpu', 'os']


def run_one_meta(row, cfg, R, device, results_csv, processed_dir=None):
    processed_dir = processed_dir or f"{R}/data/processed"
    run_id = f"{row['target']}_k{int(row['k'])}_s{row['subset_id']}"
    deepconv_ckpt = f"{R}/checkpoints/{run_id}_deepconv.pt"
    os.makedirs(f"{R}/logs", exist_ok=True); os.makedirs(f"{R}/checkpoints", exist_ok=True)
    log_path = f"{R}/logs/{run_id}_meta.log"
    ckpt_out = f"{R}/checkpoints/{run_id}_meta.pt"
    lf = open(log_path, 'w')

    def log(msg):
        line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); lf.write(line + '\n'); lf.flush()

    rec = dict(run_id=run_id, target_subject=row['target'], k=int(row['k']), subset_id=row['subset_id'],
               source_subjects=' '.join(row['sources']), source_selection_seed=row['sel_seed'],
               training_seed=cfg['training_seed'], preprocessing_version=cfg['preprocessing_version'],
               model_version='MLEMSDA meta (from saved DeepConvNet checkpoint)',
               n_source_subjects=len(row['sources']), log_path=log_path, checkpoint_path=ckpt_out,
               datetime=datetime.datetime.now().isoformat(timespec='seconds'),
               python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
               os=platform.platform())
    model = None
    try:
        assert row['target'] not in row['sources'], "LEAKAGE: target in source list"
        used_ckpt = os.path.exists(deepconv_ckpt)
        rec['used_checkpoint'] = used_ckpt
        model = deep().to(device)
        if used_ckpt:
            model.load_state_dict(torch.load(deepconv_ckpt, map_location=device))
            log(f"Loaded pretrained checkpoint: {deepconv_ckpt}")
        else:
            log(f"WARNING: checkpoint not found ({deepconv_ckpt}), starting meta from scratch weights")

        Xs, ys = load_domain(processed_dir, row['sources'])
        Xt, yt = load_domain(processed_dir, [row['target']])
        rec.update(n_source_trials=len(ys), n_target_trials=len(yt))
        log(f"RUN {run_id} sources={row['sources']} n_src={len(ys)} n_tgt={len(yt)} device={device}")

        t0 = time.time()
        model = meta_train(model, Xs, ys, cfg, device, cfg['training_seed'], log)
        rec['meta_training_time_seconds'] = round(time.time() - t0, 1)

        pred = predict(model, Xt, device)
        rec.update(compute_metrics(yt.numpy(), pred))
        torch.save(model.state_dict(), ckpt_out)
        rec.update(status='completed', error_message='')
        log(f"DONE acc={rec['accuracy']:.4f} kappa={rec['kappa']:.4f} time={rec['meta_training_time_seconds']}s")
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
