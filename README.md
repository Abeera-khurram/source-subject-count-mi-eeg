# Effect of Source Subject Count on Cross-Subject MI-EEG Decoding
Code and per-run results for the ICCII 2026 paper "Effect of Source Subject Count on Cross-Subject Motor Imagery EEG Decoding: A DeepConvNet Study with a Meta-Learning Fine-Tuning Comparison".
- src/, scripts/, configs/: Stage 1 (DeepConvNet pretraining) and Stage 2 (meta-learning fine-tuning) code.
- results/: per-run result tables (deepconv_and_meta_combined.csv contains all runs).
- logs/: training logs. figures/: paper figures.
- Data: BCI Competition IV dataset 2a (public), not included. Model checkpoints are not included.
- The Stage 2 procedure follows MLEMSDA (Miao et al., J. Neurosci. Methods, 2026). The authors' reference code is not redistributed here; obtain it from the repository linked in that paper.
