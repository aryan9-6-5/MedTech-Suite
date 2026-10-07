# Day 1: PTB-XL arrhythmia classifier

Multi-label 12-lead ECG classifier on [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (21,799 records, 100 Hz, 10 s).

- **Task:** predict the 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP (a record can carry several).
- **Split:** official stratified folds. Train 1-8, validate 9 (model selection), test 10 (reported once).
- **Model:** 1D ResNet (6 residual blocks, global average pooling), AdamW + OneCycle, sqrt-balanced BCE, mixed precision.
- **Metric:** macro AUROC on fold 10, plus per-class AUROC and F1 at 0.5.
- **Demo:** input-gradient saliency over a test ECG for the MI output.

## Run

Colab (T4): open `train_colab.ipynb` and Run all. Local:

```bash
pip install -r requirements.txt
# put ptbxl_database.csv, scp_statements.csv and records100/ in data/
cd src && python train.py && python evaluate.py
```

## Results (test fold 10, 2,158 ECGs, single run, seed 42)

| Class | AUROC | F1 @ 0.5 |
|-------|-------|----------|
| NORM | 0.945 | 0.855 |
| STTC | 0.929 | 0.750 |
| MI | 0.920 | 0.727 |
| CD | 0.916 | 0.750 |
| HYP | 0.896 | 0.603 |
| **Macro** | **0.921** | |

Validation macro AUROC peaked at epoch 19 (0.924) and drifted down afterwards, so the saved model is the epoch-19 checkpoint selected on fold 9. Fold 10 was used once. This is one seed with no confidence intervals, and it sits slightly below the best published PTB-XL benchmark numbers, so treat it as a solid baseline, not a state-of-the-art claim.

Saliency (`results/saliency_mi.png`) is plain input-gradient magnitude. On the example shown it is diffuse across the trace and does not isolate a single waveform segment, so it shows sensitivity, not a clinical explanation.

Files in `results/`: `metrics.json`, `f1_at_0.5.json`, `roc.png`, `saliency_mi.png`.

Research prototype, not a medical device.
