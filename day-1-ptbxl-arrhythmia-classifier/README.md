# Day 1: PTB-XL arrhythmia classifier

Multi-label 12-lead ECG classifier on [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (21,799 records, 100 Hz, 10 s).

- **Task:** predict the 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP (a record can carry several).
- **Split:** official stratified folds. Train 1-8, validate 9 (model selection), test 10 (reported once).
- **Model:** 1D ResNet (6 residual blocks, global average pooling), AdamW + OneCycle, sqrt-balanced BCE, mixed precision.
- **Metric:** macro AUROC on fold 10, plus per-class AUROC and F1 at 0.5.
- **Demo:** gradient saliency over a test ECG showing where the MI logit is most sensitive.

## Run

Colab (T4): open `train_colab.ipynb` and Run all. Local:

```bash
pip install -r requirements.txt
# put ptbxl_database.csv, scp_statements.csv and records100/ in data/
cd src && python train.py && python evaluate.py
```

Results land in `results/` (`metrics.json`, `roc.png`, `saliency_mi.png`).

Research prototype, not a medical device.
