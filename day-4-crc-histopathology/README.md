# Day 4: colorectal histopathology with external validation

Nine-class tissue classification on H&E-stained colorectal cancer slides, trained on **NCT-CRC-HE-100K** and evaluated on an independent set of new patients, **CRC-VAL-HE-7K**.

Data: Kather, Halama, Marx. *100,000 histological images of human colorectal cancer and healthy tissue*, Zenodo 10.5281/zenodo.1214456, CC BY 4.0 (224x224 px, 0.5 microns per pixel). Classes: adipose, background, debris, lymphocytes, mucus, smooth muscle, normal mucosa, cancer-associated stroma, tumour epithelium. Images come from a Hugging Face mirror (`1aurent/NCT-CRC-HE`); the code asserts that its class counts equal the counts in the Zenodo file lists, so a differing copy fails loudly.

## The honest-evaluation problem in this dataset

- **There is no slide or patient ID in the training set.** File names are random strings, and the 100,000 patches come from 86 slides. Any internal split therefore puts patches of the same slide on both sides, so internal accuracy is optimistic by construction. It is reported, but it is not the result.
- **The external set is the real test:** 7,180 patches from 50 patients, with no patient overlap with the training set according to the dataset description. These patients come from the same tissue bank, so this is new patients, not a new hospital. The external set also has no patient IDs, so confidence intervals bootstrapped over patches are optimistic.

## Design (fixed before running)

- **Training sample:** 2,500 patches per class (22,500), drawn at random from the 100K set. Internal validation: 250 per class. Internal test: 500 per class. External: all 7,180.
- **Model:** ImageNet-pretrained ConvNeXt-Tiny, fine-tuned for 6 epochs (AdamW, learning rate 3e-4, one-cycle, label smoothing 0.05). Fixed epochs and no model selection, so nothing is tuned on the external set.
- **Arm A:** geometric augmentation only (flips and transpose).
- **Arm B:** arm A plus haematoxylin / eosin stain jitter (alpha 10%, beta 0.08 in optical density) and brightness / contrast jitter.
- **Seeds:** 42 and 7 for each arm (4 runs).
- **Descriptive analyses, not part of the decision:** internal vs external gap, per-class recall, calibration (ECE), a nearest-neighbour similarity analysis showing how close internal and external test patches are to the training set, and Grad-CAM.

## Pre-registered decision (committed before any run)

Question: does stain-robust augmentation improve generalisation to new patients?

- **Criterion:** arm B beats arm A on **external macro-F1 by at least 1.0 percentage point, as the mean over the two seeds, and is better in both seeds**.
- **If met:** arm B is reported as the better model. **If not:** the result is "no clear gain from stain augmentation", arm A stays the headline, and the numbers are reported as they are.
- One-point margins are within what a single seed can swing, which is why two seeds are required to agree.

## Run

Colab (T4): open `train_colab.ipynb` and run the cells in order. About 15 GB of downloads (Hugging Face), then roughly 1 hour in total. Locally: `pip install -r requirements.txt`, then from `src/`: `python data.py`, `ARM=A SEED=42 python train.py` (and B, and seed 7), `python evaluate.py`, `python leak.py`, `python gradcam.py`.

Research prototype, not a medical device.

## Results (2 seeds per arm, external set used once per run)

| | Internal test (same 86 slides, optimistic) | External (50 new patients) |
|---|---|---|
| Geometry only (arm A), macro-F1 | 0.9968 | **0.9673** (accuracy 0.9772) |
| Geometry + stain jitter (arm B), macro-F1 | 0.9964 | 0.9689 (accuracy 0.9779) |

**Pre-registered decision: not met.** Arm B beat arm A on external macro-F1 by +0.13 and +0.18 points (seeds 42 and 7); the target was at least +1.0 point on average and in both seeds. Stain augmentation is therefore reported as showing **no clear gain**, and arm A stays the headline. Patch-level bootstrap interval for arm A external macro-F1: about 0.961 to 0.973 per seed (optimistic, see below).

**Where the gap is.** External recall per class, arm A, mean of two seeds: background and debris 100%, lymphocytes 99.9%, adipose 99.6%, smooth muscle 98.6%, mucus 98.4%, tumour 98.3%, normal mucosa 98.2%, and **cancer stroma 76.8%**. Internally, stroma recall is 99.5%. In the seed-42 confusion matrix, 20% of external stroma patches are called smooth muscle. Arm B seed 42 moved that boundary (stroma recall 84.3%, smooth-muscle recall 91.6%): a trade-off, not a clear gain. Per-class numbers for arm B seed 7 were not recoverable from the notebook printout (it was truncated); its aggregate numbers were.

**How close are test patches to the training set?** Nearest-neighbour cosine similarity in the fine-tuned model's embedding space: median 0.994 for internal test patches against 0.986 for external; 96.7% of internal patches have a training neighbour above 0.95, against 85.1% of external ones. Absolute values are high for any pair of patches, so read the difference, not the level.

**Caveats.**
- The internal split shares slides with training by construction (no slide IDs exist), so internal numbers are optimistic. The external set is the result.
- External patients come from the same tissue bank as the training slides: new patients, not a new hospital. External patch IDs carry no patient ID, so the bootstrap interval treats correlated patches as independent and is too narrow.
- Two seeds per arm. Grad-CAM (`results/gradcam_external.png`) uses a 7x7 feature map and mostly highlights patch borders; treat it as illustration, not evidence of how the model reasons.
- Research prototype, not a medical device.
