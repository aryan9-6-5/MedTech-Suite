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
