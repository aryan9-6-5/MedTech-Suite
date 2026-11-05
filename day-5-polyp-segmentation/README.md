# Day 5: polyp segmentation tested on unseen datasets

Pixel-level polyp segmentation in colonoscopy images. Train on **Kvasir-SEG**, test on two datasets the model never saw: **CVC-ClinicDB** and **ETIS-Larib**. The question is whether a stronger pretrained encoder carries over to other scopes and hospitals better than a smaller one.

Data (research and educational use; cite the dataset papers):
- Kvasir-SEG, 1,000 images with masks (Jha et al. 2020), `datasets.simula.no/kvasir-seg`.
- CVC-ClinicDB, 612 frames from 29 colonoscopy video sequences (Bernal et al. 2015).
- ETIS-Larib, 196 images (Silva et al. 2014).

`python src/data.py` fetches all three sets and fails loudly unless each has exactly 1000, 612 and 196 image/mask pairs, with no unmatched file and equal image and mask sizes. Sources:
- **Kvasir-SEG:** official zip, `datasets.simula.no/downloads/kvasir-seg.zip`.
- **CVC-ClinicDB:** the `.rar` linked from the official page `polyp.grand-challenge.org/CVCClinicDB/` (the live page has lost the link; it is on the archived copy). 612 original frames and 612 masks.
- **ETIS-Larib:** the official page asks for a Grand Challenge account, so the 196 image/mask pairs come from Zenodo `10.5281/zenodo.5579392` (`TestDataset/ETIS-LaribPolypDB`), read out of the archive by range requests. The Zenodo record states no licence.

All three are for research and educational use only; commercial use is forbidden. `manifest.json` lists every file with its dataset, role, size, mask path, source and SHA-256. CVC-ClinicDB and ETIS-Larib are marked `external_test_only`, are made read-only, and `verify_external()` re-hashes them against the manifest. Two data notes: Pillow cannot open CVC-ClinicDB's original frames (a colour tag in the TIFF), so `data.py` reads them directly; and ETIS-Larib contains three pairs of identical images with different masks (ids 20/22, 127/131, 172/175), kept as published.

## What makes the evaluation honest

- **Both external sets come from other sources than the training set**, so the gap between internal and external Dice is the result. The internal Kvasir test split is reported but is optimistic: Kvasir-SEG has no patient IDs, so a random split can put frames of the same patient on both sides.
- **CVC-ClinicDB is 612 frames from only 29 sequences**, so neighbouring frames are near-duplicates and the effective sample size is much smaller than 612. Image-level bootstrap intervals treat those frames as independent and are too narrow.
- **ETIS-Larib is the harder set**: only 196 images, with many small or flat polyps.

## Design (fixed before running)

- **Splits:** Kvasir-SEG 800 train, 100 validation, 100 internal test (random, split seed 0, identical for every run). All of CVC-ClinicDB and ETIS-Larib is used only for testing, once per run.
- **Models:** the same U-Net decoder on two ImageNet-pretrained encoders, **ResNet-34** (24.4M parameters in total) and **ConvNeXt-Tiny** (31.9M). Only the encoder differs.
- **Training:** 352x352 input, 40 epochs, AdamW (learning rate 3e-4, one-cycle), batch size 16, BCE plus Dice loss, flips / transpose and brightness / contrast jitter. The final-epoch model is used; the validation split is not used for selection, so nothing is tuned on the external sets.
- **Seeds:** 42 and 7 for each encoder (4 runs).
- **Metric:** mean per-image Dice at threshold 0.5, plus IoU. Masks and images are resized to 352x352, so Dice is measured at that size, not at native resolution. **External Dice** is the mean of the CVC-ClinicDB and ETIS-Larib Dice.

## Pre-registered decision (committed before any run)

Question: does the ConvNeXt-Tiny encoder generalise better to unseen datasets than ResNet-34?

- **Criterion:** ConvNeXt beats ResNet-34 on **external Dice by at least 0.010 (1.0 point), as the mean over the two seeds, and is better in both seeds**.
- **If met:** ConvNeXt is reported as the better encoder. **If not:** the result is "no clear gain from the stronger encoder", ResNet-34 stays the headline, and the numbers are reported as they are.
- A one-point margin is within what a single seed can swing on sets this small, which is why two seeds must agree.

## Run

Colab (T4): open `train_colab.ipynb` and run the cells in order (the datasets download themselves). Locally: `pip install -r requirements.txt`, then from `src/`: `python data.py`, `ENC=resnet34 SEED=42 python train.py` (and `ENC=convnext`, and seed 7), `python evaluate.py`, `python overlays.py`.

Research prototype, not a medical device.

## Results

Not run yet. This section is filled in after the four runs.
