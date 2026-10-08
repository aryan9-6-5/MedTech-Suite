# Day 2: HAM10000 skin lesion classifier

7-class dermoscopy classifier on [HAM10000](https://doi.org/10.7910/DVN/DBW86T) (10,015 images, 7,470 distinct lesions).

- **Classes:** akiec, bcc, bkl, df, mel, nv, vasc. `nv` is about 67% of the data, so accuracy alone is misleading.
- **Split:** 70/15/15 **grouped by `lesion_id`**. HAM10000 holds several photos of the same lesion; a random image-level split leaks near-duplicates across train and test and inflates every metric. The training script asserts zero lesion overlap.
- **Model:** ImageNet-pretrained ConvNeXt-Tiny fine-tuned at 192x256, AdamW + one-cycle, sqrt-inverse-frequency class weights, label smoothing, flip and brightness/contrast augmentation, mixed precision.
- **Metrics:** macro AUROC with a bootstrap 95% interval, balanced accuracy, per-class AUROC, melanoma recall. The test split is used once.
- **Demo:** Grad-CAM on a correct and a missed melanoma.

## Run

Colab (T4): open `train_colab.ipynb` and Run all. Local: download the three files from the Harvard Dataverse record into `data/` (`HAM10000_metadata.tab`, `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`), then:

```bash
pip install -r requirements.txt
cd src && python train.py && python evaluate.py
```

The dataset is CC BY-NC 4.0 and is not redistributed here. Research prototype, not a medical device.

## Results (single run, seed 42, test split used once)

| | Split by lesion (honest) | Random split (leaky) |
|---|---|---|
| Lesions shared by train and test | 0 | 496 |
| Test macro AUROC | **0.928** (95% CI 0.901 to 0.952) | 0.979 (95% CI 0.971 to 0.986) |
| Balanced accuracy | 0.678 | 0.880 |
| Accuracy | 0.783 | 0.905 |
| Melanoma AUROC | 0.899 | 0.962 |
| Melanoma recall (argmax) | 0.711 | 0.729 |

Same model, data and hyperparameters. Only the split changed. The two test sets are different samples, so the gap is indicative, not a controlled paired comparison.

Per-class AUROC (split by lesion): akiec 0.963, bcc 0.954, bkl 0.939, nv 0.939, vasc 0.937, mel 0.899, df 0.864.

Things to know:
- Validation macro AUROC peaked at epoch 5 (0.969) and settled near 0.95 afterwards. The saved model is the epoch-5 checkpoint chosen on validation, so validation is optimistic and the test number is the one to trust.
- About 21% of true melanomas were called benign nevi (see `results/confusion.png`). That is the costly error here.
- Grad-CAM (`results/gradcam_mel.png`) is a visualisation, not evidence of clinical reasoning. On the example shown, the highlighted region sits on the lesion border and a skin patch, not the darkest area. The missed melanoma was called actinic keratosis with 0.90 confidence.
- One seed, one split, one dataset from a small number of sources. Not externally validated.
