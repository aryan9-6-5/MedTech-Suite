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
