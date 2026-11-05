"""Score all runs and apply the pre-registered decision. Writes results/summary.json.

Metric: mean per-image Dice (and IoU) at threshold 0.5. "External Dice" = mean of the CVC-ClinicDB and ETIS-Larib mean Dice.
Criterion: ConvNeXt beats ResNet-34 on external Dice by >= 0.010 as the mean over seeds, and is better in both seeds.
"""
import json

import numpy as np

from data import ROOT, load, verify_external

ENCS, SEEDS, MIN_GAIN = ("resnet34", "convnext"), (42, 7), 0.010
SETS = ("kvasir_test", "cvc", "etis")


def dice_iou(pred, gt):
    inter = (pred & gt).sum((1, 2)).astype(float)
    ps, gs = pred.sum((1, 2)), gt.sum((1, 2))
    return 2 * inter / np.maximum(ps + gs, 1), inter / np.maximum(ps + gs - inter, 1)


def boot_ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    d = load()
    verify_external()
    part = d["part_kvasir"]
    gt = {"kvasir_test": d["M_kvasir"][part == 2], "val": d["M_kvasir"][part == 1], "cvc": d["M_cvc"], "etis": d["M_etis"]}
    runs = {}
    for enc in ENCS:
        for s in SEEDS:
            p = np.load(ROOT / "results" / f"preds_{enc}_s{s}.npz")
            r = {}
            for k in SETS:
                dice, iou = dice_iou(p[k] > 127, gt[k])
                r[k] = {"dice": float(dice.mean()), "iou": float(iou.mean()), "dice_ci95_image_bootstrap": boot_ci(dice)}
            r["external_dice"] = (r["cvc"]["dice"] + r["etis"]["dice"]) / 2
            runs[f"{enc}_seed{s}"] = r
    agg = {}
    for enc in ENCS:
        rr = [runs[f"{enc}_seed{s}"] for s in SEEDS]
        agg[enc] = {k: float(np.mean([x[k]["dice"] for x in rr])) for k in SETS}
        agg[enc]["external_dice"] = float(np.mean([x["external_dice"] for x in rr]))
    gain = {s: runs[f"convnext_seed{s}"]["external_dice"] - runs[f"resnet34_seed{s}"]["external_dice"] for s in SEEDS}
    mean_gain = float(np.mean(list(gain.values())))
    met = mean_gain >= MIN_GAIN and all(g > 0 for g in gain.values())
    decision = (f"Pre-registered criterion (ConvNeXt beats ResNet-34 by at least {MIN_GAIN:.3f} external Dice in the mean and in both seeds) "
                f"{'MET' if met else 'NOT met'}: gains were " + " and ".join(f"{g:+.4f}" for g in gain.values()) + ".")
    out = {"min_gain_dice": MIN_GAIN, "runs": runs, "aggregate": agg, "convnext_minus_resnet34_external_dice_by_seed": {str(k): v for k, v in gain.items()},
           "criterion_met": met, "decision": decision}
    (ROOT / "results" / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"{'encoder':10s} " + " ".join(f"{k:>12s}" for k in SETS) + f" {'external':>10s}")
    for enc in ENCS:
        print(f"{enc:10s} " + " ".join(f"{agg[enc][k]:12.4f}" for k in SETS) + f" {agg[enc]['external_dice']:10.4f}")
    print(decision)


if __name__ == "__main__":
    main()
