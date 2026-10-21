"""Grad-CAM on external patches: a confident correct tumour patch, a correct normal-mucosa patch, and the most confident error."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import CLASSES, NAMES, ROOT, load
from model import build_model, normalise
from train import dev

R = ROOT / "results"


def cam(model, x, cls):
    feats = {}
    h = model.features[-1].register_forward_hook(lambda m, i, o: feats.update(a=o))
    model.eval()
    x = x.clone().requires_grad_(True)
    out = model.classifier(model.avgpool(model.features(x)))
    h.remove()
    grad = torch.autograd.grad(out[0, cls], feats["a"])[0]
    c = torch.relu((grad.mean((2, 3), keepdim=True) * feats["a"]).sum(1, keepdim=True))
    c = torch.nn.functional.interpolate(c, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    c = c - c.min()
    return (c / (c.max() + 1e-9)).detach().cpu().numpy()


def main():
    d = load()
    z = np.load(R / "preds_A_s42.npz")
    p, y = z["ext"], z["ext_y"]
    pred, conf = p.argmax(1), p.max(1)
    tum, norm = CLASSES.index("TUM"), CLASSES.index("NORM")
    ok_t = np.flatnonzero((y == tum) & (pred == tum)); ok_n = np.flatnonzero((y == norm) & (pred == norm))
    err = np.flatnonzero(pred != y)
    picks = [("correct", ok[np.argmax(conf[ok])]) for ok in (ok_t, ok_n) if len(ok)]
    if len(err):
        picks.append(("error", err[np.argmax(conf[err])]))
    m = build_model(pretrained=False).to(dev)
    m.load_state_dict(torch.load(R / "model_A_s42.pt", map_location=dev))
    fig, axs = plt.subplots(2, len(picks), figsize=(3.6 * len(picks), 7.2), squeeze=False)
    meta, cams = [], []
    for k, (tag, i) in enumerate(picks):
        img = np.asarray(d["ext_x"][i])
        x = normalise(torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255)
        c = cam(m, x, int(pred[i]))
        axs[0, k].imshow(img); axs[0, k].set_title(f"{tag}: true {NAMES[CLASSES[y[i]]]}\npredicted {NAMES[CLASSES[pred[i]]]} ({conf[i]:.2f})", fontsize=9)
        axs[1, k].imshow(img); axs[1, k].imshow(c, cmap="jet", alpha=0.4); axs[1, k].set_title("Grad-CAM for the predicted class", fontsize=9)
        for a in axs[:, k]: a.axis("off")
        cams.append(c)
        meta.append({"tag": tag, "index": int(i), "true": CLASSES[y[i]], "pred": CLASSES[pred[i]], "conf": float(conf[i])})
    fig.tight_layout(); fig.savefig(R / "gradcam_external.png", dpi=150)
    (R / "gradcam_picks.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(R / "gradcam_arrays.npz", **{f"img{k}": np.asarray(d["ext_x"][m_["index"]]) for k, m_ in enumerate(meta)},
                        **{f"cam{k}": c.astype(np.float16) for k, c in enumerate(cams)})


if __name__ == "__main__":
    main()
