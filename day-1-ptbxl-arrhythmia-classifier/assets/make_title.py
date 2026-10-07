"""Decorative title card. The ECG trace is synthetic art, not dataset data."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

W, H = 1080, 1350
BG, TEAL, CYAN, WHITE, MUTED = "#070d1c", "#19e3c1", "#5ad1ff", "#f4f8ff", "#8a9bb8"
for f in ("segoeuib.ttf", "segoeui.ttf", "segoeuisl.ttf"):
    try: fm.fontManager.addfont(f"C:/Windows/Fonts/{f}")
    except Exception: pass
BOLD, REG = fm.FontProperties(fname="C:/Windows/Fonts/segoeuib.ttf"), fm.FontProperties(fname="C:/Windows/Fonts/segoeui.ttf")

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# soft glow blobs
yy, xx = np.mgrid[0:H, 0:W]
for cx, cy, r, col in [(180, 1150, 520, (0.10, 0.89, 0.76)), (950, 330, 600, (0.35, 0.82, 1.0))]:
    a = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2.2) ** 2))) * 0.16
    img = np.zeros((H, W, 4)); img[..., :3] = col; img[..., 3] = a
    ax.imshow(img, extent=[0, W, 0, H], origin="lower", zorder=0, aspect="auto")

# ECG paper grid
for x in range(0, W + 1, 27): ax.plot([x, x], [0, H], color="#12203d", lw=0.5 if x % 135 else 1.1, zorder=1)
for y in range(0, H + 1, 27): ax.plot([0, W], [y, y], color="#12203d", lw=0.5 if y % 135 else 1.1, zorder=1)

# synthetic ECG trace
def beat(t, c):
    g = lambda m, s, a: a * np.exp(-((t - m) ** 2) / (2 * s ** 2))
    return g(c - 0.20, 0.035, 0.14) - g(c - 0.045, 0.012, 0.16) + g(c, 0.012, 1.0) - g(c + 0.04, 0.014, 0.28) + g(c + 0.24, 0.05, 0.26)
t = np.linspace(0, 4, 3000)
sig = sum(beat(t, c) for c in (0.35, 1.3, 2.25, 3.2)) + 0.012 * np.random.default_rng(1).normal(size=t.size)
x = t / 4 * (W + 40) - 20; y = 590 + sig * 175
for lw, al in [(16, 0.06), (9, 0.12), (4.5, 0.3)]: ax.plot(x, y, color=TEAL, lw=lw, alpha=al, solid_capstyle="round", zorder=3)
ax.plot(x, y, color=TEAL, lw=2.6, zorder=4, solid_capstyle="round")
px, py = x[np.argmax(sig[:1000])], 590 + sig[:1000].max() * 175
ax.scatter([px], [py], s=120, color=WHITE, zorder=5); ax.scatter([px], [py], s=900, color=TEAL, alpha=0.18, zorder=4)

# chip
ax.add_patch(FancyBboxPatch((80, 1195), 360, 56, boxstyle="round,pad=0,rounding_size=28", fc="#0f2a33", ec=TEAL, lw=1.5, zorder=6))
ax.text(260, 1223, "MEDTECH SERIES  ·  1 OF 5", color=TEAL, fontproperties=BOLD, fontsize=15, ha="center", va="center", zorder=7)

# title
ax.text(80, 1105, "PTB-XL", color=WHITE, fontproperties=BOLD, fontsize=86, va="center", zorder=7)
ax.text(80, 1000, "Arrhythmia", color=WHITE, fontproperties=BOLD, fontsize=86, va="center", zorder=7)
ax.text(80, 895, "Classifier", color=TEAL, fontproperties=BOLD, fontsize=86, va="center", zorder=7)
ax.text(82, 805, "Teaching a neural network to read a 12-lead ECG", color=MUTED, fontproperties=REG, fontsize=23, va="center", zorder=7)

# stats row
stats = [("21,799", "clinical ECGs"), ("12", "leads"), ("5", "diagnostic classes")]
for i, (n, l) in enumerate(stats):
    cx = 80 + i * 320
    ax.text(cx, 395, n, color=WHITE, fontproperties=BOLD, fontsize=44, va="center", zorder=7)
    ax.text(cx, 340, l, color=MUTED, fontproperties=REG, fontsize=18, va="center", zorder=7)
ax.plot([80, W - 80], [270, 270], color="#233252", lw=1.5, zorder=6)
ax.text(80, 205, "1D ResNet   ·   PyTorch   ·   Macro AUROC   ·   Saliency", color=CYAN, fontproperties=BOLD, fontsize=20, va="center", zorder=7)
ax.text(80, 95, "github.com/aryan9-6-5/MedTech-Suite", color=MUTED, fontproperties=REG, fontsize=18, va="center", zorder=7)
ax.text(W - 80, 95, "Day 1", color=TEAL, fontproperties=BOLD, fontsize=22, va="center", ha="right", zorder=7)

fig.savefig("day1_title.png", dpi=100, facecolor=BG)
