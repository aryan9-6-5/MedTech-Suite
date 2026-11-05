"""Acquire, verify and cache the three polyp datasets: Kvasir-SEG (train / val / internal test) and two evaluation-only external sets.

Sources (all research / educational use only, see README):
  kvasir  official zip from datasets.simula.no, downloaded automatically.
  cvc     CVC-ClinicDB (612 pairs) as published on the official MICCAI 2015 polyp challenge page (Dropbox .rar), extracted to data/cvc.
  etis    ETIS-Larib (196 pairs) from Zenodo 10.5281/zenodo.5579392; only the ETIS members are read out of the 4 GB archive by HTTP range
          requests and written byte for byte to data/etis. (The official ETIS page asks for an account, which has to be created by hand.)

Every set must have exactly the expected number of image / mask pairs, with no unmatched file on either side and identical image / mask
sizes, otherwise loading fails. build() writes manifest.json (paths, sizes, SHA-256) and data/cache.npz. The external sets are made
read-only and verify_external() re-hashes them against the manifest, so they cannot be changed unnoticed between runs.
"""
import hashlib
import http.client
import io
import json
import os
import shutil
import stat
import struct
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MANIFEST = ROOT / "manifest.json"
SIZE = 352                      # divisible by 32; images and masks are resized, so Dice is measured at this size
EXPECTED = {"kvasir": 1000, "cvc": 612, "etis": 196}
EXTERNAL = ("cvc", "etis")      # evaluation only: never used for training, validation or any tuning
SPLIT_SEED = 0                  # fixed, independent of the training seed: both encoders see identical splits
N_VAL, N_TEST = 100, 100

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
CVC_URL = "https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=1"   # link on the archived official page polyp.grand-challenge.org/CVCClinicDB
ZENODO_URL = "https://zenodo.org/records/5579392/files/PolypDataset.zip?download=1"
ZENODO_ETIS = "TestDataset/TestDataset/ETIS-LaribPolypDB/"
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
SOURCES = {
    "kvasir": {"source": KVASIR_URL, "reference": "Jha et al. 2020, Kvasir-SEG", "licence": "research and educational use only"},
    "cvc": {"source": "https://polyp.grand-challenge.org/CVCClinicDB/ (file: " + CVC_URL + ")", "reference": "Bernal et al. 2015, CVC-ClinicDB",
            "licence": "research and educational use only, commercial use forbidden"},
    "etis": {"source": "zenodo.org/records/5579392, PolypDataset.zip, folder " + ZENODO_ETIS, "reference": "Silva et al. 2014, ETIS-Larib Polyp DB",
             "licence": "research and educational use only, commercial use forbidden (official page); Zenodo record states no licence"},
}


# ---------------------------------------------------------------------------------------------------------- download helpers
# datasets.simula.no sends a stale certificate chain: its certificate was issued by the Sectigo R36 intermediate, which the server does not send.
# Browsers and Windows fetch it on their own; OpenSSL (Linux / Colab) does not. The intermediate is fetched from the issuer's published
# URL and added to the trusted roots for this one download. OpenSSL still checks its signature and the whole chain, so verification stays on.
SIMULA_INTERMEDIATE = "http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt"


def _ca_bundle_with_intermediate():
    import ssl
    der = urllib.request.urlopen(SIMULA_INTERMEDIATE, timeout=60).read()
    cafile = ssl.get_default_verify_paths().cafile
    if not cafile or not Path(cafile).exists():
        import certifi
        cafile = certifi.where()
    out = DATA / "_raw" / "ca-bundle-simula.pem"
    out.write_text(Path(cafile).read_text() + "\n" + ssl.DER_cert_to_PEM_cert(der))
    return out


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}", flush=True)
    # curl, not urllib: Python's ssl cannot complete Simula's chain on Windows. Certificate checking stays on in every path.
    cmd = ["curl", "-fsSL", "-o", str(dest), url]
    r = subprocess.run(cmd)
    if r.returncode == 60 and "simula.no" in url:             # 60 = certificate problem
        r = subprocess.run(cmd[:1] + ["--cacert", str(_ca_bundle_with_intermediate())] + cmd[1:])
    r.check_returncode()
    assert dest.stat().st_size > 0, f"{url}: empty download"


def _extract_archive(archive, out):
    """zip via zipfile; rar via the bsdtar that ships with Windows 10+ / most Linux (tar -xf)."""
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        win_tar = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "tar.exe"     # bsdtar; GNU tar cannot read rar
        cmds = [([t, "-xf", str(archive), "-C", str(out)]) for t in (shutil.which("bsdtar"), str(win_tar) if win_tar.exists() else None) if t]
        cmds += [[t, "x", "-y", f"-o{out}", str(archive)] for t in (shutil.which("7zz"), shutil.which("7z")) if t]
        if shutil.which("unrar"):
            cmds.append(["unrar", "x", "-y", str(archive), str(out) + os.sep])
        if not any(subprocess.run(c).returncode == 0 for c in cmds):
            raise SystemExit(f"could not extract {archive}: install bsdtar (apt install libarchive-tools), 7-Zip or unrar, or extract it into {out} by hand")


def _has_files(d):
    return d.exists() and any(p.is_file() for p in d.rglob("*"))


def _fetch_kvasir():
    if _has_files(DATA / "kvasir"):
        return
    z = DATA / "_raw" / "kvasir-seg.zip"
    _download(KVASIR_URL, z)
    _extract_archive(z, DATA / "kvasir")


def _fetch_cvc():
    if _has_files(DATA / "cvc"):
        return
    rar = DATA / "_raw" / "cvc" / "CVC-ClinicDB.rar"
    if not rar.exists():
        _download(CVC_URL, rar)
    _extract_archive(rar, DATA / "cvc")


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, so a few members of a huge zip can be read without downloading all of it."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                d = urllib.request.urlopen(req, timeout=120).read()
                break
            except (OSError, http.client.HTTPException):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        assert len(d) == n, f"short range read ({len(d)} of {n} bytes)"
        b[:n] = d
        self.pos += n
        return n


def _fetch_etis():
    d = DATA / "etis"
    if _has_files(d):
        return
    tmp = DATA / "etis.part"                                   # written aside and renamed, so an interrupted run never leaves a partial set
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"reading the ETIS-Larib members of {ZENODO_URL} by range requests", flush=True)
    with zipfile.ZipFile(io.BufferedReader(_RangeFile(ZENODO_URL), buffer_size=1 << 20)) as z:
        members = [m for m in z.infolist() if m.filename.startswith(ZENODO_ETIS) and not m.is_dir()]
        assert len(members) == 2 * EXPECTED["etis"], f"expected {2 * EXPECTED['etis']} ETIS files in the archive, found {len(members)}"
        for m in members:
            rel = m.filename[len(ZENODO_ETIS):]               # images/<n>.png or masks/<n>.png
            assert rel.split("/")[0] in ("images", "masks"), rel
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(m))
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    print(f"wrote {len(members)} ETIS-Larib files", flush=True)


# --------------------------------------------------------------------------------------------------------------- reading
def _find_dir(root, names):
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name.lower() in names:
            return p
    raise FileNotFoundError(f"no folder named {names} under {root}")


def _files(folder):
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXT}


def _read_tiff_rgb(path):
    """CVC-ClinicDB's original frames are uncompressed 8-bit RGB TIFFs tagged PhotometricInterpretation=BlackIsZero with 3 samples per pixel,
    which Pillow refuses to open. Read the strips directly (little-endian, no compression, chunky layout) and fail on anything else."""
    d = path.read_bytes()
    assert d[:4] == b"II*\x00", f"{path}: not a little-endian TIFF"
    off = struct.unpack("<I", d[4:8])[0]
    n = struct.unpack("<H", d[off:off + 2])[0]
    tags = {}
    for i in range(n):
        e = off + 2 + 12 * i
        t, ty, c, v = struct.unpack("<HHII", d[e:e + 12])
        size, fmt = {1: (1, "B"), 2: (1, "B"), 3: (2, "H"), 4: (4, "I")}[ty]
        raw = d[e + 8:e + 12] if c * size <= 4 else d[v:v + c * size]
        tags[t] = struct.unpack(f"<{c}{fmt}", raw[:c * size])
    assert tags[259] == (1,) and tags[284] == (1,) and tags[258][0] == 8, f"{path}: unsupported TIFF layout"
    width, height, spp = tags[256][0], tags[257][0], tags[277][0]
    buf = b"".join(d[o:o + k] for o, k in zip(tags[273], tags[279]))
    assert len(buf) == width * height * spp, f"{path}: strip data does not add up"
    return np.frombuffer(buf, np.uint8).reshape(height, width, spp)


def _read_rgb(path):
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except UnidentifiedImageError:
        a = _read_tiff_rgb(path)
        assert a.shape[2] == 3, f"{path}: {a.shape[2]} channels"
        return a


def _read_mask(path):
    return np.asarray(Image.open(path).convert("L"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_set(name):
    root = DATA / name
    if not root.exists():
        raise SystemExit(f"{root} is missing")
    imgs, msks = _files(_find_dir(root, ("images", "original"))), _files(_find_dir(root, ("masks", "ground truth")))
    only_img, only_msk = sorted(set(imgs) - set(msks)), sorted(set(msks) - set(imgs))
    assert not only_img and not only_msk, f"{name}: {len(only_img)} images without mask {only_img[:5]}, {len(only_msk)} masks without image {only_msk[:5]}"
    stems = sorted(imgs, key=lambda s: (len(s), s))
    assert len(stems) == EXPECTED[name], f"{name}: expected {EXPECTED[name]} image/mask pairs, found {len(stems)}"
    X, M, rows = [], [], []
    for s in stems:
        im, mk = _read_rgb(imgs[s]), _read_mask(msks[s])
        assert im.shape[:2] == mk.shape, f"{name}/{s}: image {im.shape[:2]} and mask {mk.shape} differ in size"
        assert (mk > 127).any(), f"{name}/{s}: empty mask"
        X.append(np.asarray(Image.fromarray(im).resize((SIZE, SIZE), Image.BILINEAR)))
        M.append(np.asarray(Image.fromarray(mk).resize((SIZE, SIZE), Image.NEAREST)) > 127)
        rows.append({"dataset": name, "id": s, "image": imgs[s].relative_to(ROOT).as_posix(), "mask": msks[s].relative_to(ROOT).as_posix(),
                     "width": im.shape[1], "height": im.shape[0], "mask_width": mk.shape[1], "mask_height": mk.shape[0],
                     "polyp_area_fraction": round(float((mk > 127).mean()), 5),
                     "image_sha256": _sha256(imgs[s]), "mask_sha256": _sha256(msks[s])})
    return np.stack(X), np.stack(M), rows


def _lock(name):
    """Make an external set read-only on disk."""
    for p in (DATA / name).rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def verify_external():
    """Re-hash the external sets against manifest.json. Raises if any file was changed, added or removed since the manifest was written."""
    man = json.loads(MANIFEST.read_text())
    for name in EXTERNAL:
        rows = [r for r in man["files"] if r["dataset"] == name]
        assert len(rows) == EXPECTED[name], f"manifest has {len(rows)} {name} rows"
        for r in rows:
            for k in ("image", "mask"):
                assert _sha256(ROOT / r[k]) == r[k + "_sha256"], f"{r[k]} changed since the manifest was written"
        on_disk = sum(1 for p in (DATA / name).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
        assert on_disk == 2 * EXPECTED[name], f"{name}: {on_disk} image files on disk, manifest lists {2 * EXPECTED[name]}"
    print("external sets match manifest.json", flush=True)


def build():
    _fetch_kvasir(); _fetch_cvc(); _fetch_etis()
    out, files = {}, []
    for name in EXPECTED:
        out[f"X_{name}"], out[f"M_{name}"], rows = _load_set(name)
        files += rows
        print(f"{name}: {len(rows)} image/mask pairs", flush=True)
    part = np.zeros(EXPECTED["kvasir"], np.uint8)                  # 0 train, 1 val, 2 internal test
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(part))
    part[perm[:N_VAL]], part[perm[N_VAL:N_VAL + N_TEST]] = 1, 2
    out["part_kvasir"] = part
    role = {0: "train", 1: "validation", 2: "internal_test"}
    k = 0
    for r in files:
        if r["dataset"] == "kvasir":
            r["role"] = role[int(part[k])]; k += 1
        else:
            r["role"] = "external_test_only"
    man = {"image_size_for_training_and_dice": SIZE, "expected_pairs": EXPECTED, "sources": SOURCES, "split_seed": SPLIT_SEED,
           "note": "role external_test_only: never used for training, validation, model selection or tuning; files are read-only and verified by verify_external()",
           "files": files}
    MANIFEST.write_text(json.dumps(man, indent=1))
    for name in EXTERNAL:
        _lock(name)
    np.savez(DATA / "cache.npz", **out)
    return out


def load():
    f = DATA / "cache.npz"
    return dict(np.load(f)) if f.exists() else build()


if __name__ == "__main__":
    d = load()
    verify_external()
    p = d["part_kvasir"]
    print(f"kvasir train={int((p == 0).sum())} val={int((p == 1).sum())} test={int((p == 2).sum())}; cvc={len(d['X_cvc'])} etis={len(d['X_etis'])}")
    for n in EXPECTED:
        print(f"{n}: mean polyp area fraction {d[f'M_{n}'].mean():.3f}")
