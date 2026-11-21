"""Small pixel-space MAR + diffusion loss. Run from any directory."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
MODES = ("baseline", "global", "class")
STEPS = 16
# t=0 is clean. Avoid division by zero at the noise endpoint.
angles = np.linspace(0, np.arccos(0.035), STEPS + 1)
A = np.cos(angles).astype("float32")
S = np.sin(angles).astype("float32")
eye = lambda n: np.eye(n, dtype="float32")


def data():
    d = load_digits()
    tr, te = train_test_split(np.arange(len(d.target)), test_size=0.2,
                              random_state=23, stratify=d.target)
    blocks = (d.images.astype("float32") / 8 - 1).reshape(-1, 4, 2, 4, 2)
    blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(-1, 16, 4)
    return blocks[tr], d.target[tr], blocks[te], d.target[te]


def pixels(blocks):
    return blocks.reshape(-1, 4, 4, 2, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 64)


def centers(x, y, mode):
    if mode == "class":
        return np.stack([x[y == k].mean(0) for k in range(10)])
    if mode == "global":
        return np.repeat(x.mean(0, keepdims=True), 10, axis=0)
    return np.zeros((10, 16, 4), dtype="float32")


def features(canvas, mask, y, pos):
    return np.concatenate([(canvas * mask[:, :, None]).reshape(len(y), 64),
                           mask, eye(10)[y], eye(16)[pos]], axis=1)


def initialize(rng):
    params = {}
    for i, (din, dout) in enumerate([(106, 64), (64 + 4 + STEPS, 80), (80, 4)]):
        params[f"w{i}"] = (rng.standard_normal((din, dout)) * (1 / np.sqrt(din))).astype("float32")
        params[f"b{i}"] = np.zeros(dout, dtype="float32")
    return params


def head(p, context, noisy, t):
    h = np.concatenate([context, noisy, eye(STEPS)[t - 1]], axis=1)
    hidden = np.tanh(h @ p["w1"] + p["b1"])
    # Analytic skip for a standard Gaussian target. Network learns the correction.
    pred = S[t, None] * noisy + hidden @ p["w2"] + p["b2"]
    return pred, h, hidden


def train(mode, seed, steps, batch, out):
    rng = np.random.default_rng(seed)
    x, y, _, _ = data()
    prior = centers(x, y, mode)
    p = initialize(rng)
    first = {k: np.zeros_like(v) for k, v in p.items()}
    second = {k: np.zeros_like(v) for k, v in p.items()}
    begin = time.perf_counter()
    folder = out / f"{mode}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    track = []
    with (folder / "loss.csv").open("w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["step", "mean_train_noise_mse"])
        for step in range(1, steps + 1):
            idx = rng.integers(len(x), size=batch)
            labels = y[idx]
            rank = rng.random((batch, 16)).argsort(axis=1).argsort(axis=1)
            visible = rng.integers(0, 4, size=batch) * 4
            mask = (rank < visible[:, None]).astype("float32")
            pos = (rank == visible[:, None]).argmax(1)
            clean = x[idx, pos] - prior[labels, pos]
            t = rng.integers(1, STEPS + 1, size=batch)
            noise = rng.standard_normal(clean.shape, dtype="float32")
            noisy = A[t, None] * clean + S[t, None] * noise
            inp = features(x[idx], mask, labels, pos)
            ctx = np.tanh(inp @ p["w0"] + p["b0"])
            pred, h, hidden = head(p, ctx, noisy, t)
            err = pred - noise
            track.append(np.mean(err ** 2))
            delta = (2 / err.size) * err
            dh = (delta @ p["w2"].T) * (1 - hidden ** 2)
            dc = (dh @ p["w1"][:64].T) * (1 - ctx ** 2)
            grads = {"w2": hidden.T @ delta, "b2": delta.sum(0),
                     "w1": h.T @ dh, "b1": dh.sum(0),
                     "w0": inp.T @ dc, "b0": dc.sum(0)}
            for key, g in grads.items():
                first[key] = 0.9 * first[key] + 0.1 * g
                second[key] = 0.999 * second[key] + 0.001 * g ** 2
                m = first[key] / (1 - 0.9 ** step)
                v = second[key] / (1 - 0.999 ** step)
                p[key] -= 0.001 * m / (np.sqrt(v) + 1e-8)
            if step % 100 == 0 or step == steps:
                log.writerow([step, float(np.mean(track))])
                track.clear()
    np.savez_compressed(folder / "checkpoint.npz", prior=prior, **p)
    cfg = dict(mode=mode, seed=seed, steps=steps, batch=batch, lr=0.001,
               seconds=round(time.perf_counter()-begin, 3), diffusion_steps=STEPS,
               split_seed=23, trainable_parameters=sum(z.size for z in p.values()),
               device="CPU", python=platform.python_version(), numpy=np.__version__)
    (folder / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(mode, seed, cfg["seconds"], flush=True)


def sample(checkpoint, labels, seed):
    with np.load(checkpoint) as saved:
        p = dict(saved)
    rng = np.random.default_rng(seed)
    n = len(labels)
    order = rng.random((n, 16)).argsort(axis=1)
    canvas = np.zeros((n, 16, 4), dtype="float32")
    mask = np.zeros((n, 16), dtype="float32")
    row = np.repeat(np.arange(n), 4)
    for start in range(0, 16, 4):
        pos = order[:, start:start + 4].ravel()
        inp = features(canvas[row], mask[row], labels[row], pos)
        ctx = np.tanh(inp @ p["w0"] + p["b0"])
        prior = p["prior"][labels[row], pos]
        noisy = rng.standard_normal(prior.shape, dtype="float32")
        for t in range(STEPS, 0, -1):
            noise, _, _ = head(p, ctx, noisy, np.full(len(row), t))
            clean = np.clip((noisy - S[t] * noise) / A[t] + prior, -1, 1) - prior
            noisy = A[t-1] * clean + S[t-1] * noise
        canvas[row, pos] = noisy + prior
        mask[row, pos] = 1
    return pixels(canvas)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--mode", choices=MODES, required=True)
    cli.add_argument("--seed", type=int, default=11)
    cli.add_argument("--steps", type=int, default=2500)
    cli.add_argument("--batch", type=int, default=128)
    cli.add_argument("--out", type=Path, default=ROOT / "runs")
    args = cli.parse_args()
    with threadpool_limits(limits=1):
        train(args.mode, args.seed, args.steps, args.batch, args.out)
