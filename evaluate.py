"""Regenerate paired samples, metrics and figures from trained checkpoints."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from experiment import MODES, ROOT, centers, data, np, pixels, sample, threadpool_limits


def metrics(images, labels, ref, ref_y, clf):
    # Biased class-conditional MMD squared. Same reference for all candidates.
    discrepancies = []
    spreads = []
    for k in range(10):
        a, b = images[labels == k], ref[ref_y == k]
        def kernel(u, v):
            return rbf_kernel(u, v, gamma=1 / 32).mean()
        discrepancies.append(kernel(a, a) + kernel(b, b) - 2 * kernel(a, b))
        spreads.append(a.var(0).mean())
    return [float(np.mean(clf.predict(images) == labels)), float(np.mean(discrepancies)),
            float(np.mean(spreads))]


def main(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    tr, y, te, yt = data()
    train_pixels, test_pixels = pixels(tr), pixels(te)
    clf = SVC(C=10, gamma="scale").fit(train_pixels, y)
    labels = np.tile(np.arange(10), 100)
    proto = pixels(centers(tr, y, "class")[labels])
    seeds = sorted(int(path.parent.name.split("-")[-1])
                   for path in runs.glob("baseline-*/checkpoint.npz"))
    if not seeds:
        raise ValueError("Train a baseline before evaluation.")
    rows = []
    examples = {}
    for mode in (*MODES, "prototype"):
        for seed in seeds:
            imgs = proto if mode == "prototype" else sample(
                runs / f"{mode}-{seed}/checkpoint.npz", labels, 1201)
            rows.append([mode, seed, *metrics(imgs, labels, test_pixels, yt, clf)])
            if seed == seeds[0]:
                examples[mode] = imgs
            if mode == "prototype":
                break
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "train_seed", "label_agreement", "conditional_mmd2", "within_class_variance"])
        writer.writerows(rows)
    real = [np.mean(clf.predict(test_pixels) == yt),
            np.mean([test_pixels[yt == k].var(0).mean() for k in range(10)])]
    (output / "reference.txt").write_text(
        f"real_test_accuracy={real[0]:.8f}\nreal_test_within_class_variance={real[1]:.8f}\n", encoding="utf-8")
    np.savez_compressed(output / "paired_samples.npz", labels=labels, **examples)
    def grid(ax, arr, caption):
        a = arr.reshape(-1, 10, 8, 8)
        tile = np.pad(a, ((0,0),(0,0),(1,1),(1,1)), constant_values=-1)
        tile = tile.transpose(0,2,1,3).reshape(len(a)*10, 100)
        ax.imshow(tile, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        ax.set(xticks=[], yticks=[], title=caption)
        ax.set_axis_off()
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), constrained_layout=True)
    for ax, (mode, caption) in zip(axes, zip((*MODES, "prototype"),
            ("Baseline", "Global mean residual", "Class mean residual", "Class mean only"))):
        grid(ax, examples[mode][:20], caption)
    fig.savefig(output / "paired.png", dpi=145)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 3.5), constrained_layout=True)
    for mode, color in zip(MODES, ("#6a7584","#d88e29","#087e79")):
        vals = [np.loadtxt(runs / f"{mode}-{s}/loss.csv", delimiter=",", skiprows=1, ndmin=2)
                for s in seeds]
        ax.plot(vals[0][:,0], np.mean([v[:,1] for v in vals], axis=0), label=mode, color=color)
    ax.set(xlabel="Training updates", ylabel="Noise prediction MSE")
    ax.legend(frameon=False)
    fig.savefig(output / "loss.png", dpi=150)
    plt.close(fig)
    print(real)
    for mode in (*MODES, "prototype"):
        print(mode, np.mean([r[2:] for r in rows if r[0] == mode], axis=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--runs", type=Path, default=ROOT / "runs")
    cli.add_argument("--out", type=Path, default=ROOT / "results")
    args = cli.parse_args()
    with threadpool_limits(limits=1):
        main(args.runs, args.out)
