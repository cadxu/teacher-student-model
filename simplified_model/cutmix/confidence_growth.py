"""
Usage:
    python confidence_growth_viz.py <path_to_csv> [max_loops]

"""
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import accuracy_score

from train_ssl import ModelData, load_data, preprocess, make_rf
from augmentations import compositionalCutmix


FACTORS = [2, 5, 10, 20]
WEIGHTS = [0.3, 0.5, 0.7]

TAU = 0.85
N_REPEATS = 5

def loopAug(md, factor=10, weight=0.5, tau=0.85, max_loops=30, seed_offset=0):
    X_train = md.X_lab.copy()
    y_train = md.y_lab.copy()

    accuracies = []
    dataset_sizes = []
    model = None

    for loop in range(max_loops):

        X_aug, y_aug, w_aug = compositionalCutmix(
            X_train, y_train, factor=factor, weight=weight, seed=42 + loop + 1000 * seed_offset,
        )

        model = make_rf(seed=43 + seed_offset)
        model.fit(X_aug, y_aug, sample_weight=w_aug)
        acc = accuracy_score(md.y_test, model.predict(md.X_test))
        accuracies.append(acc)

        probs = model.predict_proba(md.X_unlab)
        confidences = np.max(probs, axis=1)
        pseudo_labels = np.argmax(probs, axis=1)
        mask = confidences >= tau

        X_pseudo = md.X_unlab[mask]
        y_pseudo = pseudo_labels[mask]

        if X_pseudo.size:
            X_train = np.vstack([X_train, X_pseudo])
            y_train = np.concatenate([y_train, y_pseudo])

        dataset_sizes.append(len(y_train))

        print(
            f"  [factor={factor} weight={weight}] Loop {loop + 1}/{max_loops} "
            f"| train_size={len(y_train)} | test_acc={acc:.4f}"
        )

    return model, accuracies, dataset_sizes


def plotAllCombos(md, results, starttime):
    n = len(results)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    max_loops = max(len(r[2]) for r in results)
    all_acc_lo = min(np.min(r[2] - r[3]) for r in results)
    all_acc_hi = max(np.max(r[2] + r[3]) for r in results)

    acc_pad = 0.05 * (all_acc_hi - all_acc_lo + 1e-9)
    acc_ylim = (all_acc_lo - acc_pad, all_acc_hi + acc_pad)
    xlim = (1, max_loops)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, (factor, weight, acc_mean, acc_std) in enumerate(results):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        loops = np.arange(1, len(acc_mean) + 1)

        ax.plot(loops, acc_mean, marker="^", color="tab:orange", label="Test Accuracy")
        ax.fill_between(loops, acc_mean - acc_std, acc_mean + acc_std,
                         color="tab:orange", alpha=0.2)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Loop")
        ax.set_title(f"factor={factor}, weight={weight}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(acc_ylim)
        ax.set_xlim(xlim)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.suptitle(
        f"Pseudo-labeling: Test Accuracy Across (cutmix factor, weight)\n"
        f"(mean \u00b1 std over {N_REPEATS} runs, shared axis scales, tau={TAU})",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"{md.name}.all_combos.{starttime}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {fname}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python confidence_growth_viz.py <path_to_csv> [max_loops]")
        return

    infile = sys.argv[1]
    max_loops = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    md = ModelData()
    md = load_data(infile, md)
    md = preprocess(md)
    md.name = "rf_cutmix"

    starttime = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    combos = list(itertools.product(FACTORS, WEIGHTS))
    print(f"Running {len(combos)} (factor, weight) combinations...")

    results = []
    for factor, weight in combos:
        run_accuracies = []
        for rep in range(N_REPEATS):
            _, accuracies, _ = loopAug(
                md, factor=factor, weight=weight, tau=TAU,
                max_loops=max_loops, seed_offset=rep,
            )
            run_accuracies.append(accuracies)

        run_accuracies = np.array(run_accuracies)  
        acc_mean = run_accuracies.mean(axis=0)
        acc_std = run_accuracies.std(axis=0)

        results.append((factor, weight, acc_mean, acc_std))

    plotAllCombos(md, results, starttime)


if __name__ == "__main__":
    main()