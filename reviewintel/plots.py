"""plots.py - all figures used in the report and on the web 'Model' page."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .nn import ACTIVATIONS, MLP, Optimizer, Perceptron

BLUE, RED, GREEN, ORANGE, PURPLE, GREY, TEAL, PINK = "#4F6AF5", "#EF4444", "#22A06B", "#F59E0B", "#8B5CF6", "#6B7280", "#14B8A6", "#EC4899"
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, TEAL, PINK, GREY]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110, "savefig.bbox": "tight"})


def _save(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- activations
def fig_activations(path):
    z = np.linspace(-5, 5, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (name, (f, d)), c in zip([(k, v) for k, v in ACTIVATIONS.items() if k != "linear"], PALETTE):
        a = f(z)
        ax[0].plot(z, a, label=name, color=c, lw=2)
        ax[1].plot(z, d(z, a), label=name, color=c, lw=2)
    ax[0].set_title("Activation functions  f(z)")
    ax[1].set_title("Derivatives  f'(z)  (what back-propagation multiplies by)")
    ax[0].set_ylim(-1.3, 3)
    ax[1].set_ylim(-0.1, 1.2)
    for a in ax:
        a.set_xlabel("z (weighted sum)")
        a.legend()
    _save(fig, path)


# ---------------------------------------------------------------- cost functions
def fig_cost_functions(path):
    a = np.linspace(0.001, 0.999, 500)
    z = np.linspace(-8, 8, 500)
    s = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(a, -np.log(a), color=BLUE, lw=2, label="BCE  (true label y=1)")
    ax[0].plot(a, (1 - a) ** 2, color=RED, lw=2, label="MSE  (true label y=1)")
    ax[0].set_xlabel("predicted probability a")
    ax[0].set_ylabel("cost")
    ax[0].set_title("Cost of one prediction")
    ax[0].set_ylim(0, 5)
    ax[0].legend()
    ax[1].plot(z, s - 1, color=BLUE, lw=2, label="BCE:  dJ/dz = a - y")
    ax[1].plot(z, 2 * (s - 1) * s * (1 - s), color=RED, lw=2, label="MSE:  dJ/dz = 2(a-y)a(1-a)")
    ax[1].set_xlabel("z (pre-activation) for a sample with y = 1")
    ax[1].set_ylabel("gradient")
    ax[1].set_title("Gradient seen by back-prop\n(MSE vanishes when the model is confidently wrong)")
    ax[1].legend(loc="lower right", fontsize=8)
    ax[2].plot(z, -np.log(s), color=BLUE, lw=2, label="BCE")
    ax[2].plot(z, (1 - s) ** 2, color=RED, lw=2, label="MSE")
    ax[2].set_xlabel("z for a sample with y = 1")
    ax[2].set_title("Cost vs. logit")
    ax[2].legend()
    _save(fig, path)


# ---------------------------------------------------------------- perceptron / XOR
def xor_demo(seed=0):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    perc = Perceptron(lr=0.1, epochs=100, seed=seed).fit(X, y)
    mlp = MLP([2, 6, 1], "tanh", "sigmoid", "bce", seed=3, dtype=np.float64)
    mlp.fit(X, y.reshape(-1, 1).astype(float), epochs=3000, batch_size=None, optimizer="adam", lr=0.02)
    return X, y, perc, mlp


def fig_xor(path, X, y, perc, mlp):
    gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    G = np.c_[gx.ravel(), gy.ravel()]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, pred, title in [(ax[0], perc.predict(G), f"Single perceptron  (accuracy on XOR = {np.mean(perc.predict(X) == y):.0%})"),
                           (ax[1], mlp.predict(G.astype(np.float64)), f"MLP with 1 hidden layer  (accuracy = {np.mean(mlp.predict(X) == y):.0%})")]:
        a.contourf(gx, gy, pred.reshape(gx.shape), levels=[-0.5, 0.5, 1.5], colors=["#DBEAFE", "#FEE2E2"])
        a.scatter(X[:, 0], X[:, 1], c=[BLUE if k == 0 else RED for k in y], s=140, edgecolor="k", zorder=3)
        a.set_title(title, fontsize=10)
        a.set_xlabel("x1")
        a.set_ylabel("x2")
        a.grid(False)
    _save(fig, path)


def fig_perceptron_errors(path, errs):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(range(1, len(errs) + 1), errs, marker="o", color=BLUE, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("misclassified samples (updates)")
    ax.set_title("Perceptron learning on review features")
    _save(fig, path)


# ---------------------------------------------------------------- training curves
def fig_cost_compare(path, runs):
    """runs: name -> hist.  Solid = normal start, dashed = 'confidently wrong' start."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for (n, h), c in zip(runs.items(), PALETTE):
        ls = "--" if "wrong" in n else "-"
        ax[0].plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label=n, color=c, lw=2, ls=ls)
        ax[1].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=n, color=c, lw=2, ls=ls)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("training cost (own scale, log)")
    ax[0].set_title("Cost vs. epochs")
    ax[0].set_yscale("log")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("validation accuracy")
    ax[1].set_title("Validation accuracy vs. epochs")
    ax[1].legend(fontsize=7, loc="lower right")
    _save(fig, path)


def fig_loss_curve(path, h, title="Training and validation cost vs. epochs"):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    e = range(1, len(h["train_loss"]) + 1)
    ax[0].plot(e, h["train_loss"], color=BLUE, lw=2, label="train")
    ax[0].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], color=RED, lw=2, label="validation")
    if h.get("best_epoch"):
        ax[0].axvline(h["best_epoch"], color=GREEN, ls="--", label=f"best epoch = {h['best_epoch']}")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("binary cross-entropy")
    ax[0].set_title(title)
    ax[0].legend()
    ax[1].plot(e, h["train_acc"], color=BLUE, lw=2, label="train")
    ax[1].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], color=RED, lw=2, label="validation")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy vs. epochs")
    ax[1].legend()
    if h["grad_norm"]:
        ax[2].plot(range(1, len(h["grad_norm"]) + 1), h["grad_norm"], color=PURPLE, lw=2)
        ax[2].set_yscale("log")
        ax[2].set_xlabel("epoch")
        ax[2].set_ylabel("||gradient||")
        ax[2].set_title("Gradient norm -> 0 means convergence")
    _save(fig, path)


def fig_gd_variants(path, runs, summary):
    """runs: label -> hist ; summary: list of dict(label, val_f1, time, updates)."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1.35, 1, 0.8], wspace=0.55))
    for (n, h), c in zip(runs.items(), PALETTE + ["#111827", "#A16207"]):
        ax[0].plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label=n, color=c, lw=2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("training cost (log scale)")
    ax[0].set_title("Convergence of each variant")
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    labels = [f"{s['label']}\n({s['updates']:,} updates)" for s in summary]
    ax[1].barh(labels, [s["val_f1"] for s in summary], color=BLUE)
    ax[1].set_xlabel("validation F1 (spam)")
    ax[1].set_title("Quality after the same number of epochs")
    ax[1].invert_yaxis()
    ax[2].barh(labels, [s["time"] for s in summary], color=ORANGE)
    ax[2].set_xlabel("training time (s)")
    ax[2].set_title("Wall-clock cost")
    ax[2].invert_yaxis()
    ax[2].set_yticklabels([])
    for a in ax[1:]:
        a.grid(axis="y", alpha=0)
    _save(fig, path)


def fig_gd_trajectories(path, seed=3):
    """Toy 2-D ravine  f = 0.1 x^2 + 2 y^2 : how each update rule travels (noisy gradients mimic mini-batches)."""
    f = lambda x, y: 0.1 * x ** 2 + 2.0 * y ** 2
    grad = lambda p: np.array([0.2 * p[0], 4.0 * p[1]])
    start = np.array([-8.0, 2.0])
    setups = [("Batch GD", "gd", 0.40, 0.0), ("SGD (noisy)", "gd", 0.40, 0.4), ("Momentum", "momentum", 0.03, 0.4),
              ("Nesterov", "nesterov", 0.03, 0.4), ("AdaGrad", "adagrad", 1.5, 0.4), ("RMSProp", "rmsprop", 0.20, 0.4), ("Adam", "adam", 0.20, 0.4)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8))
    fig.subplots_adjust(hspace=0.35)
    gx, gy = np.meshgrid(np.linspace(-9, 3, 200), np.linspace(-3, 3, 200))
    for ax, (name, opt, lr, noise), c in zip(axes.ravel(), setups, PALETTE):
        rng = np.random.default_rng(seed)
        p = start.copy()
        o = Optimizer(opt, lr)
        path_ = [p.copy()]
        for _ in range(50):
            g = grad(p) + rng.normal(0, noise, 2)
            o.step([p], [g])
            path_.append(p.copy())
        P = np.array(path_)
        ax.contour(gx, gy, f(gx, gy), levels=18, cmap="Blues", alpha=0.7)
        ax.plot(P[:, 0], P[:, 1], "-o", color=c, ms=3, lw=1.3)
        ax.plot(*start, "ks", label="start")
        ax.plot(0, 0, "*", color=GREEN, ms=14)
        ax.set_title(f"{name}  (lr={lr})\nfinal f = {f(*P[-1]):.3f}", fontsize=10)
        ax.set_xlim(-9, 3)
        ax.set_ylim(-3, 3)
        ax.grid(False)
    axes.ravel()[-1].axis("off")
    _save(fig, path)


def fig_hyper(path, res):
    """res: dict hyperparam -> dict(values=[...], mean=[...], std=[...]) (validation F1)."""
    titles = {"lr": "Learning rate", "neurons": "Hidden neurons", "layers": "Hidden layers",
              "batch": "Batch size", "activation": "Hidden activation", "epochs": "Number of epochs"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for ax, key in zip(axes.ravel(), ["lr", "neurons", "layers", "batch", "activation", "epochs"]):
        r = res[key]
        x = np.arange(len(r["values"]))
        ax.errorbar(x, r["mean"], yerr=r["std"], marker="o", color=BLUE, lw=2, capsize=3)
        best = int(np.argmax(r["mean"]))
        ax.plot(x[best], r["mean"][best], "o", color=GREEN, ms=11, zorder=5, label="best")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in r["values"]], rotation=0)
        ax.set_title(titles[key])
        ax.set_ylabel("validation F1 (spam)")
        ax.legend(loc="lower right")
    _save(fig, path)


def fig_epochs(path, h_long):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    e = np.arange(1, len(h_long["train_loss"]) + 1)
    ax.plot(e, h_long["train_loss"], color=BLUE, lw=2, label="train cost")
    ax.plot(e, h_long["val_loss"], color=RED, lw=2, label="validation cost")
    b = int(np.argmin(h_long["val_loss"])) + 1
    ax.axvline(b, color=GREEN, ls="--", label=f"min val cost @ epoch {b}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cost")
    ax.set_title("Number of epochs: under-fitting -> best -> over-fitting")
    ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------- evaluation
def fig_confusion(path, cm, labels, title, cmap="Blues"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.4 + 0.5 * len(labels), 3.8 + 0.4 * len(labels)))
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    _save(fig, path)


def fig_roc(path, curves):
    """curves: name -> (fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for (n, (fpr, tpr, auc)), c in zip(curves.items(), PALETTE):
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{n} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (spam = positive)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def fig_model_ladder(path, rows):
    names = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.2
    x = np.arange(len(names))
    for k, (m, c) in enumerate(zip(["accuracy", "precision", "recall", "f1"], [BLUE, GREEN, ORANGE, RED])):
        ax.bar(x + (k - 1.5) * w, [r[m] for r in rows], w, label=m, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Test-set metrics: from perceptron to tuned MLP")
    ax.legend(ncol=4, loc="lower right", fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------- diagrams
def fig_pipeline(path):
    from matplotlib.patches import FancyBboxPatch
    steps = ["1  Dataset\n(reviews, ratings)", "2  Pre-processing\n(clean, tokens)", "3  Feature extraction\nTF-IDF + 10 style feats",
             "4  Perceptron\nbaseline", "5  MLP / ANN\nforward propagation",
             "6  Cost function\nBCE vs MSE", "7  Back-propagation\n+ GD variants", "8  Hyper-parameter\ntuning",
             "9  Best model\n(spam + sentiment)", "10  Flask web app\nGoogle Cloud Run"]
    cols = [GREY, GREY, GREY, ORANGE, BLUE, RED, GREEN, PURPLE, TEAL, PINK]
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 2.3)
    ax.axis("off")
    for i, (t, c) in enumerate(zip(steps, cols)):
        r, k = divmod(i, 5)
        x, y = 0.05 + k * 1.03, 1.35 - r * 1.2
        ax.add_patch(FancyBboxPatch((x, y), 0.85, 0.75, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="none"))
        ax.text(x + 0.425, y + 0.375, t, ha="center", va="center", color="white", fontsize=8.6, fontweight="bold")
        if k < 4:
            ax.annotate("", xy=(x + 1.0, y + 0.375), xytext=(x + 0.88, y + 0.375), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.6))
    ax.annotate("", xy=(0.475, 1.28 - 0.0), xytext=(4.575, 1.28), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.4, connectionstyle="angle,angleA=0,angleB=90,rad=0"), annotation_clip=False)
    _save(fig, path)


def fig_network(path, n_in, hidden, n_out=1):
    layers = [n_in] + list(hidden) + [n_out]
    names = ["Input\n(TF-IDF + style)"] + [f"Hidden {i + 1}" for i in range(len(hidden))] + ["Output\n(P spam)"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axis("off")
    show = [min(n, 6) for n in layers]
    xs = np.linspace(0.08, 0.92, len(layers))
    pts = []
    for x, k, n, name in zip(xs, show, layers, names):
        ys = np.linspace(0.2, 0.8, k) if k > 1 else [0.5]
        pts.append([(x, y) for y in ys])
        ax.text(x, 0.06, f"{name}\n[{n}]", ha="center", fontsize=8.5)
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#CBD5E1", lw=0.5, zorder=1)
    for i, layer in enumerate(pts):
        for p in layer:
            ax.scatter(*p, s=140, color=[BLUE, GREEN, TEAL, PURPLE, ORANGE][min(i, 4)] if i < len(pts) - 1 else RED, zorder=3, edgecolor="k")
        if layers[i] > 6:
            ax.text(layer[0][0], 0.5, "...", ha="center", fontsize=16, zorder=5, backgroundcolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, path)
