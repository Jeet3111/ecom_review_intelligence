#!/usr/bin/env python3
"""
run_pipeline.py  -  Indian E-commerce Review Intelligence : end-to-end experiment pipeline
===========================================================================================
Runs, in order
  1  dataset (synthetic generator OR your own CSV) + EDA
  2  pre-processing + feature extraction (TF-IDF + style features)
  3  gradient check (proves back-propagation is correct)
  4  Perceptron (+ XOR limitation demo)                       -> baseline
  5  MLP baseline                                             -> "baseline accuracy"
  6  Cost functions: BCE vs MSE                               (DLN topic 1)
  7  Variants of gradient descent (8 optimisers)              (DLN topic 3)
  8  Hyper-parameter tuning (6 hyper-parameters)
  9  Final model, convergence, evaluation (acc/prec/rec/F1/confusion matrix)
 10  Sentiment head (3-class softmax + categorical cross-entropy)
 11  Save models + figures + dashboard + results.json

Usage:   python run_pipeline.py            (full run, ~6-10 min on a laptop CPU)
         python run_pipeline.py --quick    (smoke test, ~1-2 min)
         python run_pipeline.py --data my_reviews.csv
"""
import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reviewintel import plots as P                              # noqa: E402
from reviewintel.dashboard import build_dashboard               # noqa: E402
from reviewintel.features import FeatureExtractor, HAND_NAMES   # noqa: E402
from reviewintel.nn import MLP, Perceptron, gradient_check      # noqa: E402
from reviewintel.synth import generate                          # noqa: E402

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
MOD = ROOT / "models"
T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def f(x, k=4):
    return round(float(x), k)


# ------------------------------------------------------------------------------------------ data
def load_data(args):
    if args.data:
        df = pd.read_csv(args.data)
        if not {"review_text", "is_spam"} <= set(df.columns):
            sys.exit("Custom CSV needs at least the columns: review_text, is_spam  (see README)")
        if df.is_spam.dtype == object:
            df["is_spam"] = df.is_spam.str.strip().str.lower().isin(["spam", "1", "true", "fake", "yes"]).astype(int)
        df["is_spam"] = df.is_spam.astype(int)
        for c, d in dict(review_id=None, product_id="P0", product_name="NA", category="General", platform="NA",
                         price_inr=np.nan, rating=np.nan, review_date=np.nan).items():
            if c not in df.columns:
                df[c] = [f"R{i + 1:06d}" for i in range(len(df))] if c == "review_id" else d
        df["review_date"] = pd.to_datetime(df.review_date, errors="coerce")
        df["sentiment"] = np.where(df.rating.isna(), None, np.where(df.rating >= 4, "Positive", np.where(df.rating == 3, "Neutral", "Negative")))
        df["spam_type"] = np.where(df.is_spam == 1, "spam", "genuine")
        src = f"custom CSV ({args.data})"
    else:
        df = generate(args.n, args.seed)
        df.to_csv(ROOT / "data" / "reviews.csv", index=False)
        src = f"synthetic generator (n={args.n}, seed={args.seed})"
    df["review_text"] = df.review_text.fillna("").astype(str)
    df["review_date"] = pd.to_datetime(df.review_date, errors="coerce")
    return df.reset_index(drop=True), src


def metrics(y, pred, prob=None):
    out = dict(accuracy=f(accuracy_score(y, pred)), precision=f(precision_score(y, pred, zero_division=0)),
               recall=f(recall_score(y, pred, zero_division=0)), f1=f(f1_score(y, pred, zero_division=0)))
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    if prob is not None:
        out["auc"] = f(roc_auc_score(y, prob))
    return out


# ------------------------------------------------------------------------------------------ experiment helper
class Runner:
    def __init__(self, Xtr, ytr, Xva, yva):
        self.Xtr, self.ytr, self.Xva, self.yva = Xtr, ytr, Xva, yva
        self.n_in = Xtr.shape[1]

    def run(self, cfg, seed=0):
        m = MLP([self.n_in] + list(cfg["hidden"]) + [1], cfg["act"], "sigmoid", cfg["loss"], l2=cfg.get("l2", 0.0), seed=seed)
        if cfg.get("out_bias") is not None:                       # "confidently wrong" initialisation demo
            m.b[-1][...] = cfg["out_bias"]
        t = time.perf_counter()
        h = m.fit(self.Xtr, self.ytr, self.Xva, self.yva.reshape(-1, 1).astype(np.float32), epochs=cfg["epochs"],
                  batch_size=cfg["bs"], optimizer=cfg["opt"], lr=cfg["lr"], seed=seed,
                  patience=cfg.get("patience"), track_grad=cfg.get("track_grad", False))
        h["time"] = time.perf_counter() - t
        if h["diverged"] or not np.isfinite(m.W[0]).all():
            return m, h, 0.0
        return m, h, f1_score(self.yva, m.predict(self.Xva), zero_division=0)

    def multi(self, cfg, seeds):
        scores = [self.run(cfg, s)[2] for s in range(seeds)]
        return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="optional CSV with review_text,is_spam[,rating,category,...]")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="small grids / fewer epochs (smoke test)")
    args = ap.parse_args()
    Q = args.quick
    for d in (FIG, TAB, MOD, ROOT / "webapp/static/dash", ROOT / "webapp/static/model"):
        d.mkdir(parents=True, exist_ok=True)
    R = {"quick_mode": Q}

    # ================================================================ 1. dataset
    log("1/11  dataset")
    df, src = load_data(args)
    R["dataset"] = dict(source=src, rows=int(len(df)), products=int(df.product_id.nunique()), categories=int(df.category.nunique()),
                        spam_rate=f(df.is_spam.mean()), mean_chars=f(df.review_text.str.len().mean(), 1),
                        category_counts=df.category.value_counts().to_dict())
    tv, te = train_test_split(df, test_size=0.15, stratify=df.is_spam, random_state=args.seed)
    tr, va = train_test_split(tv, test_size=0.15 / 0.85, stratify=tv.is_spam, random_state=args.seed)
    R["split"] = dict(train=len(tr), val=len(va), test=len(te))
    log(f"      {len(df)} reviews | spam {df.is_spam.mean():.1%} | train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    # ================================================================ 2. features
    log("2/11  pre-processing + feature extraction")
    fx = FeatureExtractor(max_features=1500 if not Q else 600, min_df=3).fit(tr.review_text)
    Xtr, Xva, Xte = (fx.transform(d.review_text) for d in (tr, va, te))
    fx.save(MOD / "features.json")
    ytr = tr.is_spam.values.reshape(-1, 1).astype(np.float32)
    yva, yte = va.is_spam.values, te.is_spam.values
    R["features"] = dict(n_tfidf=len(fx.vocab), n_hand=len(HAND_NAMES), n_total=fx.n_features, hand_names=HAND_NAMES,
                         top_terms=list(fx.vocab)[:25])
    run = Runner(Xtr, ytr, Xva, yva)

    # ================================================================ 3. gradient check
    log("3/11  gradient check")
    gc = {}
    for h_, l_, o_, k_ in [("tanh", "bce", "sigmoid", 1), ("relu", "bce", "sigmoid", 1), ("sigmoid", "mse", "sigmoid", 1),
                           ("leaky_relu", "mse", "linear", 1), ("relu", "cce", "softmax", 3)]:
        gc[f"{h_}+{l_}+{o_}"] = gradient_check(h_, l_, o_, k_)
    R["gradient_check"] = {k: float(f"{v:.3e}") for k, v in gc.items()}
    log(f"      max relative error = {max(gc.values()):.2e}  (< 1e-7 means back-prop is correct)")

    # ---------------- static concept figures
    P.fig_activations(FIG / "01_activations.png")
    P.fig_cost_functions(FIG / "02_cost_functions.png")
    P.fig_pipeline(FIG / "00_pipeline.png")

    # ================================================================ 4. perceptron
    log("4/11  perceptron")
    perc = Perceptron(lr=1.0, epochs=8 if Q else 15, seed=0).fit(Xtr, ytr)
    pp = perc.predict(Xte)
    R["perceptron"] = dict(test=metrics(yte, pp), errors_per_epoch=perc.errors_per_epoch)
    P.fig_perceptron_errors(FIG / "03_perceptron_errors.png", perc.errors_per_epoch)
    X4, y4, p4, m4 = P.xor_demo()
    R["xor"] = dict(perceptron_acc=f(np.mean(p4.predict(X4) == y4)), mlp_acc=f(np.mean(m4.predict(X4) == y4)))
    P.fig_xor(FIG / "04_xor_perceptron_vs_mlp.png", X4, y4, p4, m4)
    log(f"      perceptron test acc={R['perceptron']['test']['accuracy']}  F1={R['perceptron']['test']['f1']} | XOR perceptron={R['xor']['perceptron_acc']} MLP={R['xor']['mlp_acc']}")

    # ================================================================ 5. baseline MLP
    log("5/11  baseline MLP")
    BASE = dict(hidden=[32], act="relu", loss="bce", opt="adam", lr=1e-3, bs=64, epochs=20)
    mb, hb, fb = run.run(BASE)
    base_prob = mb.predict_proba(Xte)[:, 0]
    R["baseline"] = dict(config=BASE, params=mb.n_params(), val_f1=f(fb), test=metrics(yte, (base_prob >= .5).astype(int), base_prob))
    P.fig_network(FIG / "05_network.png", fx.n_features, BASE["hidden"])
    log(f"      baseline test acc={R['baseline']['test']['accuracy']}  F1={R['baseline']['test']['f1']}  params={mb.n_params():,}")

    # ================================================================ 6. cost functions
    log("6/11  cost functions: BCE vs MSE")
    ce = 8 if Q else 20
    gd_grid = [0.1, 0.2, 0.5, 1.0, 2.0] if not Q else [0.5, 1.0]
    setups = [("BCE + Adam", "bce", "adam", [1e-3], None), ("MSE + Adam", "mse", "adam", [1e-3], None),
              ("BCE + GD", "bce", "gd", gd_grid, None), ("MSE + GD", "mse", "gd", gd_grid, None),
              ("BCE + GD (confidently-wrong start)", "bce", "gd", [0.5], -8.0),    # same lr for both -> isolates the cost function
              ("MSE + GD (confidently-wrong start)", "mse", "gd", [0.5], -8.0)]
    cost_runs, cost_tab = {}, []
    for name, loss, opt, grid, ob in setups:
        best = None
        for lr in grid:                                             # each cost function gets its own best learning rate
            m, h, f1v = run.run(dict(BASE, loss=loss, opt=opt, lr=lr, epochs=ce, out_bias=ob))
            key = (f1v, -h["val_loss"][-1] if h["val_loss"] else 0)
            if best is None or key > best[0]:
                best = (key, lr, h, f1v)
        _, lr, h, f1v = best
        cost_runs[name] = h
        cost_tab.append(dict(setup=name, loss=loss, optimizer=opt, lr=lr, final_train_cost=f(h["train_loss"][-1], 5),
                             val_acc=f(h["val_acc"][-1]), val_f1=f(f1v), first_epoch_val_acc=f(h["val_acc"][0])))
        log(f"      {name:38s} lr={lr:<6} cost={h['train_loss'][-1]:.4f}  val_acc={h['val_acc'][-1]:.4f}  val_f1={f1v:.4f}")
    R["cost_compare"] = dict(table=cost_tab, curves={k: [f(x, 5) for x in v["train_loss"]] for k, v in cost_runs.items()})
    P.fig_cost_compare(FIG / "06_cost_bce_vs_mse.png", cost_runs)

    # ================================================================ 7. gradient-descent variants
    log("7/11  variants of gradient descent")
    ge = 6 if Q else 15
    n_tr = len(Xtr)
    variants = [  # label, optimizer, batch size, lr grid
        ("Batch GD", "gd", n_tr, [5, 20, 50, 100, 300]),
        ("Stochastic GD (bs=1)", "gd", 1, [0.002, 0.01, 0.05, 0.2]),
        ("Mini-batch GD (bs=64)", "gd", 64, [0.05, 0.2, 0.5, 1.0, 2.0]),
        ("Momentum", "momentum", 64, [0.02, 0.05, 0.1, 0.3, 1.0]),
        ("Nesterov (NAG)", "nesterov", 64, [0.02, 0.05, 0.1, 0.3, 1.0]),
        ("AdaGrad", "adagrad", 64, [0.01, 0.03, 0.1, 0.3, 1.0]),
        ("RMSProp", "rmsprop", 64, [3e-4, 1e-3, 3e-3, 1e-2]),
        ("Adam", "adam", 64, [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]),
    ]
    if Q:
        variants = [(a, b, c, d[:2]) for a, b, c, d in variants]
    gd_runs, gd_sum = {}, []
    for label, opt, bs, grid in variants:
        best = None
        for lr in grid:
            m, h, f1v = run.run(dict(BASE, opt=opt, bs=bs, lr=lr, epochs=ge))
            # an optimiser is judged on how well it MINIMISES the training cost in a fixed budget of epochs
            tc = h["train_loss"][-1] if not h["diverged"] and np.isfinite(h["train_loss"][-1]) else np.inf
            if best is None or tc < best[0]:
                best = (tc, lr, h, f1v)
        _, lr, h, f1v = best
        vl = h["val_loss"][-1] if h["val_loss"] else float("nan")
        gd_runs[label] = h
        gd_sum.append(dict(label=label, optimizer=opt, batch_size=int(bs), best_lr=lr, grid=grid, updates=h["updates"],
                           final_train_cost=f(h["train_loss"][-1], 5), val_loss=f(vl, 5), val_f1=f(f1v), time=f(h["time"], 2)))
        log(f"      {label:24s} lr={lr:<7} updates={h['updates']:<7} cost={h['train_loss'][-1]:.4f} val_f1={f1v:.4f} t={h['time']:.1f}s")
    R["gd_variants"] = dict(epochs=ge, table=gd_sum, curves={k: [f(x, 5) for x in v["train_loss"]] for k, v in gd_runs.items()})
    P.fig_gd_variants(FIG / "07_gd_variants.png", gd_runs, gd_sum)
    P.fig_gd_trajectories(FIG / "08_gd_trajectories.png")

    # ================================================================ 8. hyper-parameter tuning
    log("8/11  hyper-parameter tuning (one-at-a-time around the baseline)")
    seeds = 1 if Q else 2
    grids = dict(lr=[1e-4, 1e-3, 1e-2] if Q else [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
                 neurons=[8, 32, 128] if Q else [4, 8, 16, 32, 64, 128, 256],
                 layers=[1, 2, 3] if Q else [1, 2, 3, 4],
                 batch=[16, 64, 256] if Q else [8, 16, 32, 64, 128, 256, 512],
                 activation=["relu", "leaky_relu", "tanh", "sigmoid"],
                 epochs=[3, 10, 20] if Q else [2, 5, 10, 20, 40, 60])
    mk = {"lr": lambda v: dict(BASE, lr=v), "neurons": lambda v: dict(BASE, hidden=[v]),
          "layers": lambda v: dict(BASE, hidden=[32] * v), "batch": lambda v: dict(BASE, bs=v),
          "activation": lambda v: dict(BASE, act=v), "epochs": lambda v: dict(BASE, epochs=v)}
    H = {}
    for key, vals in grids.items():
        means, stds = [], []
        for v in vals:
            mu, sd = run.multi(mk[key](v), seeds)
            means.append(f(mu))
            stds.append(f(sd))
        H[key] = dict(values=vals, mean=means, std=stds)
        log(f"      {key:10s} " + "  ".join(f"{v}:{m:.3f}" for v, m in zip(vals, means)))
    P.fig_hyper(FIG / "09_hyperparameters.png", H)
    # epochs illustration: long run with validation cost
    _, h_long, _ = run.run(dict(BASE, epochs=30 if Q else 80))
    P.fig_epochs(FIG / "10_epochs_overfitting.png", h_long)
    R["hyperparameters"] = H
    pd.DataFrame([dict(hyperparameter=k, value=v, val_f1_mean=m, val_f1_std=s)
                  for k, d in H.items() for v, m, s in zip(d["values"], d["mean"], d["std"])]).to_csv(TAB / "hyperparameter_results.csv", index=False)

    best = {k: d["values"][int(np.argmax(d["mean"]))] for k, d in H.items()}
    R["hyperparameter_best"] = best
    cand = {"baseline": dict(BASE),
            "combined best": dict(BASE, lr=best["lr"], hidden=[best["neurons"]] * best["layers"], bs=best["batch"], act=best["activation"]),
            }
    # a mildly regularised, wider variant of the combination (guards against over-fitting)
    cand["combined + L2"] = dict(cand["combined best"], l2=1e-4)
    cand_res = {}
    for name, cfg in cand.items():
        c2 = dict(cfg, epochs=60 if not Q else 20, patience=8)
        mu, sd = run.multi(c2, seeds + 1)
        cand_res[name] = dict(config=cfg, val_f1_mean=f(mu), val_f1_std=f(sd))
        log(f"      candidate {name:16s} val F1 = {mu:.4f} +/- {sd:.4f}  {cfg['hidden']} lr={cfg['lr']} bs={cfg['bs']} {cfg['act']}")
    win = max(cand_res, key=lambda k: cand_res[k]["val_f1_mean"])
    FINAL = dict(cand_res[win]["config"], epochs=60 if not Q else 20, patience=8, track_grad=True)
    R["final_candidates"] = cand_res
    R["final_choice"] = dict(name=win, config=FINAL)
    log(f"      -> final configuration: {win}")

    # ================================================================ 9. final model
    log("9/11  final model, convergence, evaluation")
    runs_final = []
    for s in range(3 if Q else 5):
        m, h, f1v = run.run(FINAL, seed=s)
        runs_final.append((f1v, s, m, h))
    seed_scores = []
    for f1v, s, m, h in runs_final:
        pr = m.predict_proba(Xte)[:, 0]
        seed_scores.append(metrics(yte, (pr >= .5).astype(int), pr))
    deploy = max(runs_final, key=lambda r: r[0])                  # selected on VALIDATION F1 only
    _, dseed, model, hist = deploy
    prob = model.predict_proba(Xte)[:, 0]
    pval = model.predict_proba(Xva)[:, 0]
    thr_grid = np.round(np.arange(0.15, 0.86, 0.05), 2)
    val_f1_thr = [f1_score(yva, (pval >= t).astype(int), zero_division=0) for t in thr_grid]
    thr = float(thr_grid[int(np.argmax(val_f1_thr))])
    if max(val_f1_thr) - f1_score(yva, (pval >= 0.5).astype(int)) < 0.002:   # only move the threshold if it clearly helps
        thr = 0.5
    pred = (prob >= thr).astype(int)
    test_m = metrics(yte, pred, prob)
    test_05 = metrics(yte, (prob >= 0.5).astype(int), prob)
    model.save(MOD / "spam_model.npz")
    R["final"] = dict(config=FINAL, params=model.n_params(), deployed_seed=dseed, best_epoch=hist["best_epoch"],
                      stopped_epoch=hist["stopped_epoch"], epochs_run=len(hist["train_loss"]), updates=hist["updates"],
                      train_loss_first=f(hist["train_loss"][0], 5), train_loss_last=f(hist["train_loss"][-1], 5),
                      val_loss_best=f(min(hist["val_loss"]), 5), grad_norm_first=f(hist["grad_norm"][0], 5),
                      grad_norm_last=f(hist["grad_norm"][-1], 6), test=test_m, test_at_0_5=test_05, threshold=thr,
                      threshold_grid=thr_grid.tolist(), threshold_val_f1=[f(x) for x in val_f1_thr],
                      seed_mean={k: f(np.mean([s[k] for s in seed_scores])) for k in ["accuracy", "precision", "recall", "f1"]},
                      seed_std={k: f(np.std([s[k] for s in seed_scores])) for k in ["accuracy", "precision", "recall", "f1"]},
                      curves=dict(train_loss=[f(x, 5) for x in hist["train_loss"]], val_loss=[f(x, 5) for x in hist["val_loss"]],
                                  train_acc=[f(x) for x in hist["train_acc"]], val_acc=[f(x) for x in hist["val_acc"]]))
    P.fig_loss_curve(FIG / "11_final_loss_curve.png", hist)
    P.fig_confusion(FIG / "12_confusion_matrix_spam.png", test_m["confusion_matrix"], ["Genuine", "Spam"], f"Spam detector - test set (threshold {thr})")
    fpr, tpr, _ = roc_curve(yte, prob)
    bfpr, btpr, _ = roc_curve(yte, base_prob)
    P.fig_roc(FIG / "13_roc.png", {"Baseline MLP": (bfpr, btpr, R["baseline"]["test"]["auc"]), "Tuned MLP": (fpr, tpr, test_m["auc"])})
    ladder = [dict(name="Majority\nclass", **{k: v for k, v in metrics(yte, np.zeros_like(yte)).items() if k in ("accuracy", "precision", "recall", "f1")}),
              dict(name="Perceptron", **{k: R["perceptron"]["test"][k] for k in ("accuracy", "precision", "recall", "f1")}),
              dict(name="Baseline\nMLP", **{k: R["baseline"]["test"][k] for k in ("accuracy", "precision", "recall", "f1")}),
              dict(name="Tuned\nMLP", **{k: test_m[k] for k in ("accuracy", "precision", "recall", "f1")})]
    R["ladder"] = ladder
    P.fig_model_ladder(FIG / "14_model_ladder.png", ladder)
    P.fig_network(FIG / "05_network.png", fx.n_features, FINAL["hidden"])
    # error analysis
    te_ = te.reset_index(drop=True).assign(prob=prob, pred=pred)
    fn = te_[(te_.is_spam == 1) & (te_.pred == 0)].sort_values("prob").head(8)
    fp = te_[(te_.is_spam == 0) & (te_.pred == 1)].sort_values("prob", ascending=False).head(8)
    R["errors"] = dict(false_negatives=[dict(text=t, prob=f(p), type=k) for t, p, k in zip(fn.review_text, fn.prob, fn.spam_type)],
                       false_positives=[dict(text=t, prob=f(p)) for t, p in zip(fp.review_text, fp.prob)])
    if "spam_type" in te_.columns:
        spam_only = te_[te_.is_spam == 1].assign(ok=lambda d: d.pred == 1)
        R["recall_by_spam_type"] = {k: f(v) for k, v in spam_only.groupby("spam_type").ok.mean().items()}
    log(f"      TEST  acc={test_m['accuracy']}  prec={test_m['precision']}  rec={test_m['recall']}  F1={test_m['f1']}  AUC={test_m['auc']}")
    log(f"      decision threshold chosen on validation = {thr}  (test F1 at 0.5 = {test_05['f1']})")
    log(f"      {len(runs_final)}-seed mean F1 = {R['final']['seed_mean']['f1']} +/- {R['final']['seed_std']['f1']}")

    # ================================================================ 10. sentiment head
    log("10/11 sentiment head (softmax + categorical cross-entropy)")
    if df.rating.notna().any():
        lab = {"Negative": 0, "Neutral": 1, "Positive": 2}
        gtr, gva, gte = (d[(d.is_spam == 0) & d.sentiment.notna()] for d in (tr, va, te))
        Xs = [fx.transform(d.review_text) for d in (gtr, gva, gte)]
        Ys = [np.eye(3, dtype=np.float32)[d.sentiment.map(lab).values] for d in (gtr, gva, gte)]
        sm = MLP([fx.n_features] + list(FINAL["hidden"]) + [3], FINAL["act"], "softmax", "cce", l2=FINAL.get("l2", 0), seed=0)
        sh = sm.fit(Xs[0], Ys[0], Xs[1], Ys[1], epochs=60 if not Q else 20, batch_size=FINAL["bs"], optimizer="adam",
                    lr=FINAL["lr"], patience=8, track_grad=True)
        yt = Ys[2].argmax(1)
        sp = sm.predict(Xs[2])
        sm.save(MOD / "sentiment_model.npz")
        cm = confusion_matrix(yt, sp, labels=[0, 1, 2])
        R["sentiment"] = dict(accuracy=f(accuracy_score(yt, sp)), macro_f1=f(f1_score(yt, sp, average="macro")),
                              per_class_f1={n: f(v) for n, v in zip(["Negative", "Neutral", "Positive"], f1_score(yt, sp, average=None))},
                              confusion_matrix=cm.tolist(), best_epoch=sh["best_epoch"], train=len(gtr), test=len(gte),
                              loss_first=f(sh["train_loss"][0], 4), loss_last=f(sh["train_loss"][-1], 4))
        P.fig_confusion(FIG / "15_confusion_matrix_sentiment.png", cm, ["Neg", "Neu", "Pos"], "Sentiment head - test set", "Greens")
        log(f"      sentiment test acc={R['sentiment']['accuracy']}  macro-F1={R['sentiment']['macro_f1']}")
    else:
        log("      (no rating column -> sentiment head skipped)")

    # ================================================================ 11. dashboard + artefacts
    log("11/11 dashboard + artefacts")
    meta = dict(title="Indian E-commerce Review Intelligence", classes=["Genuine", "Spam"], sentiment_classes=["Negative", "Neutral", "Positive"],
                has_sentiment="sentiment" in R, hidden=FINAL["hidden"], activation=FINAL["act"], threshold=thr,
                test=dict(accuracy=test_m["accuracy"], precision=test_m["precision"], recall=test_m["recall"], f1=test_m["f1"], auc=test_m["auc"]),
                sentiment_test=R.get("sentiment", {}).get("accuracy"), data_source=src)
    json.dump(meta, open(MOD / "meta.json", "w"), indent=2)
    R["dashboard"] = build_dashboard(df, str(ROOT / "webapp/static/dash"), str(ROOT / "webapp/dashboard.json"))
    json.dump(R["dashboard"], open(TAB / "dashboard_stats.json", "w"), indent=2)
    pd.DataFrame(gd_sum).to_csv(TAB / "gd_variants.csv", index=False)
    pd.DataFrame(cost_tab).to_csv(TAB / "cost_functions.csv", index=False)
    for p in FIG.glob("*.png"):
        shutil.copy(p, ROOT / "webapp/static/model" / p.name)
    json.dump(meta, open(ROOT / "webapp/model_meta.json", "w"), indent=2)
    R["runtime_seconds"] = f(time.time() - T0, 1)
    json.dump(R, open(ROOT / "outputs" / "results.json", "w"), indent=2, default=str)
    log(f"DONE.  results -> outputs/results.json | figures -> outputs/figures | models -> models/")


if __name__ == "__main__":
    main()
