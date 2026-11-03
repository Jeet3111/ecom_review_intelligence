"""dashboard.py - builds the 'Review Intelligence' dashboard (PNG charts + JSON KPIs) like the reference screenshot."""
import json
import os
import textwrap
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import clean_text, tokens
from .plots import BLUE, GREEN, ORANGE, RED, GREY, PURPLE
from .topics import TOPICS, detect_topics

STOP = set("the a an and or but is are was were it this that i my me we you to of for in on at with so very too have has had be been as by from "
           "they them their its not no just also all one get got will would can could than then if do does did product urltoken phonetoken "
           "handletoken buy bought items item please really".split())


def _save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=8)


def build_dashboard(df, out_dir, json_path):
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    gen = df[df.is_spam == 0].copy()
    has_rating = df["rating"].notna().any()
    has_date = df["review_date"].notna().any()
    kpi = dict(products=int(df.product_id.nunique()), categories=int(df.category.nunique()), reviews=int(len(df)),
               spam_pct=round(100 * float(df.is_spam.mean()), 1), genuine=int(len(gen)),
               avg_rating=round(float(gen.rating.mean()), 2) if has_rating else None)
    out = dict(kpi=kpi, has_rating=bool(has_rating), has_date=bool(has_date))

    # 1) review trend over time
    if has_date:
        d = gen.assign(m=pd.to_datetime(gen.review_date).dt.to_period("M").dt.to_timestamp())
        g = d.groupby("m").agg(n=("review_text", "size"), r=("rating", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        ax.plot(g.m, g.n, color=GREEN, lw=2.2, marker="o", ms=4)
        ax.fill_between(g.m, g.n, alpha=0.08, color=GREEN)
        ax.set_ylabel("genuine reviews / month", fontsize=8)
        _style(ax)
        if has_rating:
            ax2 = ax.twinx()
            ax2.plot(g.m, g.r, color=ORANGE, lw=1.8, ls="--")
            ax2.set_ylim(1, 5)
            ax2.set_ylabel("avg rating", fontsize=8, color=ORANGE)
            ax2.tick_params(labelsize=8)
            ax2.spines[["top"]].set_visible(False)
        ax.tick_params(axis="x", rotation=30)
        _save(fig, f"{out_dir}/trend.png")

    # 2) sentiment pie
    if has_rating:
        sc = gen.sentiment.value_counts()
        out["sentiment"] = {k: int(v) for k, v in sc.items()}
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        order = [k for k in ["Positive", "Neutral", "Negative"] if k in sc.index]
        cols = {"Positive": GREEN, "Neutral": ORANGE, "Negative": RED}
        ax.pie([sc[k] for k in order], labels=order, colors=[cols[k] for k in order], autopct="%1.0f%%", startangle=90,
               wedgeprops=dict(width=0.45, edgecolor="white"), textprops=dict(fontsize=9))
        _save(fig, f"{out_dir}/sentiment.png")

        # 3) rating distribution
        rc = gen.rating.value_counts(normalize=True).reindex([5, 4, 3, 2, 1]).fillna(0)
        out["rating_dist"] = {str(k): round(float(v), 4) for k, v in rc.items()}
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.barh([f"{k} star" for k in rc.index], rc.values * 100, color=[GREEN, "#84CC16", ORANGE, "#FB923C", RED])
        for i, v in enumerate(rc.values * 100):
            ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% of genuine reviews", fontsize=8)
        _style(ax)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0)
        _save(fig, f"{out_dir}/rating_dist.png")

        # 4) topics: positive / negative
        tp, tn = Counter(), Counter()
        for t, s in zip(gen.review_text, gen.sentiment):
            for k in detect_topics(t):
                if s == "Positive":
                    tp[k] += 1
                elif s == "Negative":
                    tn[k] += 1
        out["top_pos_topics"] = tp.most_common(6)
        out["top_neg_topics"] = tn.most_common(6)
        for name, cnt, col in [("topics_pos", tp, BLUE), ("topics_neg", tn, RED)]:
            items = cnt.most_common(5)
            fig, ax = plt.subplots(figsize=(5.0, 3.0))
            ax.bar([textwrap.fill(k, 11) for k, _ in items], [v for _, v in items], color=col, width=0.55)
            _style(ax)
            ax.tick_params(axis="x", labelsize=6.5)
            _save(fig, f"{out_dir}/{name}.png")

        # 5) price vs rating (product level)
        pr = gen.groupby("product_id").agg(price=("price_inr", "first"), rating=("rating", "mean"), n=("rating", "size"),
                                           cat=("category", "first")).dropna()
        pr = pr[pr.n >= 5]
        if len(pr):
            fig, ax = plt.subplots(figsize=(5.0, 3.0))
            c = np.where(pr.rating >= 3.8, GREEN, np.where(pr.rating >= 3.2, ORANGE, RED))
            ax.scatter(pr.price, pr.rating, c=c, s=26, alpha=0.85, edgecolor="white", lw=0.4)
            ax.set_xscale("log")
            ax.set_xlabel("selling price (INR, log scale)", fontsize=8)
            ax.set_ylabel("avg rating", fontsize=8)
            _style(ax)
            _save(fig, f"{out_dir}/price_rating.png")

        # keywords
        pos_c, neg_c = Counter(), Counter()
        for t, s in zip(gen.review_text, gen.sentiment):
            toks = {w for w in tokens(clean_text(t)) if w not in STOP and len(w) > 2}
            (pos_c if s == "Positive" else neg_c if s == "Negative" else Counter()).update(toks)
        npos, nneg = max(sum(gen.sentiment == "Positive"), 1), max(sum(gen.sentiment == "Negative"), 1)
        sc_pos = {w: c * np.log(((c + 1) / npos) / ((neg_c[w] + 1) / nneg)) for w, c in pos_c.items() if c >= 15}
        sc_neg = {w: c * np.log(((c + 1) / nneg) / ((pos_c[w] + 1) / npos)) for w, c in neg_c.items() if c >= 15}
        out["kw_pos"] = [w for w, _ in sorted(sc_pos.items(), key=lambda x: -x[1])[:12]]
        out["kw_neg"] = [w for w, _ in sorted(sc_neg.items(), key=lambda x: -x[1])[:12]]

    # 6) spam by category
    sp = df.groupby("category").is_spam.mean().sort_values() * 100
    out["spam_by_category"] = {k: round(float(v), 1) for k, v in sp.items()}
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.barh(sp.index, sp.values, color=PURPLE, height=0.55)
    for i, v in enumerate(sp.values):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("% reviews flagged spam / fake", fontsize=8)
    _style(ax)
    ax.grid(axis="y", alpha=0)
    _save(fig, f"{out_dir}/spam_category.png")

    # 7) spam types (only when available - synthetic data)
    if "spam_type" in df.columns and (df.spam_type != "genuine").any():
        st = df[df.is_spam == 1].spam_type.value_counts()
        st = st[st.index != "genuine"]
        out["spam_types"] = {k: int(v) for k, v in st.items()}
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.barh([k.replace("_", " ") for k in st.index], st.values, color=GREY, height=0.55)
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()
        _style(ax)
        ax.grid(axis="y", alpha=0)
        _save(fig, f"{out_dir}/spam_types.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
