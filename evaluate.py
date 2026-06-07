"""
evaluate.py — Accuracy comparison: Hybrid variants vs pure SVD
==============================================================
Algorithms evaluated:
  Raw-collab hybrids  (TF-IDF + item-item cosine):  0.6/0.4 · 0.3/0.7 · 0.2/0.8 · 0.1/0.9
  SVD-collab hybrids  (TF-IDF + SVD latent cosine): 0.5/0.5 · 0.1/0.9
  Baseline: pure SVD (rating-weighted latent vector ranking)

Methodology:
  - For each qualifying user (≥ 20 movies rated ≥ 4.0):
      • Seeds   = their top 10 highest-rated movies
      • Holdout = positions 11–20 of their top-rated list
      • Generate 10 recommendations from seeds using each algorithm
      • Score   = |recommendations ∩ holdout| / 10
  - Report mean accuracy and produce comparison plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

# ── Configuration ──────────────────────────────────────────────────────────────
MIN_HIGH_RATINGS  = 20
RATING_THRESHOLD  = 4.0
N_SEEDS           = 10
N_HOLDOUT         = 10
N_RECS            = 10
SVD_FACTORS       = 50
RANDOM_SEED       = 42

# Raw item-item collab hybrids: (content_w, collab_w)
RAW_CONFIGS = [
    (0.6, 0.4),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
]

# SVD latent-vector collab hybrids: (content_w, collab_w)
SVD_CONFIGS = [
    (0.5, 0.5),
    (0.1, 0.9),
]

# Color palette
COLORS = {
    # raw hybrids — warm
    ("raw", 0.6, 0.4): "#e50914",
    ("raw", 0.3, 0.7): "#f97316",
    ("raw", 0.2, 0.8): "#eab308",
    ("raw", 0.1, 0.9): "#22c55e",
    # SVD hybrids — cool
    ("svd", 0.5, 0.5): "#06b6d4",
    ("svd", 0.1, 0.9): "#a855f7",
    # pure SVD
    "svd_pure":         "#3b82f6",
}

np.random.seed(RANDOM_SEED)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
movies  = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
movies["genres"] = movies["genres"].fillna("")

mid_to_idx = {int(mid): i for i, mid in enumerate(movies["movieId"])}
idx_to_mid = {i: int(mid) for i, mid in enumerate(movies["movieId"])}

# ── 2. Build content similarity (TF-IDF on genres) ────────────────────────────
print("Building content similarity matrix...")
tfidf        = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
content_sim  = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ── 3. Build raw item-item collab matrix ──────────────────────────────────────
print("Building collaborative filtering structures...")
user_movie = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
rated_ids  = user_movie.columns.tolist()
mid_to_col = {int(mid): i for i, mid in enumerate(rated_ids)}

item_dense = csr_matrix(user_movie.values).T.toarray()   # (n_rated_movies × n_users)
item_norm  = normalize(item_dense, norm="l2")             # L2-normalised for raw cosine
col_map    = np.array([mid_to_col.get(int(mid), -1) for mid in movies["movieId"]])

# ── 4. Build SVD model ────────────────────────────────────────────────────────
print(f"Fitting SVD model ({SVD_FACTORS} factors)...")
svd_model        = TruncatedSVD(n_components=SVD_FACTORS, random_state=RANDOM_SEED)
svd_model.fit(user_movie.values)
item_factors_svd = svd_model.components_.T                # (n_rated_movies × n_factors)
item_factors_norm = normalize(item_factors_svd, norm="l2")  # L2-normalised for SVD cosine

# ── 5. Recommendation functions ───────────────────────────────────────────────

def recommend_hybrid_raw(seed_mids, content_w, collab_w, n=N_RECS):
    """TF-IDF content + raw item-item cosine collab, blended at content_w / collab_w."""
    exclude     = set(seed_mids)
    n_movies    = len(movies)
    agg_content = np.zeros(n_movies)
    agg_collab  = np.zeros(n_movies)
    count       = 0

    for mid in seed_mids:
        idx = mid_to_idx.get(mid)
        if idx is None:
            continue
        cs  = content_sim[idx].copy()
        cf  = np.zeros(n_movies)
        col = mid_to_col.get(mid)
        if col is not None:
            raw_sims  = item_norm[col] @ item_norm.T
            valid     = col_map >= 0
            cf[valid] = raw_sims[col_map[valid]]
        if cs.max() > 0: cs /= cs.max()
        if cf.max() > 0: cf /= cf.max()
        agg_content += cs
        agg_collab  += cf
        count       += 1

    if count == 0:
        return []

    hybrid = content_w * agg_content + collab_w * agg_collab
    for mid in exclude:
        idx = mid_to_idx.get(mid)
        if idx is not None:
            hybrid[idx] = -1.0

    top_idxs = np.argsort(hybrid)[::-1]
    results  = []
    for idx in top_idxs:
        mid = idx_to_mid.get(idx)
        if mid and mid not in exclude:
            results.append(mid)
        if len(results) == n:
            break
    return results


def recommend_hybrid_svd(seed_mids, content_w, collab_w, n=N_RECS):
    """TF-IDF content + SVD latent-vector cosine collab, blended at content_w / collab_w."""
    exclude     = set(seed_mids)
    n_movies    = len(movies)
    agg_content = np.zeros(n_movies)
    agg_collab  = np.zeros(n_movies)
    count       = 0

    for mid in seed_mids:
        idx = mid_to_idx.get(mid)
        if idx is None:
            continue
        cs  = content_sim[idx].copy()
        cf  = np.zeros(n_movies)
        col = mid_to_col.get(mid)
        if col is not None:
            svd_sims  = item_factors_norm[col] @ item_factors_norm.T
            valid     = col_map >= 0
            cf[valid] = svd_sims[col_map[valid]]
        if cs.max() > 0: cs /= cs.max()
        if cf.max() > 0: cf /= cf.max()
        agg_content += cs
        agg_collab  += cf
        count       += 1

    if count == 0:
        return []

    hybrid = content_w * agg_content + collab_w * agg_collab
    for mid in exclude:
        idx = mid_to_idx.get(mid)
        if idx is not None:
            hybrid[idx] = -1.0

    top_idxs = np.argsort(hybrid)[::-1]
    results  = []
    for idx in top_idxs:
        mid = idx_to_mid.get(idx)
        if mid and mid not in exclude:
            results.append(mid)
        if len(results) == n:
            break
    return results


def recommend_svd_pure(seed_mids, seed_ratings_list, n=N_RECS):
    """Rating-weighted average of seed latent vectors → dot product ranking."""
    exclude      = set(seed_mids)
    weighted_sum = np.zeros(SVD_FACTORS)
    total_weight = 0.0
    for mid, r in zip(seed_mids, seed_ratings_list):
        col = mid_to_col.get(mid)
        if col is None:
            continue
        weighted_sum += r * item_factors_svd[col]
        total_weight += r
    if total_weight == 0:
        return []
    user_vec = weighted_sum / total_weight
    scores   = item_factors_svd @ user_vec
    ranked   = sorted(
        [(int(rated_ids[i]), float(scores[i])) for i in range(len(rated_ids))
         if int(rated_ids[i]) not in exclude],
        key=lambda x: -x[1]
    )
    return [mid for mid, _ in ranked[:n]]


# ── 6. Identify qualifying users ──────────────────────────────────────────────
print("Identifying qualifying users...")
high       = ratings[ratings["rating"] >= RATING_THRESHOLD]
counts     = high.groupby("userId").size()
qualifying = counts[counts >= MIN_HIGH_RATINGS + N_HOLDOUT].index.tolist()
print(f"  Qualifying users: {len(qualifying)}")

# ── 7. Evaluation loop ────────────────────────────────────────────────────────
print("Running evaluation (this may take a minute)...")

scores_raw  = {cfg: [] for cfg in RAW_CONFIGS}
scores_svdh = {cfg: [] for cfg in SVD_CONFIGS}
scores_svd  = []
n_skipped   = 0

for i, uid in enumerate(qualifying):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(qualifying)} users...")

    user_high = (
        high[high["userId"] == uid]
        .sort_values(["rating", "movieId"], ascending=[False, True])
        .head(N_SEEDS + N_HOLDOUT)
    )
    if len(user_high) < N_SEEDS + N_HOLDOUT:
        n_skipped += 1
        continue

    top_mids     = user_high["movieId"].tolist()
    top_ratings  = user_high["rating"].tolist()
    seeds        = top_mids[:N_SEEDS]
    seed_ratings = top_ratings[:N_SEEDS]
    holdout      = set(top_mids[N_SEEDS:N_SEEDS + N_HOLDOUT])

    if sum(1 for m in seeds if m in mid_to_idx) < 3:
        n_skipped += 1
        continue

    for cfg in RAW_CONFIGS:
        cw, lw = cfg
        recs = recommend_hybrid_raw(seeds, cw, lw)
        scores_raw[cfg].append(len(set(recs) & holdout) / N_RECS)

    for cfg in SVD_CONFIGS:
        cw, lw = cfg
        recs = recommend_hybrid_svd(seeds, cw, lw)
        scores_svdh[cfg].append(len(set(recs) & holdout) / N_RECS)

    recs_s = recommend_svd_pure(seeds, seed_ratings)
    scores_svd.append(len(set(recs_s) & holdout) / N_RECS)

scores_raw  = {cfg: np.array(v) for cfg, v in scores_raw.items()}
scores_svdh = {cfg: np.array(v) for cfg, v in scores_svdh.items()}
scores_svd  = np.array(scores_svd)
n_users     = len(scores_svd)

# ── 8. Console summary ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Users evaluated : {n_users}  (skipped: {n_skipped})")
print(f"{'='*65}")
print("  RAW COLLAB HYBRIDS (TF-IDF + item-item cosine):")
for cfg in RAW_CONFIGS:
    cw, lw = cfg
    s = scores_raw[cfg]
    print(f"    {cw:.1f}/{lw:.1f}   mean: {s.mean():.4f}  std: {s.std():.4f}  "
          f"SVD-beats-hybrid: {(scores_svd > s).sum()}/{n_users}")
print("  SVD COLLAB HYBRIDS (TF-IDF + SVD latent cosine):")
for cfg in SVD_CONFIGS:
    cw, lw = cfg
    s = scores_svdh[cfg]
    print(f"    {cw:.1f}/{lw:.1f}   mean: {s.mean():.4f}  std: {s.std():.4f}  "
          f"SVD-beats-hybrid: {(scores_svd > s).sum()}/{n_users}")
print(f"  PURE SVD         mean: {scores_svd.mean():.4f}  std: {scores_svd.std():.4f}")
print(f"{'='*65}\n")

# ── 9. Plots ──────────────────────────────────────────────────────────────────
print("Generating plots...")

# Build ordered list of all algorithms for shared plots
all_entries = (
    [(("raw", cw, lw), f"{cw:.1f}/{lw:.1f}\n(raw)",  scores_raw[  (cw, lw)]) for cw, lw in RAW_CONFIGS] +
    [(("svd", cw, lw), f"{cw:.1f}/{lw:.1f}\n(SVD)",  scores_svdh[(cw, lw)]) for cw, lw in SVD_CONFIGS] +
    [("svd_pure",       "SVD\n(pure)",                scores_svd)]
)
all_keys    = [e[0] for e in all_entries]
all_labels  = [e[1] for e in all_entries]
all_scores  = [e[2] for e in all_entries]
all_colors  = [COLORS[k] for k in all_keys]
all_means   = [s.mean() for s in all_scores]
all_stds    = [s.std()  for s in all_scores]

fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor("#0d0d0d")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.36)

def style_ax(ax, title):
    ax.set_facecolor("#141414")
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
    ax.set_title(title, color="#f0f0f0", fontsize=10.5, fontweight="bold", pad=10)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")

# ── Plot 1: Mean accuracy bar chart ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Mean Accuracy @ 10")
x    = np.arange(len(all_labels))
bars = ax1.bar(x, all_means, color=all_colors, width=0.55,
               edgecolor="#2a2a2a", linewidth=0.8, zorder=3)
ax1.errorbar(x, all_means, yerr=all_stds, fmt="none",
             color="#f0f0f0", capsize=4, linewidth=1.2, zorder=4)
for bar, mean in zip(bars, all_means):
    ax1.text(bar.get_x() + bar.get_width()/2,
             mean + max(all_stds) * 0.18,
             f"{mean:.4f}", ha="center", va="bottom",
             color="#f0f0f0", fontsize=7.5, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(all_labels, fontsize=7.5)
ax1.set_ylabel("Accuracy (hits / 10)")
ax1.set_ylim(0, max(all_means) + max(all_stds) * 3.2)
ax1.grid(axis="y", color="#2a2a2a", linewidth=0.6, zorder=0)

# ── Plot 2: Box plot ──────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Score Distribution (Box Plot)")
bp = ax2.boxplot(
    all_scores, labels=all_labels,
    patch_artist=True, notch=False, widths=0.42,
    medianprops=dict(color="#f0f0f0", linewidth=2),
    whiskerprops=dict(color="#888", linewidth=1.1),
    capprops=dict(color="#888", linewidth=1.1),
    flierprops=dict(marker="o", markerfacecolor="#555",
                    markersize=3, linestyle="none", markeredgecolor="none"),
)
for patch, color in zip(bp["boxes"], all_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
    patch.set_edgecolor("#2a2a2a")
ax2.tick_params(axis="x", labelsize=7.5)
ax2.set_ylabel("Score")
ax2.grid(axis="y", color="#2a2a2a", linewidth=0.6, zorder=0)

# ── Plot 3: CDF ───────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Cumulative Distribution (CDF)")
for scores, label, color in zip(all_scores, all_labels, all_colors):
    xs = np.sort(scores)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax3.step(xs, ys, color=color, linewidth=2,
             label=label.replace("\n", " "), where="post")
ax3.set_xlabel("Score")
ax3.set_ylabel("Cumulative proportion of users")
ax3.set_xlim(-0.02, 1.02)
ax3.set_ylim(0, 1.05)
ax3.grid(color="#2a2a2a", linewidth=0.6, zorder=0)
ax3.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a",
           labelcolor="#ccc", fontsize=7.5, loc="lower right")

# ── Plot 4: Weight sensitivity (two lines: raw vs SVD collab) ─────────────────
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "Accuracy vs Content Weight")

svd_pure_mean = scores_svd.mean()
svd_pure_std  = scores_svd.std()

# Raw collab line: x = [0.6, 0.3, 0.2, 0.1, 0.0]
raw_x     = [cw for cw, _ in RAW_CONFIGS] + [0.0]
raw_means = [scores_raw[cfg].mean() for cfg in RAW_CONFIGS] + [svd_pure_mean]
raw_cols  = [COLORS[("raw", cw, lw)] for cw, lw in RAW_CONFIGS] + [COLORS["svd_pure"]]

ax4.plot(raw_x, raw_means, color="#888", linewidth=1.5,
         linestyle="--", zorder=2, label="Raw collab")
for xv, yv, c in zip(raw_x, raw_means, raw_cols):
    ax4.scatter([xv], [yv], color=c, s=75, zorder=5)

# SVD collab line: x = [0.5, 0.1, 0.0]
svd_x     = [cw for cw, _ in SVD_CONFIGS] + [0.0]
svd_means = [scores_svdh[cfg].mean() for cfg in SVD_CONFIGS] + [svd_pure_mean]
svd_cols  = [COLORS[("svd", cw, lw)] for cw, lw in SVD_CONFIGS] + [COLORS["svd_pure"]]

ax4.plot(svd_x, svd_means, color="#555", linewidth=1.5,
         linestyle="-", zorder=2, label="SVD collab")
for xv, yv, c in zip(svd_x, svd_means, svd_cols):
    ax4.scatter([xv], [yv], color=c, s=75, zorder=5)

ax4.set_xlabel("Content weight  (collab = 1 − content)")
ax4.set_ylabel("Mean accuracy")
ax4.grid(color="#2a2a2a", linewidth=0.6, zorder=0)
ax4.text(0.0, svd_pure_mean + svd_pure_std * 0.25,
         "pure SVD", ha="center", color=COLORS["svd_pure"], fontsize=8)
ax4.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a", labelcolor="#ccc", fontsize=8)

# ── Plot 5: Win rate vs pure SVD ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "Win Rate vs Pure SVD (%)")

hybrid_entries = (
    [(f"{cw:.1f}/{lw:.1f}\n(raw)", scores_raw[  (cw, lw)], COLORS[("raw", cw, lw)])
     for cw, lw in RAW_CONFIGS] +
    [(f"{cw:.1f}/{lw:.1f}\n(SVD)", scores_svdh[(cw, lw)], COLORS[("svd", cw, lw)])
     for cw, lw in SVD_CONFIGS]
)
h_labels = [e[0] for e in hybrid_entries]
h_scores = [e[1] for e in hybrid_entries]
h_colors = [e[2] for e in hybrid_entries]

win_pcts  = [(s > scores_svd).mean() * 100 for s in h_scores]

tie_pcts  = [(s == scores_svd).mean() * 100 for s in h_scores]
loss_pcts = [(s < scores_svd).mean() * 100 for s in h_scores]

xb    = np.arange(len(hybrid_entries))
bw    = 0.26
ax5.bar(xb - bw, win_pcts,  width=bw, color="#22c55e", alpha=0.85,
        edgecolor="#2a2a2a", linewidth=0.7, label="Hybrid wins", zorder=3)
ax5.bar(xb,      tie_pcts,  width=bw, color="#555",    alpha=0.85,
        edgecolor="#2a2a2a", linewidth=0.7, label="Tie",           zorder=3)
ax5.bar(xb + bw, loss_pcts, width=bw, color="#e50914", alpha=0.85,
        edgecolor="#2a2a2a", linewidth=0.7, label="SVD wins",      zorder=3)
ax5.set_xticks(xb)
ax5.set_xticklabels(h_labels, fontsize=7.5)
ax5.set_ylabel("% of users")
ax5.set_ylim(0, 108)
ax5.grid(axis="y", color="#2a2a2a", linewidth=0.6, zorder=0)
ax5.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a",
           labelcolor="#ccc", fontsize=8)

# ── Plot 6: Mean delta vs pure SVD ────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, "Mean Score Delta vs Pure SVD")
deltas = [(s - scores_svd).mean() for s in h_scores]
bars6  = ax6.bar(xb, deltas, color=h_colors, width=0.5,
                 edgecolor="#2a2a2a", linewidth=0.8, zorder=3)
ax6.axhline(0, color="#888", linewidth=1.2, linestyle="--", zorder=2)
for bar, delta in zip(bars6, deltas):
    yoff = 0.0005 if delta >= 0 else -0.0015
    va   = "bottom" if delta >= 0 else "top"
    ax6.text(bar.get_x() + bar.get_width()/2, delta + yoff,
             f"{delta:+.4f}", ha="center", va=va,
             color="#f0f0f0", fontsize=8, fontweight="bold")
ax6.set_xticks(xb)
ax6.set_xticklabels(h_labels, fontsize=7.5)
ax6.set_ylabel("Mean accuracy - SVD mean")
ax6.grid(axis="y", color="#2a2a2a", linewidth=0.6, zorder=0)
ax6.text(len(hybrid_entries) - 0.5, 0.0002,
         "SVD baseline", color="#888", fontsize=8)

# ── Title and save ────────────────────────────────────────────────────────────
fig.suptitle(
    f"Recommendation Accuracy: Raw-Collab Hybrids vs SVD-Collab Hybrids vs Pure SVD  "
    f"({n_users} users, {N_RECS} recs @ {N_HOLDOUT}-item holdout)",
    color="#f0f0f0", fontsize=12, fontweight="bold", y=0.98
)

output_path = "recommendation_accuracy.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"Plot saved to {output_path}")
