"""
evaluate.py — Accuracy comparison: Hybrid (content + collab) vs SVD
=====================================================================
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
MIN_HIGH_RATINGS  = 20    # minimum ≥4.0 ratings a user must have
RATING_THRESHOLD  = 4.0   # what counts as "high-rated"
N_SEEDS           = 10    # movies used to seed recommendations
N_HOLDOUT         = 10    # movies used as ground truth
N_RECS            = 10    # recommendations to generate
SVD_FACTORS       = 50    # latent dimensions for SVD model
RANDOM_SEED       = 42

np.random.seed(RANDOM_SEED)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
movies  = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
movies["genres"] = movies["genres"].fillna("")

# Lookup structures
mid_to_idx = {int(mid): i for i, mid in enumerate(movies["movieId"])}
idx_to_mid = {i: int(mid) for i, mid in enumerate(movies["movieId"])}

# ── 2. Build content similarity (TF-IDF on genres) ────────────────────────────
print("Building content similarity matrix...")
tfidf        = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
content_sim  = cosine_similarity(tfidf_matrix, tfidf_matrix)  # (n_movies × n_movies)

# ── 3. Build collaborative item matrix ────────────────────────────────────────
print("Building collaborative filtering structures...")
user_movie  = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
rated_ids   = user_movie.columns.tolist()               # movieIds present in ratings.csv
mid_to_col  = {int(mid): i for i, mid in enumerate(rated_ids)}

# Pre-normalise item matrix for fast cosine similarity via dot product
item_dense  = csr_matrix(user_movie.values).T.toarray()  # (n_rated_movies × n_users)
item_norm   = normalize(item_dense, norm="l2")            # L2-normalised rows
col_map     = np.array([mid_to_col.get(int(mid), -1) for mid in movies["movieId"]])

# ── 4. Build SVD model ────────────────────────────────────────────────────────
print(f"Fitting SVD model ({SVD_FACTORS} factors)...")
svd_model   = TruncatedSVD(n_components=SVD_FACTORS, random_state=RANDOM_SEED)
svd_model.fit(user_movie.values)
# item_factors: shape (n_rated_movies × n_factors)
item_factors_svd = svd_model.components_.T

# ── 5. Recommendation functions ───────────────────────────────────────────────

def recommend_hybrid(seed_mids, n=N_RECS):
    """Content (60%) + item-item collab cosine (40%), averaged over seeds."""
    exclude  = set(seed_mids)
    n_movies = len(movies)
    agg_content = np.zeros(n_movies)
    agg_collab  = np.zeros(n_movies)
    count = 0

    for mid in seed_mids:
        idx = mid_to_idx.get(mid)
        if idx is None:
            continue

        # Content similarity row
        cs = content_sim[idx].copy()

        # Collab similarity row (dot product of normalised item vectors)
        cf = np.zeros(n_movies)
        col = mid_to_col.get(mid)
        if col is not None:
            raw_sims = item_norm[col] @ item_norm.T   # (n_rated_movies,)
            valid    = col_map >= 0
            cf[valid] = raw_sims[col_map[valid]]

        if cs.max()  > 0: cs /= cs.max()
        if cf.max()  > 0: cf /= cf.max()

        agg_content += cs
        agg_collab  += cf
        count       += 1

    if count == 0:
        return []

    hybrid = 0.6 * agg_content + 0.4 * agg_collab

    # Zero-out seeds
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


def recommend_svd(seed_mids, seed_ratings_list, n=N_RECS):
    """Rating-weighted average of seed item vectors → dot product ranking."""
    exclude = set(seed_mids)

    weighted_sum  = np.zeros(SVD_FACTORS)
    total_weight  = 0.0

    for mid, r in zip(seed_mids, seed_ratings_list):
        col = mid_to_col.get(mid)
        if col is None:
            continue
        weighted_sum += r * item_factors_svd[col]
        total_weight += r

    if total_weight == 0:
        return []

    user_vec = weighted_sum / total_weight
    scores   = item_factors_svd @ user_vec          # (n_rated_movies,)

    # Build ranked list, exclude seeds
    ranked = sorted(
        [(int(rated_ids[i]), float(scores[i])) for i in range(len(rated_ids))
         if int(rated_ids[i]) not in exclude],
        key=lambda x: -x[1]
    )
    return [mid for mid, _ in ranked[:n]]


# ── 6. Identify qualifying users ──────────────────────────────────────────────
print("Identifying qualifying users...")
high = ratings[ratings["rating"] >= RATING_THRESHOLD]
counts = high.groupby("userId").size()
qualifying = counts[counts >= MIN_HIGH_RATINGS + N_HOLDOUT].index.tolist()
# (require MIN_HIGH_RATINGS + N_HOLDOUT so seeds + holdout are both full)
print(f"  Qualifying users: {len(qualifying)}")

# ── 7. Evaluation loop ────────────────────────────────────────────────────────
print("Running evaluation (this may take a minute)...")
scores_hybrid = []
scores_svd    = []
n_skipped     = 0

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

    # Skip if fewer than 3 seeds exist in movies.csv (not enough signal)
    if sum(1 for m in seeds if m in mid_to_idx) < 3:
        n_skipped += 1
        continue

    recs_h = recommend_hybrid(seeds, n=N_RECS)
    recs_s = recommend_svd(seeds, seed_ratings, n=N_RECS)

    scores_hybrid.append(len(set(recs_h) & holdout) / N_RECS)
    scores_svd.append(   len(set(recs_s) & holdout) / N_RECS)

scores_hybrid = np.array(scores_hybrid)
scores_svd    = np.array(scores_svd)

print(f"\n{'='*52}")
print(f"  Users evaluated : {len(scores_hybrid)}  (skipped: {n_skipped})")
print(f"{'='*52}")
print(f"  Hybrid  — mean: {scores_hybrid.mean():.4f}  std: {scores_hybrid.std():.4f}")
print(f"  SVD     — mean: {scores_svd.mean():.4f}  std: {scores_svd.std():.4f}")
wins_hybrid = (scores_hybrid > scores_svd).sum()
wins_svd    = (scores_svd > scores_hybrid).sum()
ties        = (scores_hybrid == scores_svd).sum()
print(f"\n  Hybrid wins : {wins_hybrid}  |  SVD wins : {wins_svd}  |  Ties : {ties}")
print(f"{'='*52}\n")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
print("Generating plots...")

COLORS = {"hybrid": "#e50914", "svd": "#3b82f6"}
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0d0d0d")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

def style_ax(ax, title):
    ax.set_facecolor("#141414")
    ax.tick_params(colors="#888", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")
    ax.set_title(title, color="#f0f0f0", fontsize=11, fontweight="bold", pad=10)
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")

# ── Plot 1: Mean accuracy bar chart with error bars ─────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Mean Accuracy @ 10")
means  = [scores_hybrid.mean(), scores_svd.mean()]
stds   = [scores_hybrid.std(),  scores_svd.std()]
labels = ["Hybrid\n(Content+Collab)", "SVD"]
bars   = ax1.bar(labels, means, color=[COLORS["hybrid"], COLORS["svd"]],
                 width=0.45, edgecolor="#2a2a2a", linewidth=0.8, zorder=3)
ax1.errorbar(labels, means, yerr=stds, fmt="none",
             color="#f0f0f0", capsize=6, linewidth=1.5, zorder=4)
for bar, mean in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, mean + max(stds)*0.15,
             f"{mean:.4f}", ha="center", va="bottom", color="#f0f0f0",
             fontsize=10, fontweight="bold")
ax1.set_ylabel("Accuracy (hits / 10)")
ax1.set_ylim(0, max(means) + max(stds) * 2.5)
ax1.grid(axis="y", color="#2a2a2a", linewidth=0.6, zorder=0)

# ── Plot 2: Score distributions (histogram) ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Score Distribution")
bins = np.arange(0, 1.05, 0.1) - 0.05
ax2.hist(scores_hybrid, bins=bins, alpha=0.7, color=COLORS["hybrid"],
         label="Hybrid", edgecolor="#0d0d0d", linewidth=0.5)
ax2.hist(scores_svd,    bins=bins, alpha=0.7, color=COLORS["svd"],
         label="SVD",    edgecolor="#0d0d0d", linewidth=0.5)
ax2.axvline(scores_hybrid.mean(), color=COLORS["hybrid"],
            linestyle="--", linewidth=1.5, alpha=0.9)
ax2.axvline(scores_svd.mean(),    color=COLORS["svd"],
            linestyle="--", linewidth=1.5, alpha=0.9)
ax2.set_xlabel("Score")
ax2.set_ylabel("Number of users")
legend = ax2.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a", labelcolor="#ccc", fontsize=9)

# ── Plot 3: Box plot ──────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Score Box Plot")
bp = ax3.boxplot(
    [scores_hybrid, scores_svd], labels=["Hybrid", "SVD"],
    patch_artist=True, notch=False, widths=0.4,
    medianprops=dict(color="#f0f0f0", linewidth=2),
    whiskerprops=dict(color="#888", linewidth=1.2),
    capprops=dict(color="#888", linewidth=1.2),
    flierprops=dict(marker="o", markerfacecolor="#555",
                    markersize=3, linestyle="none", markeredgecolor="none"),
)
for patch, color in zip(bp["boxes"], [COLORS["hybrid"], COLORS["svd"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
    patch.set_edgecolor("#2a2a2a")
ax3.set_ylabel("Score")
ax3.grid(axis="y", color="#2a2a2a", linewidth=0.6, zorder=0)

# ── Plot 4: Scatter — per-user Hybrid vs SVD ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "Per-user: Hybrid vs SVD")
jitter = np.random.uniform(-0.012, 0.012, size=len(scores_hybrid))
sc = ax4.scatter(scores_hybrid + jitter, scores_svd + jitter,
                 alpha=0.35, s=18, c="#9ca3af", edgecolors="none")
diag = np.linspace(0, 1, 100)
ax4.plot(diag, diag, color="#444", linewidth=1, linestyle="--", label="Equal")
ax4.set_xlabel("Hybrid score")
ax4.set_ylabel("SVD score")
ax4.set_xlim(-0.05, 1.05)
ax4.set_ylim(-0.05, 1.05)
ax4.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a", labelcolor="#ccc", fontsize=9)

# Annotate quadrants
ax4.text(0.75, 0.1, f"Hybrid wins\n({wins_hybrid})",
         color=COLORS["hybrid"], fontsize=8, ha="center", va="center", alpha=0.8)
ax4.text(0.1, 0.75, f"SVD wins\n({wins_svd})",
         color=COLORS["svd"], fontsize=8, ha="center", va="center", alpha=0.8)

# ── Plot 5: Win/loss/tie breakdown (pie) ─────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "Win / Tie / Loss Breakdown")
pie_data   = [wins_hybrid, wins_svd, ties]
pie_labels = [f"Hybrid wins\n({wins_hybrid})", f"SVD wins\n({wins_svd})", f"Tie\n({ties})"]
pie_colors = [COLORS["hybrid"], COLORS["svd"], "#555"]
wedges, texts, autotexts = ax5.pie(
    pie_data, labels=pie_labels, colors=pie_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": "#ccc", "fontsize": 9},
    wedgeprops={"edgecolor": "#0d0d0d", "linewidth": 1.5}
)
for at in autotexts:
    at.set_color("#f0f0f0")
    at.set_fontsize(9)

# ── Plot 6: Cumulative distribution (CDF) ────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, "Cumulative Distribution (CDF)")
for scores, label, color in [
    (scores_hybrid, "Hybrid", COLORS["hybrid"]),
    (scores_svd,    "SVD",    COLORS["svd"]),
]:
    xs = np.sort(scores)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax6.step(xs, ys, color=color, linewidth=2, label=label, where="post")
ax6.set_xlabel("Score")
ax6.set_ylabel("Cumulative proportion of users")
ax6.set_xlim(-0.02, 1.02)
ax6.set_ylim(0, 1.05)
ax6.grid(color="#2a2a2a", linewidth=0.6, zorder=0)
ax6.legend(facecolor="#1a1a1a", edgecolor="#2a2a2a", labelcolor="#ccc", fontsize=9)

# ── Title & save ──────────────────────────────────────────────────────────────
fig.suptitle(
    f"Recommendation Accuracy: Hybrid vs SVD  "
    f"({len(scores_hybrid)} users, {N_RECS} recommendations @ {N_HOLDOUT}-item holdout)",
    color="#f0f0f0", fontsize=13, fontweight="bold", y=0.98
)

output_path = "recommendation_accuracy.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"Plot saved to {output_path}")
