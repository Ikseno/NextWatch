# NextWatch — Hybrid Movie Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/TMDB-API-01B4E4?style=flat-square"/>
  <img src="https://img.shields.io/badge/Live-Demo-success?style=flat-square"/>
</p>

<p align="center">
  <b>Machine Learning Project · Hanyang University</b><br/>
  <a href="https://nextwa7ch.streamlit.app">🌐 Live Demo</a>
</p>

---

## Abstract

NextWatch is a content-collaborative hybrid recommender system for films and television series, built on the MovieLens dataset and deployed as a full-stack Streamlit application. The system combines TF-IDF genre similarity with item-item cosine collaborative filtering in a weighted blend optimised through Leave-N-Out cross-validation, achieving a Precision@10 of **0.1170** — outperforming both a pure SVD baseline (0.1107) and an equal-weight hybrid (0.1021).

---

## Table of Contents

- [System Overview](#system-overview)
- [Recommendation Algorithm](#recommendation-algorithm)
- [Evaluation](#evaluation)
- [Application Features](#application-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Limitations](#limitations)

---

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│         Streamlit Web App  ·  Android APK            │
└───────────────────┬─────────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │    Hybrid Recommender  │
        │   0.10 × TF-IDF cosine │
        │ + 0.90 × Item-Item cos │
        └───────────┬────────────┘
                    │
      ┌─────────────┴──────────────┐
      │                            │
┌─────▼──────┐             ┌───────▼──────┐
│  Content   │             │Collaborative │
│  TF-IDF    │             │  Item-Item   │
│  on Genres │             │  Cosine Sim  │
└────────────┘             └──────────────┘
      │                            │
      └──────────┬─────────────────┘
                 │
        ┌────────▼────────┐
        │  MovieLens Data │
        │  movies.csv     │
        │  ratings.csv    │
        └─────────────────┘
```

---

## Recommendation Algorithm

### 1. Content-Based Component — TF-IDF Cosine Similarity

Each film is represented as a genre string (e.g. `"Action|Adventure|Sci-Fi"`). A TF-IDF vectoriser transforms these strings into weighted term vectors, and pairwise cosine similarity is computed to form a pre-cached M×M similarity matrix.

```python
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies["genres"])
content_sim  = cosine_similarity(tfidf_matrix)   # shape: (M, M)
```

**Complexity:** O(M²) offline, O(1) lookup at query time.

### 2. Collaborative Filtering Component — Item-Item Cosine Similarity

Each film is represented as a vector of all user ratings it received. Vectors are L2-normalised and stored as a dense matrix. At query time, the dot product between a seed film's vector and all other film vectors yields cosine similarities directly.

```python
item_matrix = csr_matrix(user_movie.values).T.tocsr()
item_norm   = normalize(item_matrix.toarray(), norm="l2")  # shape: (M, U)

# At query time:
sims = item_norm[seed_col] @ item_norm.T   # O(U × M)
```

This captures *shared audience taste* without dimensionality reduction — two films rated similarly by the same users have high cosine similarity regardless of genre.

**Complexity:** O(U × M) per seed at query time.

### 3. Hybrid Blend

Content and collaborative scores are independently max-normalised to [0, 1], then linearly blended:

```
hybrid_score = 0.10 × content_score + 0.90 × collab_score
```

### 4. Multi-Seed Aggregation (`recommend_multi`)

When a user has rated multiple films, all rated titles serve as seeds. Scores are **max-pooled** across seeds rather than averaged, so a single strong match is sufficient to surface a recommendation. This preserves genre diversity across a user's taste profile.

```python
for seed in seeds:
    hybrid = 0.10 * content[seed] + 0.90 * collab[seed]
    scores = np.maximum(scores, hybrid)
```

---

## Evaluation

### Protocol — Leave-N-Out

- **Dataset:** 362 users with ≥ 20 ratings in the MovieLens corpus
- **Seeds:** Top-10 highest-rated films per user
- **Holdout:** Films ranked 11–20 (ground truth)
- **Metric:** Precision@10 — fraction of the 10 recommendations that appear in the holdout set

### Results

| Method | Content Weight | Collab Weight | Precision@10 |
|---|---|---|---|
| Hybrid (raw) | 0.60 | 0.40 | 0.1021 |
| Hybrid (raw) | 0.30 | 0.70 | 0.1089 |
| Hybrid (raw) | 0.20 | 0.80 | 0.1134 |
| **Hybrid (raw) ← deployed** | **0.10** | **0.90** | **0.1170** |
| Hybrid (SVD) | 0.50 | 0.50 | 0.1098 |
| Hybrid (SVD) | 0.10 | 0.90 | 0.1141 |
| Pure SVD baseline | — | — | 0.1107 |

The evaluation script reproducing these results is available in [`evaluate.py`](evaluate.py).

---

## Application Features

| Feature | Description |
|---|---|
| **Browse** | Trending wall for Movies and TV Shows, sourced from TMDB |
| **Search** | Real-time TMDB search with full detail dialog — synopsis, cast, trailer, streaming providers |
| **Find Similar** | One-click jump from any title to its recommendations |
| **Wishlist** | Persistent list of titles to watch later |
| **Watched** | Personal rating history (1–5 stars) |
| **For You** | Hybrid recommendations seeded from highest-rated films |
| **Import / Export** | CSV-based persistence for cross-device continuity |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML / Numerics | scikit-learn, NumPy, SciPy |
| Web Framework | Streamlit |
| External API | TMDB (The Movie Database) |
| Dataset | MovieLens (GroupLens Research) |
| Mobile | Android WebView (Kotlin) |

---

## Project Structure

```
nextwatch/
├── app.py            # Main Streamlit application (~1 500 lines)
├── evaluate.py       # Leave-N-Out evaluation harness & plots
├── movies.csv        # MovieLens movie metadata (movieId, title, genres)
├── ratings.csv       # MovieLens user ratings (userId, movieId, rating)
├── requirements.txt  # Python dependencies
└── .streamlit/
    └── secrets.toml  # TMDB_API_KEY — not committed
```

---

## Installation

### Prerequisites

- Python 3.10+
- A free TMDB API key → [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/nextwatch.git
cd nextwatch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure TMDB API key
mkdir -p .streamlit
echo 'TMDB_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# 4. Run
streamlit run app.py
```

### Run the evaluation

```bash
python evaluate.py
```

Outputs 6 comparative plots (accuracy bar, box plot, CDF, weight sensitivity, win rate vs SVD, mean delta vs SVD).

---

## Dataset

**MovieLens** — [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4):19:1–19:19.

The dataset covers user ratings up to **2017**. Films released after this date have no collaborative signal and rely exclusively on TMDB metadata for content-based similarity.

---

## Limitations

| Limitation | Description |
|---|---|
| **Cold-start** | New users with no rating history receive no personalised recommendations |
| **Pre-2018 data** | MovieLens ratings cap at 2017; newer films depend solely on TMDB metadata |
| **Genre coarseness** | TF-IDF on genre strings loses intra-genre nuance (e.g. tone, subgenre) |

### Future directions

- Extend the rating corpus with a more recent dataset
- Incorporate richer item features — plot embeddings, cast, director
- Replace item-item cosine with a learned model (matrix factorisation or neural collaborative filtering)

---

*Machine Learning Project · Hanyang University · 2026*
