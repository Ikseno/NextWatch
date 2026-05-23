import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

st.set_page_config(
    page_title="NextWatch", page_icon="🎬", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family:'Inter',sans-serif!important; box-sizing:border-box; }

.stApp { background:#080808!important; color:#f0f0f0; }
#MainMenu,footer,.stDeployButton,[data-testid="stToolbar"]{visibility:hidden!important;display:none!important;}
[data-testid="stAppViewContainer"]{background:#080808;}
[data-testid="block-container"]{padding-top:0!important;}


/* Sidebar */
[data-testid="stSidebar"]{background:#0f0f0f!important;border-right:1px solid #1a1a1a!important;}
[data-testid="stSidebar"]>div{padding-top:1.2rem;}

/* Slider */
[data-testid="stSlider"] [role="slider"]{background:#e50914!important;border-color:#e50914!important;}
[data-testid="stSlider"] [data-testid="stSliderTrack"]>div:first-child{background:#e50914!important;}

/* Multiselect */
[data-baseweb="tag"]{background:rgba(229,9,20,.85)!important;}

/* Selectbox trigger */
[data-testid="stSelectbox"] [data-baseweb="select"]>div:first-child{
    background:#141414!important;border:1px solid #2a2a2a!important;
    border-radius:10px!important;color:#f0f0f0!important;font-size:1rem!important;min-height:46px!important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]>div:first-child:focus-within{
    border-color:#e50914!important;box-shadow:0 0 0 3px rgba(229,9,20,.15)!important;
}
[data-testid="stSelectbox"] input{background:transparent!important;color:#f0f0f0!important;}
[data-testid="stSelectbox"] [data-baseweb="select"] span{color:#555!important;}

/* Primary button */
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#e50914,#b00710)!important;border:none!important;
    color:#fff!important;font-weight:700!important;padding:.6rem 2rem!important;
    border-radius:8px!important;font-size:.9rem!important;letter-spacing:.04em!important;
    text-transform:uppercase!important;transition:all .25s!important;
}
.stButton>button[kind="primary"]:hover:not(:disabled){transform:translateY(-2px)!important;box-shadow:0 10px 25px rgba(229,9,20,.35)!important;}
.stButton>button[kind="primary"]:disabled{background:#1e1e1e!important;color:#444!important;}

/* Secondary / ghost buttons */
.stButton>button:not([kind="primary"]){
    background:transparent!important;border:1px solid #252525!important;color:#666!important;
    font-size:.7rem!important;padding:.2rem .55rem!important;font-weight:500!important;
    text-transform:none!important;letter-spacing:0!important;min-height:28px!important;
    border-radius:6px!important;transition:all .2s!important;
}
.stButton>button:not([kind="primary"]):hover:not(:disabled){border-color:#e50914!important;color:#e50914!important;}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:transparent;gap:4px;border-bottom:1px solid #1a1a1a!important;}
[data-testid="stTabs"] [data-baseweb="tab"]{background:#111;border-radius:8px 8px 0 0;color:#555;padding:8px 20px;border:1px solid #1e1e1e;border-bottom:none;}
[data-testid="stTabs"] [aria-selected="true"]{background:#e50914!important;color:#fff!important;border-color:#e50914!important;}
[data-testid="stTabs"] [data-baseweb="tab-border"],[data-testid="stTabs"] [data-baseweb="tab-highlight"]{display:none!important;}

/* Expander */
[data-testid="stExpander"]{background:#111!important;border:1px solid #1e1e1e!important;border-radius:8px!important;margin-top:3px!important;}
[data-testid="stExpander"] summary p{color:#555!important;font-size:.73rem!important;}

/* Alerts */
[data-testid="stInfo"]{background:#111!important;border-left:3px solid #e50914!important;color:#888!important;}

/* Scrollbar */
::-webkit-scrollbar{width:5px;background:#0f0f0f;}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#e50914;}

/* ── Hero ── */
.hero{position:relative;border-bottom:1px solid #1a1a1a;padding:4.5rem 2rem 3rem;text-align:center;margin-bottom:1.5rem;overflow:hidden;}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(229,9,20,.13) 0%,transparent 70%);pointer-events:none;}
.hero::after{content:'';position:absolute;inset:0;background:linear-gradient(180deg,transparent 60%,#080808 100%);pointer-events:none;}
.hero-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(229,9,20,.1);border:1px solid rgba(229,9,20,.25);color:#e50914;padding:5px 16px;border-radius:20px;font-size:.7rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:1.4rem;}
.hero-title{font-size:clamp(4rem,10vw,7.5rem);font-weight:900;line-height:.95;margin:0 0 .7rem;letter-spacing:-.04em;background:linear-gradient(160deg,#fff 40%,#555 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-accent{-webkit-text-fill-color:transparent;background:linear-gradient(135deg,#ff1a1a,#e50914);-webkit-background-clip:text;background-clip:text;filter:drop-shadow(0 0 40px rgba(229,9,20,.5));}
.hero-sub{color:#4a4a4a;font-size:.98rem;margin:0 auto 2rem;max-width:400px;line-height:1.6;}
.hero-pills{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-bottom:2.2rem;}
.hero-pill{background:rgba(255,255,255,.03);border:1px solid #1e1e1e;color:#4a4a4a;padding:6px 14px;border-radius:20px;font-size:.73rem;font-weight:500;letter-spacing:.02em;}
.hero-stats{display:flex;gap:0;justify-content:center;border-top:1px solid #111;padding-top:1.8rem;position:relative;z-index:1;}
.hero-stat{display:flex;flex-direction:column;align-items:center;gap:4px;padding:0 2.5rem;border-right:1px solid #111;}
.hero-stat:last-child{border-right:none;}
.hero-stat-n{font-size:1.6rem;font-weight:900;color:#f0f0f0;letter-spacing:-.03em;line-height:1;}
.hero-stat-l{font-size:.62rem;color:#383838;text-transform:uppercase;letter-spacing:.14em;}


/* ── Section header ── */
.sec-hdr{display:flex;align-items:center;gap:10px;margin:0 0 1.2rem;}
.sec-hdr-title{font-size:1.1rem;font-weight:700;color:#f0f0f0;white-space:nowrap;}
.sec-hdr-line{flex:1;height:1px;background:linear-gradient(90deg,#222,transparent);}
.sec-hdr-count{font-size:.7rem;color:#444;background:#111;border:1px solid #1e1e1e;padding:3px 10px;border-radius:12px;}

/* ── Card elements ── */
.no-poster{width:100%;aspect-ratio:2/3;background:linear-gradient(160deg,#181818,#0d0d0d);display:flex;flex-direction:column;align-items:center;justify-content:center;color:#2a2a2a;font-size:2.2rem;}
.card-body{padding:.65rem .7rem .55rem;}
.card-title{font-size:.8rem;font-weight:700;color:#f0f0f0;margin:0 0 .38rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.genre-row{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:.42rem;}
.gtag{background:#1a1a1a;color:#555;padding:2px 6px;border-radius:4px;font-size:.58rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;border:1px solid #222;}
.sim-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:20px;font-size:.67rem;font-weight:700;border:1px solid rgba(229,9,20,.25);background:rgba(229,9,20,.07);color:#ff4d4d;}
.sim-dot{width:5px;height:5px;background:#e50914;border-radius:50%;display:inline-block;}
.tmdb-badge{display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:20px;font-size:.67rem;font-weight:700;border:1px solid rgba(250,204,21,.2);background:rgba(250,204,21,.06);color:#facc15;margin-left:4px;}
.card-badges{display:flex;align-items:center;flex-wrap:wrap;gap:4px;}

/* ── List row ── */
.list-row{display:flex;align-items:center;gap:12px;background:#111;border:1px solid #1e1e1e;border-radius:10px;padding:10px 12px;margin-bottom:6px;}
.list-thumb{width:40px;height:60px;object-fit:cover;border-radius:5px;flex-shrink:0;}
.list-thumb-ph{width:40px;height:60px;background:#1a1a1a;border-radius:5px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.9rem;color:#333;}
.list-info{flex:1;min-width:0;}
.list-title{font-size:.85rem;font-weight:700;color:#f0f0f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.list-genres{font-size:.68rem;color:#555;margin-top:3px;}
.list-meta{display:flex;gap:6px;align-items:center;flex-shrink:0;}


/* ── Sidebar brand ── */
.sb-brand{text-align:center;padding:.3rem 0 1.2rem;border-bottom:1px solid #1a1a1a;margin-bottom:1.2rem;}
.sb-brand-title{font-size:1.5rem;font-weight:900;color:#f0f0f0;letter-spacing:-.02em;}
.sb-accent{color:#e50914;}
.sb-sub{color:#333;font-size:.65rem;letter-spacing:.15em;text-transform:uppercase;margin-top:2px;}

/* ── Stat block ── */
.stat-block{background:#111;border:1px solid #1a1a1a;border-radius:10px;padding:.8rem 1rem;margin-top:1.2rem;}
.stat-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;}
.stat-label{color:#444;font-size:.7rem;}
.stat-value{color:#777;font-size:.75rem;font-weight:600;}
.stat-title{color:#333;font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.55rem;}


/* ── Watched stars ── */
.star-display{color:#facc15;font-size:.95rem;letter-spacing:1px;}

/* ── Empty state ── */
.empty{text-align:center;padding:4rem 2rem;}
.empty-icon{font-size:3rem;margin-bottom:1rem;}
.empty-text{font-size:1rem;font-weight:500;color:#333;}
.empty-sub{font-size:.8rem;color:#2a2a2a;margin-top:.4rem;}


/* ── Where to Watch ── */
.wtw-section{margin-top:.8rem;padding-top:.8rem;border-top:1px solid #1a1a1a;}
.wtw-label{font-size:.63rem;color:#444;text-transform:uppercase;letter-spacing:.14em;font-weight:700;margin-bottom:.7rem;}
.wtw-logos{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;}
.wtw-item{display:flex;flex-direction:column;align-items:center;gap:5px;cursor:pointer;}
.wtw-logo{width:50px;height:50px;border-radius:10px;object-fit:cover;border:1px solid #222;transition:transform .2s,box-shadow .2s;}
.wtw-logo:hover{transform:scale(1.08);box-shadow:0 4px 16px rgba(229,9,20,.3);}
.wtw-name{font-size:.54rem;color:#555;text-align:center;max-width:54px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.wtw-none{font-size:.78rem;color:#333;font-style:italic;}


/* ── Beautiful dialog action buttons ── */
[data-testid="stDialog"] .dlg-actions .stButton>button{
    border-radius:25px!important;font-weight:700!important;font-size:.82rem!important;
    letter-spacing:.06em!important;padding:.52rem 1.4rem!important;
    text-transform:uppercase!important;transition:all .2s!important;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_data():
    movies = pd.read_csv("movies.csv")
    movies["genres"] = movies["genres"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    ratings = pd.read_csv("ratings.csv")
    n_ratings = len(ratings)

    avg = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    avg.columns = ["movieId", "avg_rating", "n_ratings_movie"]
    movies2 = movies.merge(avg, on="movieId", how="left")
    movies2["avg_rating"] = movies2["avg_rating"].fillna(0)
    movies2["n_ratings_movie"] = movies2["n_ratings_movie"].fillna(0)
    C = movies2["avg_rating"].mean()
    m = movies2["n_ratings_movie"].quantile(0.7)
    movies2["pop_score"] = (
        (movies2["n_ratings_movie"] / (movies2["n_ratings_movie"] + m)) * movies2["avg_rating"]
        + (m / (movies2["n_ratings_movie"] + m)) * C
    )

    user_movie = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    rated_ids = user_movie.columns.tolist()
    mid_to_col = {int(mid): i for i, mid in enumerate(rated_ids)}
    item_matrix = csr_matrix(user_movie.values).T.tocsr()
    col_map = np.array([mid_to_col.get(int(mid), -1) for mid in movies2["movieId"]])

    return movies2, content_sim, indices, item_matrix, mid_to_col, col_map, n_ratings


movies, content_sim, indices, item_matrix, mid_to_col, col_map, n_ratings = load_data()

N_RECS = 20

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in [("wishlist", []), ("watched", {}),
               ("_mv_results", []), ("_mv_query", ""), ("_mv_similar", []), ("_mv_seed", None), ("_mv_sim_page", 0),
               ("_tv_results", []), ("_tv_query", ""), ("_tv_similar", []), ("_tv_seed", None), ("_tv_sim_page", 0),
               ("_wish_import", False), ("_seen_import", False), ("_goto_tab", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_title(t):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", t).strip()

@st.cache_data(show_spinner=False)
def get_movie_info(title, api_key):
    if not api_key:
        return None, None, None
    params = {"api_key": api_key, "query": clean_title(title), "language": "en-US"}
    ym = re.search(r"\((\d{4})\)", title)
    if ym:
        params["year"] = ym.group(1)
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=5)
        res = r.json().get("results", [])
        if res:
            m = res[0]
            poster = f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get("poster_path") else None
            return poster, m["id"], m.get("vote_average")
    except Exception:
        pass
    return None, None, None


@st.cache_data(show_spinner=False)
def get_tv_info(title, api_key):
    """Like get_movie_info but searches the TV endpoint."""
    if not api_key:
        return None, None, None
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/tv",
            params={"api_key": api_key, "query": clean_title(title), "language": "en-US"}, timeout=5
        )
        res = r.json().get("results", [])
        if res:
            s = res[0]
            poster = f"https://image.tmdb.org/t/p/w342{s['poster_path']}" if s.get("poster_path") else None
            return poster, s["id"], s.get("vote_average")
    except Exception:
        pass
    return None, None, None


@st.cache_data(show_spinner=False)
def get_media_info(title, api_key, genre_hint=""):
    """Fetch poster/id/rating for a movie or TV show.
    Uses genre_hint to try the right endpoint first, then falls back to the other.
    Returns (poster, tmdb_id, rating, media_type) where media_type is 'movie' or 'tv'."""
    if not api_key:
        return None, None, None, "movie"
    hint_tv = "TV Show" in str(genre_hint)
    if hint_tv:
        poster, tmdb_id, rating = get_tv_info(title, api_key)
        if tmdb_id:
            return poster, tmdb_id, rating, "tv"
        poster, tmdb_id, rating = get_movie_info(title, api_key)
        return poster, tmdb_id, rating, "movie"
    else:
        poster, tmdb_id, rating = get_movie_info(title, api_key)
        if tmdb_id:
            return poster, tmdb_id, rating, "movie"
        # fall back to TV search for items like anime series or shows
        poster, tmdb_id, rating = get_tv_info(title, api_key)
        if tmdb_id:
            return poster, tmdb_id, rating, "tv"
        return None, None, None, "movie"


@st.cache_data(show_spinner=False)
def get_trailer_url(tmdb_id, api_key):
    if not tmdb_id or not api_key:
        return None
    try:
        vids = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos",
            params={"api_key": api_key}, timeout=5
        ).json().get("results", [])
        for v in vids:
            if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                return f"https://www.youtube.com/watch?v={v['key']}"
        for v in vids:
            if v.get("site") == "YouTube":
                return f"https://www.youtube.com/watch?v={v['key']}"
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def get_trending(api_key):
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/trending/movie/week",
            params={"api_key": api_key, "language": "en-US"}, timeout=5
        )
        return r.json().get("results", [])[:30]
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=300)
def search_tv(query, api_key):
    if not query or not api_key:
        return []
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/tv",
            params={"api_key": api_key, "query": query, "language": "en-US"}, timeout=5
        )
        return r.json().get("results", [])[:8]
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def get_trending_tv(api_key):
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/trending/tv/week",
            params={"api_key": api_key, "language": "en-US"}, timeout=5
        )
        return r.json().get("results", [])[:20]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_similar_tv(tv_id, api_key):
    if not tv_id or not api_key:
        return []
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/tv/{tv_id}/recommendations",
            params={"api_key": api_key, "language": "en-US"}, timeout=5
        )
        return r.json().get("results", [])[:20]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_tv_trailer(tv_id, api_key):
    if not tv_id or not api_key:
        return None
    try:
        vids = requests.get(
            f"https://api.themoviedb.org/3/tv/{tv_id}/videos",
            params={"api_key": api_key}, timeout=5
        ).json().get("results", [])
        for v in vids:
            if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                return f"https://www.youtube.com/watch?v={v['key']}"
        for v in vids:
            if v.get("site") == "YouTube":
                return f"https://www.youtube.com/watch?v={v['key']}"
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def get_tv_details(tv_id, api_key):
    if not tv_id or not api_key:
        return {}
    try:
        det  = requests.get(f"https://api.themoviedb.org/3/tv/{tv_id}",
                            params={"api_key": api_key, "language": "en-US"}, timeout=5).json()
        cred = requests.get(f"https://api.themoviedb.org/3/tv/{tv_id}/credits",
                            params={"api_key": api_key}, timeout=5).json()
        return {
            "overview":   det.get("overview", ""),
            "status":     det.get("status", ""),
            "seasons":    det.get("number_of_seasons", 0),
            "episodes":   det.get("number_of_episodes", 0),
            "first_air":  (det.get("first_air_date") or "")[:4],
            "last_air":   (det.get("last_air_date") or "")[:4],
            "network":    det.get("networks", [{}])[0].get("name", "") if det.get("networks") else "",
            "genres":     [g["name"] for g in det.get("genres", [])],
            "vote":       det.get("vote_average"),
            "poster":     f"https://image.tmdb.org/t/p/w342{det['poster_path']}" if det.get("poster_path") else None,
            "cast":       [c["name"] for c in cred.get("cast", [])[:8]],
            "created_by": ", ".join(c["name"] for c in det.get("created_by", [])),
        }
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def get_movie_details_full(movie_id, api_key):
    if not movie_id or not api_key:
        return {}
    try:
        det  = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                            params={"api_key": api_key, "language": "en-US"}, timeout=5).json()
        cred = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits",
                            params={"api_key": api_key}, timeout=5).json()
        return {
            "overview":  det.get("overview", ""),
            "tagline":   det.get("tagline", ""),
            "runtime":   det.get("runtime"),
            "vote":      det.get("vote_average"),
            "votes":     det.get("vote_count"),
            "year":      (det.get("release_date") or "")[:4],
            "genres":    [g["name"] for g in det.get("genres", [])],
            "poster":    f"https://image.tmdb.org/t/p/w342{det['poster_path']}" if det.get("poster_path") else None,
            "cast":      [c["name"] for c in cred.get("cast", [])[:8]],
            "director":  next((c["name"] for c in cred.get("crew", []) if c["job"] == "Director"), None),
            "budget":    det.get("budget", 0),
        }
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=300)
def search_movies_tmdb(query, api_key):
    if not query or not api_key:
        return []
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": query, "language": "en-US"}, timeout=5
        )
        return r.json().get("results", [])[:8]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_similar_movies(movie_id, api_key):
    if not movie_id or not api_key:
        return []
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations",
            params={"api_key": api_key, "language": "en-US"}, timeout=5
        )
        return r.json().get("results", [])[:20]
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def get_watch_providers(tmdb_id, api_key, media_type="movie", country="FR"):
    if not tmdb_id or not api_key:
        return [], None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/watch/providers",
            params={"api_key": api_key}, timeout=5
        ).json()
        results = r.get("results", {})
        data = results.get(country) or results.get("US") or {}
        flatrate = data.get("flatrate", [])[:6]
        link = data.get("link")
        return flatrate, link
    except Exception:
        return [], None

def recommend(title, n=10):
    idx = int(indices[title])
    movie_id = int(movies.iloc[idx]["movieId"])

    content_scores = np.array(content_sim[idx]).flatten()

    collab_scores = np.zeros(len(movies))
    if movie_id in mid_to_col:
        c = mid_to_col[movie_id]
        sims = cosine_similarity(item_matrix[c], item_matrix).flatten()
        valid = col_map >= 0
        collab_scores[valid] = sims[col_map[valid]]

    if content_scores.max() > 0: content_scores /= content_scores.max()
    if collab_scores.max() > 0:  collab_scores  /= collab_scores.max()

    hybrid = 0.6 * content_scores + 0.4 * collab_scores
    hybrid[idx] = -1

    top = np.argsort(hybrid)[::-1][:n]
    res = movies[["title", "genres"]].iloc[top].copy()
    res["score"]   = [float(hybrid[i]) for i in top]
    res["c_score"] = [float(content_scores[i]) for i in top]
    res["k_score"] = [float(collab_scores[i]) for i in top]
    return res


# ── Movie detail dialog ───────────────────────────────────────────────────────
@st.dialog("Movie Details", width="large")
def show_movie_detail(mv, api_key):
    title = mv.get("title", "")
    iw = any(w["title"] == title for w in st.session_state.wishlist)
    iv = title in st.session_state.watched

    # ── Beautiful action buttons at TOP ────────────────────────────────────
    st.markdown('<div class="dlg-actions">', unsafe_allow_html=True)
    ba, bb, bc, _ = st.columns([1.1, 1.1, 1.1, 1])
    with ba:
        wish_lbl = "✓ In Wishlist" if iw else "🔖 Wishlist"
        if st.button(wish_lbl, key="dlg_mv_wish", use_container_width=True,
                     type="secondary"):
            if not iw:
                st.session_state.wishlist.append({"title": title, "genres": "Movie"})
            else:
                st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != title]
            st.rerun()
    with bb:
        seen_lbl = "✓ Watched" if iv else "👁 Mark Watched"
        if st.button(seen_lbl, key="dlg_mv_seen", use_container_width=True,
                     type="secondary"):
            if not iv:
                st.session_state.watched[title] = 0
            st.rerun()
    with bc:
        if st.button("🎭 Find Similar", key="dlg_mv_sim",
                     use_container_width=True, type="secondary"):
            st.session_state._mv_seed    = mv
            st.session_state._mv_similar = get_similar_movies(mv["id"], api_key)
            st.session_state._mv_sim_page = 0
            st.session_state._goto_tab   = "movies"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:.6rem 0 1rem'>", unsafe_allow_html=True)

    # ── Main content ───────────────────────────────────────────────────────
    det = get_movie_details_full(mv["id"], api_key)
    col_img, col_info = st.columns([1, 2])
    with col_img:
        poster = det.get("poster") or (f"https://image.tmdb.org/t/p/w342{mv['poster_path']}" if mv.get("poster_path") else None)
        if poster:
            st.markdown(f'<img src="{poster}" style="width:100%;border-radius:10px;">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-poster" style="height:280px;border-radius:10px;">🎬</div>', unsafe_allow_html=True)
    with col_info:
        year  = det.get("year", "")
        st.markdown(f"### {title}{' (' + year + ')' if year else ''}")
        if det.get("tagline"):
            st.caption(f'*{det["tagline"]}*')
        m1, m2, m3 = st.columns(3)
        if det.get("vote"):    m1.metric("TMDb",    f"★ {det['vote']:.1f}")
        if det.get("runtime"): m2.metric("Runtime", f"{det['runtime']} min")
        if det.get("votes"):   m3.metric("Votes",   f"{det['votes']:,}")
        if det.get("genres"):
            st.markdown(" ".join(f'`{g}`' for g in det["genres"]))
        if det.get("director"):
            st.markdown(f"**Director:** {det['director']}")
        if det.get("overview"):
            st.markdown("**Synopsis**")
            st.write(det["overview"])
        if det.get("cast"):
            st.markdown(f"**Cast:** {', '.join(det['cast'])}")

    # ── Where to Watch ─────────────────────────────────────────────────────
    providers, watch_link = get_watch_providers(mv["id"], api_key, "movie")
    if providers:
        logos_html = "".join(
            f'<div class="wtw-item"><img class="wtw-logo" src="https://image.tmdb.org/t/p/w92{p["logo_path"]}" title="{p["provider_name"]}"><div class="wtw-name">{p["provider_name"]}</div></div>'
            for p in providers if p.get("logo_path")
        )
        st.markdown(f'<div class="wtw-section"><div class="wtw-label">Available on</div><div class="wtw-logos">{logos_html}</div></div>', unsafe_allow_html=True)
        if watch_link:
            st.link_button("View all platforms →", watch_link)
    else:
        st.markdown('<div class="wtw-section"><div class="wtw-label">Available on</div><div class="wtw-none">Not available for streaming in your country</div></div>', unsafe_allow_html=True)

    trailer = get_trailer_url(mv["id"], api_key)
    if trailer:
        st.markdown("---")
        vid_id = trailer.split("v=")[-1]
        st.markdown(f'<iframe width="100%" height="220" src="https://www.youtube.com/embed/{vid_id}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

# ── TV Show detail dialog ─────────────────────────────────────────────────────
@st.dialog("Series Details", width="large")
def show_tv_detail(show, api_key):
    name = show.get("name", "")
    iw = any(w["title"] == name for w in st.session_state.wishlist)
    iv = name in st.session_state.watched

    # ── Beautiful action buttons at TOP ────────────────────────────────────
    st.markdown('<div class="dlg-actions">', unsafe_allow_html=True)
    ba, bb, bc, _ = st.columns([1.1, 1.1, 1.1, 1])
    with ba:
        wish_lbl = "✓ In Wishlist" if iw else "🔖 Wishlist"
        if st.button(wish_lbl, key="dlg_tv_wish", use_container_width=True,
                     type="secondary"):
            if not iw:
                st.session_state.wishlist.append({"title": name, "genres": "TV Show"})
            else:
                st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != name]
            st.rerun()
    with bb:
        seen_lbl = "✓ Watched" if iv else "👁 Mark Watched"
        if st.button(seen_lbl, key="dlg_tv_seen", use_container_width=True,
                     type="secondary"):
            if not iv:
                st.session_state.watched[name] = 0
            st.rerun()
    with bc:
        if st.button("🎭 Find Similar", key="dlg_tv_sim",
                     use_container_width=True, type="secondary"):
            st.session_state._tv_seed    = show
            st.session_state._tv_similar = get_similar_tv(show["id"], api_key)
            st.session_state._tv_sim_page = 0
            st.session_state._goto_tab   = "tv"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #1a1a1a;margin:.6rem 0 1rem'>", unsafe_allow_html=True)

    # ── Main content ───────────────────────────────────────────────────────
    det = get_tv_details(show["id"], api_key)
    col_img, col_info = st.columns([1, 2])
    with col_img:
        poster = det.get("poster") or (f"https://image.tmdb.org/t/p/w342{show['poster_path']}" if show.get("poster_path") else None)
        if poster:
            st.markdown(f'<img src="{poster}" style="width:100%;border-radius:10px;">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-poster" style="height:280px;border-radius:10px;">📺</div>', unsafe_allow_html=True)
    with col_info:
        st.markdown(f"### {name}")
        m1, m2, m3, m4 = st.columns(4)
        if det.get("seasons"):  m1.metric("Seasons",  det["seasons"])
        if det.get("episodes"): m2.metric("Episodes", det["episodes"])
        if det.get("vote"):     m3.metric("TMDb",     f"★ {det['vote']:.1f}")
        if det.get("status"):   m4.metric("Status",   det["status"])
        if det.get("genres"):
            st.markdown(" ".join(f'`{g}`' for g in det["genres"]))
        if det.get("network"):
            st.caption(f"📡 {det['network']}  ·  {det.get('first_air','')}–{det.get('last_air','')}")
        if det.get("created_by"):
            st.markdown(f"**Created by:** {det['created_by']}")
        if det.get("overview"):
            st.markdown("**Synopsis**")
            st.write(det["overview"])
        if det.get("cast"):
            st.markdown(f"**Cast:** {', '.join(det['cast'])}")

    # ── Where to Watch ─────────────────────────────────────────────────────
    providers, watch_link = get_watch_providers(show["id"], api_key, "tv")
    if providers:
        logos_html = "".join(
            f'<div class="wtw-item"><img class="wtw-logo" src="https://image.tmdb.org/t/p/w92{p["logo_path"]}" title="{p["provider_name"]}"><div class="wtw-name">{p["provider_name"]}</div></div>'
            for p in providers if p.get("logo_path")
        )
        st.markdown(f'<div class="wtw-section"><div class="wtw-label">Available on</div><div class="wtw-logos">{logos_html}</div></div>', unsafe_allow_html=True)
        if watch_link:
            st.link_button("View all platforms →", watch_link)
    else:
        st.markdown('<div class="wtw-section"><div class="wtw-label">Available on</div><div class="wtw-none">Not available for streaming in your country</div></div>', unsafe_allow_html=True)

    trailer = get_tv_trailer(show["id"], api_key)
    if trailer:
        st.markdown("---")
        vid_id = trailer.split("v=")[-1]
        st.markdown(f'<iframe width="100%" height="220" src="https://www.youtube.com/embed/{vid_id}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)


def render_poster_btn(key, poster_url, title, year, vote, iw=False, iv=False):
    """
    Renders a clickable poster as a styled Streamlit button + info strip below.
    The button itself IS the poster image via CSS background-image.
    No JavaScript required.
    Returns True if the poster was clicked.
    """
    if poster_url:
        bg_image   = f'url("{poster_url}")'
        bg_size    = 'contain'
        bg_pos     = 'center center'
        bg_repeat  = 'no-repeat'
        bg_color   = '#0d0d0d'
    else:
        bg_image   = 'none'
        bg_size    = 'auto'
        bg_pos     = 'center'
        bg_repeat  = 'no-repeat'
        bg_color   = '#111'
    # Per-card CSS: use descendant selector (space) not direct-child (>)
    # because Streamlit wraps the <button> inside an extra .stButton div.
    # Split background into individual properties so nothing gets overridden.
    st.markdown(f"""<style>
    .st-key-{key}{{margin-bottom:0!important;}}
    .st-key-{key} .stButton>button,
    .st-key-{key} [data-testid="stBaseButton-secondary"]{{
        background-image:{bg_image}!important;
        background-size:{bg_size}!important;
        background-position:{bg_pos}!important;
        background-repeat:{bg_repeat}!important;
        background-color:{bg_color}!important;
        height:300px;width:100%;padding:0;
        border-radius:12px 12px 0 0;
        border:1px solid #1e1e1e!important;border-bottom:none!important;
        transition:border-color .28s,transform .28s,box-shadow .28s;cursor:pointer;
        color:transparent!important;
    }}
    .st-key-{key} .stButton>button:hover,
    .st-key-{key} [data-testid="stBaseButton-secondary"]:hover{{
        border-color:#e50914!important;transform:translateY(-4px);
        box-shadow:0 16px 36px rgba(229,9,20,.28);
    }}
    .st-key-{key} .stButton>button:focus,
    .st-key-{key} [data-testid="stBaseButton-secondary"]:focus{{outline:none!important;box-shadow:none!important;}}
    .st-key-{key} .stButton>button p,
    .st-key-{key} [data-testid="stBaseButton-secondary"] p{{display:none;}}
    </style>""", unsafe_allow_html=True)

    clicked = st.button("", key=key, use_container_width=True)

    # Info strip — visually attached to the bottom of the poster
    tmdb_b  = f'<span class="tmdb-badge">★ {vote:.1f}</span>' if vote else ""
    state_b = ""
    if iw: state_b += '<span style="font-size:.72rem;color:#e50914;margin-left:3px">🔖</span>'
    if iv: state_b += '<span style="font-size:.72rem;color:#22c55e;margin-left:3px">✓</span>'
    st.markdown(f"""<div style="background:#111;border:1px solid #1e1e1e;border-top:none;
        border-radius:0 0 12px 12px;padding:.65rem .7rem .55rem;margin-top:0;">
        <div class="card-title" style="margin:0 0 .38rem;">{title}</div>
        <div class="genre-row"><span class="gtag">{year}</span></div>
        <div class="card-badges">{tmdb_b}{state_b}</div>
    </div>""", unsafe_allow_html=True)

    return clicked


def render_grid(rows, api_key, show_score=False, prefix=""):
    in_wishlist = {w["title"] for w in st.session_state.wishlist}
    in_watched  = set(st.session_state.watched.keys())
    chunks = [rows.iloc[i:i+5] for i in range(0, len(rows), 5)]
    for chunk in chunks:
        cols = st.columns(5)
        for col, (row_idx, row) in zip(cols, chunk.iterrows()):
            with col:
                poster, tmdb_id, rating = get_movie_info(row["title"], api_key) if api_key else (None, None, None)
                iw = row["title"] in in_wishlist
                iv = row["title"] in in_watched
                score = row.get("score") if show_score else None
                vote_for_display = rating
                key = f"{prefix}_{row_idx}"
                if render_poster_btn(key, poster, clean_title(row["title"]),
                                     "", vote_for_display, iw, iv):
                    mv_dict = {"id": tmdb_id, "title": row["title"], "poster_path": None}
                    show_movie_detail(mv_dict, api_key)



# ── Fragment renderers ────────────────────────────────────────────────────────
# @st.fragment limits reruns to just these sections, so Remove/rate actions
# don't scroll the page back to the top.

@st.fragment
def _wishlist_cards(api_key):
    if not st.session_state.wishlist:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">🔖</div>
            <div class="empty-text">Your wishlist is empty</div>
            <div class="empty-sub">Search for a movie and click 🔖 Wishlist, or import a CSV above.</div>
        </div>""", unsafe_allow_html=True)
        return
    st.markdown(f"""
    <div class="sec-hdr" style="margin-top:1rem">
        <span class="sec-hdr-title">My Wishlist</span>
        <div class="sec-hdr-line"></div>
        <span class="sec-hdr-count">{len(st.session_state.wishlist)} items</span>
    </div>""", unsafe_allow_html=True)
    wish_items = st.session_state.wishlist
    chunks = [(i, wish_items[i:i+5]) for i in range(0, len(wish_items), 5)]
    for chunk_start, chunk in chunks:
        cols = st.columns(5)
        for col_idx, (col, item) in enumerate(zip(cols, chunk)):
            with col:
                global_idx = chunk_start + col_idx
                poster, tmdb_id, rating, mtype = get_media_info(item["title"], api_key, item.get("genres", "")) if api_key else (None, None, None, "movie")
                year_m = re.search(r"\((\d{4})\)", item["title"])
                year = year_m.group(1) if year_m else ""
                if render_poster_btn(f"wish_{global_idx}", poster, clean_title(item["title"]),
                                     year, rating, iw=True):
                    if mtype == "tv":
                        show_tv_detail({"id": tmdb_id, "name": item["title"], "poster_path": None}, api_key)
                    else:
                        show_movie_detail({"id": tmdb_id, "title": item["title"], "poster_path": None}, api_key)
                if st.button("✕ Remove", key=f"wrm_{global_idx}", use_container_width=True):
                    st.session_state.wishlist = [w for w in st.session_state.wishlist
                                                 if w["title"] != item["title"]]
                    st.rerun(scope="fragment")


@st.fragment
def _watched_items(api_key):
    st.markdown(f"""
    <div class="sec-hdr" style="margin-top:1rem">
        <span class="sec-hdr-title">Watched Movies</span>
        <div class="sec-hdr-line"></div>
        <span class="sec-hdr-count">{len(st.session_state.watched)} movies</span>
    </div>""", unsafe_allow_html=True)

    for sw_idx, (title, personal_rating) in enumerate(list(st.session_state.watched.items())):
        poster, tmdb_id, tmdb_rating, mtype = get_media_info(title, api_key) if api_key else (None, None, None, "movie")
        img_html   = f'<img class="list-thumb" src="{poster}">' if poster else '<div class="list-thumb-ph">🎬</div>'
        genres_row = movies[movies["title"] == title]["genres"].iloc[0] if title in movies["title"].values else ""
        genre_str  = " · ".join(g for g in genres_row.split("|")[:3] if g and g != "(no genres listed)")
        stars      = "★" * personal_rating + "☆" * (5 - personal_rating) if personal_rating else "☆☆☆☆☆"
        tmdb_html  = f'<span class="tmdb-badge">★ {tmdb_rating:.1f}</span>' if tmdb_rating else ""
        sw_key     = f"sw_{sw_idx}"

        st.markdown(f"""<style>
        .st-key-{sw_key} .stButton>button{{
            background:transparent!important;border:none!important;
            color:#f0f0f0!important;font-size:.85rem!important;font-weight:700!important;
            padding:0!important;min-height:auto!important;text-align:left!important;
            white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;
            line-height:1.4!important;box-shadow:none!important;letter-spacing:0!important;
            text-transform:none!important;
        }}
        .st-key-{sw_key} .stButton>button:hover{{
            color:#e50914!important;border:none!important;box-shadow:none!important;
            transform:none!important;
        }}
        </style>""", unsafe_allow_html=True)

        col_img, col_info, col_meta = st.columns([1, 6, 3])
        with col_img:
            st.markdown(img_html, unsafe_allow_html=True)
        with col_info:
            if st.button(clean_title(title), key=sw_key, use_container_width=True):
                if mtype == "tv":
                    show_tv_detail({"id": tmdb_id, "name": title, "poster_path": None}, api_key)
                else:
                    show_movie_detail({"id": tmdb_id, "title": title, "poster_path": None}, api_key)
            st.markdown(f'<div class="list-genres" style="margin-top:2px">{genre_str}</div>', unsafe_allow_html=True)
        with col_meta:
            st.markdown(f'<div style="padding-top:.35rem"><span class="star-display">{stars}</span>{tmdb_html}</div>', unsafe_allow_html=True)

        ca, cb, _ = st.columns([1, 1, 5])
        with ca:
            new_r = st.select_slider("Rating", [0,1,2,3,4,5], value=personal_rating,
                                      format_func=lambda x: "★"*x or "—",
                                      key=f"rate_{title}", label_visibility="collapsed")
            if new_r != personal_rating:
                st.session_state.watched[title] = new_r
                st.rerun(scope="fragment")
        with cb:
            if st.button("✕ Remove", key=f"sr_{title}"):
                del st.session_state.watched[title]
                st.rerun(scope="fragment")
        st.markdown("<hr style='border:none;border-top:1px solid #141414;margin:.4rem 0'>", unsafe_allow_html=True)

    # Recommendations based on top-rated watched movies
    top_rated = [t for t, r in sorted(st.session_state.watched.items(), key=lambda x: -x[1]) if r >= 4]
    if top_rated:
        st.markdown("""
        <div class="sec-hdr" style="margin-top:2rem">
            <span class="sec-hdr-title">Recommended Based on Your Top-Rated Films</span>
            <div class="sec-hdr-line"></div>
        </div>""", unsafe_allow_html=True)
        seed = top_rated[0]
        with st.spinner("Computing..."):
            recs_from_seen = recommend(seed, N_RECS)
        if not recs_from_seen.empty:
            render_grid(recs_from_seen, api_key, show_score=True, prefix="seen")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-title">Next<span class="sb-accent">Watch</span></div>
        <div class="sb-sub">Movie Discovery</div>
    </div>""", unsafe_allow_html=True)

    tmdb_api_key = st.secrets.get("TMDB_API_KEY", "")

    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-title">Dataset</div>
        <div class="stat-row"><span class="stat-label">Movies</span><span class="stat-value">{len(movies):,}</span></div>
        <div class="stat-row"><span class="stat-label">Ratings</span><span class="stat-value">{n_ratings:,}</span></div>
        <div class="stat-row"><span class="stat-label">Wishlist</span><span class="stat-value">{len(st.session_state.wishlist)}</span></div>
        <div class="stat-row"><span class="stat-label">Watched</span><span class="stat-value">{len(st.session_state.watched)}</span></div>
    </div>""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🎬 Your Personal Cinema</div>
    <h1 class="hero-title">Next<span class="hero-accent">Watch</span></h1>
    <p class="hero-sub">Discover movies &amp; series — powered by TMDb</p>
    <div class="hero-pills">
        <span class="hero-pill">🎬 Movies</span>
        <span class="hero-pill">📺 TV Shows</span>
        <span class="hero-pill">🔥 Trending</span>
        <span class="hero-pill">🔖 Wishlist</span>
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-n">TMDb</span>
            <span class="hero-stat-l">Powered by</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-n">∞</span>
            <span class="hero-stat-l">Movies &amp; Shows</span>
        </div>
        <div class="hero-stat" style="border-right:none">
            <span class="hero-stat-n">Free</span>
            <span class="hero-stat-l">Always</span>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_disc, tab_tv, tab_wish, tab_seen = st.tabs([
    "🎬 Movies",
    "📺 TV Shows",
    "🔖 Wishlist",
    "✓ Watched",
])

# ══ TAB 1 — Movies ═══════════════════════════════════════════════════════════
with tab_disc:
    if True:
        mv_q = st.text_input("", placeholder="Search a movie...",
                             label_visibility="collapsed", key="mv_search")

        if mv_q != st.session_state._mv_query:
            st.session_state._mv_query   = mv_q
            st.session_state._mv_results = search_movies_tmdb(mv_q, tmdb_api_key) if mv_q else []
            st.session_state._mv_similar = []
            st.session_state._mv_seed    = None

        if st.session_state._mv_results:
            results = st.session_state._mv_results
            chunks  = [results[i:i+5] for i in range(0, len(results), 5)]
            for chunk in chunks:
                cols = st.columns(5)
                for col, mv in zip(cols, chunk):
                    with col:
                        iw = any(w["title"] == mv["title"] for w in st.session_state.wishlist)
                        poster_url = f"https://image.tmdb.org/t/p/w342{mv['poster_path']}" if mv.get('poster_path') else None
                        if render_poster_btn(f"mvd_{mv['id']}", poster_url, mv.get('title',''),
                                             (mv.get('release_date') or '')[:4], mv.get('vote_average'), iw):
                            show_movie_detail(mv, tmdb_api_key)

        elif mv_q and not st.session_state._mv_results:
            st.caption("No results found.")

        if st.session_state._mv_similar:
            seed    = st.session_state._mv_seed
            similar = st.session_state._mv_similar
            PAGE    = 10
            page    = st.session_state._mv_sim_page
            total_p = max(1, (len(similar) + PAGE - 1) // PAGE)
            page    = min(page, total_p - 1)
            _hcol, _xcol = st.columns([10, 1])
            with _hcol:
                st.markdown(f"""
                <div class="sec-hdr" style="margin-top:1.5rem">
                    <span class="sec-hdr-title">Similar to · <em>{seed['title']}</em></span>
                    <div class="sec-hdr-line"></div>
                    <span class="sec-hdr-count">{len(similar)} movies</span>
                </div>""", unsafe_allow_html=True)
            with _xcol:
                st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)
                if st.button("✕", key="mv_sim_close", help="Close similar section"):
                    st.session_state._mv_similar = []
                    st.session_state._mv_seed = None
                    st.session_state._mv_sim_page = 0
                    st.rerun()
            page_items = similar[page*PAGE:(page+1)*PAGE]
            sim_chunks = [page_items[i:i+5] for i in range(0, len(page_items), 5)]
            for chunk in sim_chunks:
                cols = st.columns(5)
                for col, mv in zip(cols, chunk):
                    with col:
                        iw = any(w["title"] == mv["title"] for w in st.session_state.wishlist)
                        poster_url = f"https://image.tmdb.org/t/p/w342{mv['poster_path']}" if mv.get('poster_path') else None
                        if render_poster_btn(f"mvsim_d_{mv['id']}", poster_url, mv.get('title',''),
                                             (mv.get('release_date') or '')[:4], mv.get('vote_average'), iw):
                            show_movie_detail(mv, tmdb_api_key)
            if total_p > 1:
                cp, ci, cn = st.columns([1, 2, 1])
                with cp:
                    if st.button("← Prev", disabled=page==0, key="mv_prev", use_container_width=True):
                        st.session_state._mv_sim_page -= 1
                        st.rerun()
                with ci:
                    st.markdown(f'<p style="text-align:center;color:#555;font-size:.8rem;margin-top:.5rem">Page {page+1} / {total_p}</p>', unsafe_allow_html=True)
                with cn:
                    if st.button("Next →", disabled=page>=total_p-1, key="mv_next", use_container_width=True):
                        st.session_state._mv_sim_page += 1
                        st.rerun()

        if not mv_q:
            st.markdown("""
            <div class="sec-hdr" style="margin-top:.5rem">
                <span class="sec-hdr-title">Trending This Week</span>
                <div class="sec-hdr-line"></div>
            </div>""", unsafe_allow_html=True)
            with st.spinner("Loading trending movies..."):
                trending_mv = get_trending(tmdb_api_key)
            if trending_mv:
                mv_chunks = [trending_mv[i:i+5] for i in range(0, len(trending_mv), 5)]
                for chunk in mv_chunks:
                    cols = st.columns(5)
                    for col, mv in zip(cols, chunk):
                        with col:
                            iw = any(w["title"] == mv["title"] for w in st.session_state.wishlist)
                            poster_url = f"https://image.tmdb.org/t/p/w342{mv['poster_path']}" if mv.get('poster_path') else None
                            if render_poster_btn(f"mvtrd_{mv['id']}", poster_url, mv.get('title',''),
                                                 (mv.get('release_date') or '')[:4], mv.get('vote_average'), iw):
                                show_movie_detail(mv, tmdb_api_key)


# ══ TAB 2 — TV Shows ═════════════════════════════════════════════════════════
with tab_tv:
    if True:
        # ── Search ──────────────────────────────────────────────────────────
        tv_q = st.text_input("", placeholder="Search a TV show...",
                             label_visibility="collapsed", key="tv_search")

        if tv_q != st.session_state._tv_query:
            st.session_state._tv_query   = tv_q
            st.session_state._tv_results = search_tv(tv_q, tmdb_api_key) if tv_q else []
            st.session_state._tv_similar = []
            st.session_state._tv_seed    = None

        # ── Show search results ──────────────────────────────────────────────
        if st.session_state._tv_results:
            results = st.session_state._tv_results
            chunks  = [results[i:i+5] for i in range(0, len(results), 5)]
            for chunk in chunks:
                cols = st.columns(5)
                for col, show in zip(cols, chunk):
                    with col:
                        iw = any(w["title"] == show["name"] for w in st.session_state.wishlist)
                        poster_url = f"https://image.tmdb.org/t/p/w342{show['poster_path']}" if show.get('poster_path') else None
                        if render_poster_btn(f"tvd_{show['id']}", poster_url, show.get('name',''),
                                             (show.get('first_air_date') or '')[:4], show.get('vote_average'), iw):
                            show_tv_detail(show, tmdb_api_key)

        elif tv_q and not st.session_state._tv_results:
            st.caption("No results found.")

        # ── Similar shows ────────────────────────────────────────────────────
        if st.session_state._tv_similar:
            seed    = st.session_state._tv_seed
            similar = st.session_state._tv_similar
            PAGE    = 10
            page    = min(st.session_state._tv_sim_page, max(0, (len(similar)-1)//PAGE))
            _hcol, _xcol = st.columns([10, 1])
            with _hcol:
                st.markdown(f"""
                <div class="sec-hdr" style="margin-top:1.5rem">
                    <span class="sec-hdr-title">Similar to · <em>{seed['name']}</em></span>
                    <div class="sec-hdr-line"></div>
                    <span class="sec-hdr-count">{len(similar)} shows</span>
                </div>""", unsafe_allow_html=True)
            with _xcol:
                st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)
                if st.button("✕", key="tv_sim_close", help="Close similar section"):
                    st.session_state._tv_similar = []
                    st.session_state._tv_seed = None
                    st.session_state._tv_sim_page = 0
                    st.rerun()
            page_items = similar[page*PAGE:(page+1)*PAGE]
            sim_chunks = [page_items[i:i+5] for i in range(0, len(page_items), 5)]
            for chunk in sim_chunks:
                cols = st.columns(5)
                for col, show in zip(cols, chunk):
                    with col:
                        iw = any(w["title"] == show["name"] for w in st.session_state.wishlist)
                        poster_url = f"https://image.tmdb.org/t/p/w342{show['poster_path']}" if show.get('poster_path') else None
                        if render_poster_btn(f"tvsimd_{show['id']}", poster_url, show.get('name',''),
                                             (show.get('first_air_date') or '')[:4], show.get('vote_average'), iw):
                            show_tv_detail(show, tmdb_api_key)
            total_p = max(1, (len(similar) + PAGE - 1) // PAGE)
            if total_p > 1:
                cp, ci, cn = st.columns([1, 2, 1])
                with cp:
                    if st.button("← Prev", disabled=page==0, key="tv_prev", use_container_width=True):
                        st.session_state._tv_sim_page -= 1
                        st.rerun()
                with ci:
                    st.markdown(f'<p style="text-align:center;color:#555;font-size:.8rem;margin-top:.5rem">Page {page+1} / {total_p}</p>', unsafe_allow_html=True)
                with cn:
                    if st.button("Next →", disabled=page>=total_p-1, key="tv_next", use_container_width=True):
                        st.session_state._tv_sim_page += 1
                        st.rerun()

        # ── Trending TV (shown when no search) ───────────────────────────────
        if not tv_q:
            st.markdown("""
            <div class="sec-hdr" style="margin-top:.5rem">
                <span class="sec-hdr-title">Trending This Week</span>
                <div class="sec-hdr-line"></div>
            </div>""", unsafe_allow_html=True)
            with st.spinner("Loading trending shows..."):
                trending_tv = get_trending_tv(tmdb_api_key)
            if trending_tv:
                tv_chunks = [trending_tv[i:i+5] for i in range(0, len(trending_tv), 5)]
                for chunk in tv_chunks:
                    cols = st.columns(5)
                    for col, show in zip(cols, chunk):
                        with col:
                            iw = any(w["title"] == show["name"] for w in st.session_state.wishlist)
                            poster_url = f"https://image.tmdb.org/t/p/w342{show['poster_path']}" if show.get('poster_path') else None
                            if render_poster_btn(f"tvtrd_{show['id']}", poster_url, show.get('name',''),
                                                 (show.get('first_air_date') or '')[:4], show.get('vote_average'), iw):
                                show_tv_detail(show, tmdb_api_key)

# ══ TAB 3 — Wishlist ═════════════════════════════════════════════════════════
with tab_wish:
    # ── Toolbar ──────────────────────────────────────────────────────────────
    if not st.session_state.wishlist:
        ca0, cb0, _ = st.columns([1.5, 1.5, 5])
        with ca0:
            st.download_button("⬇ Export CSV", pd.DataFrame(columns=["title","genres"]).to_csv(index=False),
                               "nextwatch_wishlist.csv", "text/csv",
                               use_container_width=True, disabled=True)
        with cb0:
            if st.button("⬆ Import CSV", key="wish_import_btn_empty", use_container_width=True):
                st.session_state._wish_import = not st.session_state._wish_import
    else:
        ca, cb, cc, _ = st.columns([1.5, 1.5, 1.5, 3])
        with ca:
            df_wish = pd.DataFrame(st.session_state.wishlist)
            st.download_button("⬇ Export CSV", df_wish.to_csv(index=False),
                               "nextwatch_wishlist.csv", "text/csv",
                               use_container_width=True)
        with cb:
            if st.button("⬆ Import CSV", key="wish_import_btn", use_container_width=True):
                st.session_state._wish_import = not st.session_state._wish_import
        with cc:
            if st.button("🗑 Clear all", key="wish_clear", use_container_width=True):
                st.session_state.wishlist = []
                st.rerun()

    if st.session_state._wish_import:
        uploaded = st.file_uploader(
            "Upload a CSV with columns: **title**, **genres**  "
            "(same format as the export)",
            type=["csv"], key="wish_uploader"
        )
        if uploaded is not None:
            import io
            try:
                df_in = pd.read_csv(io.BytesIO(uploaded.read()))
                if "title" not in df_in.columns:
                    st.error("CSV must have a 'title' column.")
                else:
                    existing = {w["title"] for w in st.session_state.wishlist}
                    added = 0
                    for _, row in df_in.iterrows():
                        t = str(row["title"]).strip()
                        g = str(row.get("genres", "Movie")).strip() if "genres" in df_in.columns else "Movie"
                        if t and t not in existing:
                            st.session_state.wishlist.append({"title": t, "genres": g})
                            existing.add(t)
                            added += 1
                    st.success(f"Imported {added} new item(s).")
                    st.session_state._wish_import = False
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    _wishlist_cards(tmdb_api_key)

# ══ TAB 5 — Watched ══════════════════════════════════════════════════════════
with tab_seen:
    # ── Toolbar always visible ───────────────────────────────────────────────
    if not st.session_state.watched:
        _empty_df = pd.DataFrame(columns=["title","personal_rating","genres"])
        ca0, cb0, _ = st.columns([1.5, 1.5, 5])
        with ca0:
            st.download_button("⬇ Export CSV", _empty_df.to_csv(index=False),
                               "nextwatch_watched.csv", "text/csv",
                               use_container_width=True, disabled=True)
        with cb0:
            if st.button("⬆ Import CSV", key="seen_import_btn_empty", use_container_width=True):
                st.session_state._seen_import = not st.session_state._seen_import

        if st.session_state._seen_import:
            uploaded_s = st.file_uploader(
                "Upload a CSV with columns: **title**, **personal_rating**",
                type=["csv"], key="seen_uploader_empty"
            )
            if uploaded_s is not None:
                import io
                try:
                    df_in = pd.read_csv(io.BytesIO(uploaded_s.read()))
                    if "title" not in df_in.columns:
                        st.error("CSV must have a 'title' column.")
                    else:
                        added = 0
                        for _, row in df_in.iterrows():
                            t = str(row["title"]).strip()
                            r = int(row["personal_rating"]) if "personal_rating" in df_in.columns and pd.notna(row["personal_rating"]) else 0
                            r = max(0, min(5, r))
                            if t and t not in st.session_state.watched:
                                st.session_state.watched[t] = r
                                added += 1
                        st.success(f"Imported {added} new item(s).")
                        st.session_state._seen_import = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
        else:
            st.markdown("""
            <div class="empty">
                <div class="empty-icon">✓</div>
                <div class="empty-text">No movies marked as watched</div>
                <div class="empty-sub">Click "✓ Watched" on any movie, or import a CSV above.</div>
            </div>""", unsafe_allow_html=True)
    else:
        df_seen = pd.DataFrame([
            {"title": t, "personal_rating": r,
             "genres": movies[movies["title"] == t]["genres"].iloc[0] if t in movies["title"].values else ""}
            for t, r in st.session_state.watched.items()
        ])
        ca, cb, cc, _ = st.columns([1.5, 1.5, 1.5, 3])
        with ca:
            st.download_button("⬇ Export CSV", df_seen.to_csv(index=False),
                               "nextwatch_watched.csv", "text/csv", use_container_width=True)
        with cb:
            if st.button("⬆ Import CSV", key="seen_import_btn", use_container_width=True):
                st.session_state._seen_import = not st.session_state._seen_import
        with cc:
            if st.button("🗑 Clear all", key="seen_clear", use_container_width=True):
                st.session_state.watched = {}
                st.rerun()

        if st.session_state._seen_import:
            uploaded_s = st.file_uploader(
                "Upload a CSV with columns: **title**, **personal_rating**  "
                "(same format as the export)",
                type=["csv"], key="seen_uploader"
            )
            if uploaded_s is not None:
                import io
                try:
                    df_in = pd.read_csv(io.BytesIO(uploaded_s.read()))
                    if "title" not in df_in.columns:
                        st.error("CSV must have a 'title' column.")
                    else:
                        added = 0
                        for _, row in df_in.iterrows():
                            t = str(row["title"]).strip()
                            r = int(row["personal_rating"]) if "personal_rating" in df_in.columns and pd.notna(row["personal_rating"]) else 0
                            r = max(0, min(5, r))
                            if t and t not in st.session_state.watched:
                                st.session_state.watched[t] = r
                                added += 1
                        st.success(f"Imported {added} new item(s).")
                        st.session_state._seen_import = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

        _watched_items(tmdb_api_key)

# ── Auto-switch tab after Find Similar ───────────────────────────────────────
if st.session_state._goto_tab:
    import streamlit.components.v1 as _cmp
    _emoji = "🎬" if st.session_state._goto_tab == "movies" else "📺"
    _cmp.html(f"""<script>
    (function() {{
        var _switch = function() {{
            var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
            for (var i = 0; i < tabs.length; i++) {{
                if (tabs[i].textContent.indexOf('{_emoji}') !== -1) {{
                    tabs[i].click();
                    return;
                }}
            }}
        }};
        // Small delay to let Streamlit finish rendering the tab bar
        window.parent.setTimeout(_switch, 120);
    }})();
    </script>""", height=0)
    st.session_state._goto_tab = None
