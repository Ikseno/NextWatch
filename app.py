import streamlit as st
import streamlit.components.v1 as components
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

/* ── Search box ── */
.search-wrap{margin-bottom:.8rem;}

/* ── Section header ── */
.sec-hdr{display:flex;align-items:center;gap:10px;margin:0 0 1.2rem;}
.sec-hdr-title{font-size:1.1rem;font-weight:700;color:#f0f0f0;white-space:nowrap;}
.sec-hdr-line{flex:1;height:1px;background:linear-gradient(90deg,#222,transparent);}
.sec-hdr-count{font-size:.7rem;color:#444;background:#111;border:1px solid #1e1e1e;padding:3px 10px;border-radius:12px;}

/* ── Movie card ── */
.movie-card{background:#111;border-radius:12px;overflow:hidden;border:1px solid #1e1e1e;transition:transform .3s,box-shadow .3s,border-color .3s;position:relative;}
.movie-card:hover{transform:translateY(-5px) scale(1.01);box-shadow:0 18px 40px rgba(229,9,20,.22);border-color:#e50914;}
.movie-poster{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;}
.no-poster{width:100%;aspect-ratio:2/3;background:linear-gradient(160deg,#181818,#0d0d0d);display:flex;flex-direction:column;align-items:center;justify-content:center;color:#2a2a2a;font-size:2.2rem;}
.no-poster-lbl{font-size:.6rem;color:#2a2a2a;margin-top:6px;letter-spacing:.12em;text-transform:uppercase;}
.card-body{padding:.65rem .7rem .55rem;}
.card-title{font-size:.8rem;font-weight:700;color:#f0f0f0;margin:0 0 .38rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.genre-row{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:.42rem;}
.gtag{background:#1a1a1a;color:#555;padding:2px 6px;border-radius:4px;font-size:.58rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;border:1px solid #222;}
.sim-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:20px;font-size:.67rem;font-weight:700;border:1px solid rgba(229,9,20,.25);background:rgba(229,9,20,.07);color:#ff4d4d;}
.sim-dot{width:5px;height:5px;background:#e50914;border-radius:50%;display:inline-block;}
.tmdb-badge{display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:20px;font-size:.67rem;font-weight:700;border:1px solid rgba(250,204,21,.2);background:rgba(250,204,21,.06);color:#facc15;margin-left:4px;}
.card-badges{display:flex;align-items:center;flex-wrap:wrap;gap:4px;}
.overlay-badge{position:absolute;top:7px;right:7px;display:flex;gap:4px;}
.ob{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.7rem;backdrop-filter:blur(6px);}
.ob-wish{background:rgba(229,9,20,.75);}
.ob-seen{background:rgba(34,197,94,.75);}

/* ── List row ── */
.list-row{display:flex;align-items:center;gap:12px;background:#111;border:1px solid #1e1e1e;border-radius:10px;padding:10px 12px;margin-bottom:6px;}
.list-thumb{width:40px;height:60px;object-fit:cover;border-radius:5px;flex-shrink:0;}
.list-thumb-ph{width:40px;height:60px;background:#1a1a1a;border-radius:5px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.9rem;color:#333;}
.list-info{flex:1;min-width:0;}
.list-title{font-size:.85rem;font-weight:700;color:#f0f0f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.list-genres{font-size:.68rem;color:#555;margin-top:3px;}
.list-meta{display:flex;gap:6px;align-items:center;flex-shrink:0;}

/* ── Genre pills ── */
.genre-pill-grid{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1.5rem;}
.genre-pill{
    background:#111;border:1px solid #222;color:#666;
    padding:7px 18px;border-radius:20px;font-size:.78rem;font-weight:600;
    cursor:pointer;transition:all .2s;letter-spacing:.03em;
}
.genre-pill:hover{border-color:#e50914;color:#e50914;}
.genre-pill.active{background:rgba(229,9,20,.12);border-color:#e50914;color:#e50914;}

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

/* ── Trending card ── */
.trend-card{background:#111;border-radius:12px;overflow:hidden;border:1px solid #1e1e1e;transition:transform .3s,box-shadow .3s,border-color .3s;}
.trend-card:hover{transform:translateY(-5px);box-shadow:0 18px 40px rgba(229,9,20,.18);border-color:#e50914;}

/* ── Watched stars ── */
.star-display{color:#facc15;font-size:.95rem;letter-spacing:1px;}

/* ── Empty state ── */
.empty{text-align:center;padding:4rem 2rem;}
.empty-icon{font-size:3rem;margin-bottom:1rem;}
.empty-text{font-size:1rem;font-weight:500;color:#333;}
.empty-sub{font-size:.8rem;color:#2a2a2a;margin-top:.4rem;}

/* ── Explanation bars ── */
.expl-bar{display:flex;align-items:center;gap:8px;margin:3px 0;}
.expl-label{font-size:.65rem;color:#555;width:80px;flex-shrink:0;}
.expl-track{flex:1;height:4px;background:#1a1a1a;border-radius:2px;}
.expl-fill-c{height:100%;border-radius:2px;background:#e50914;}
.expl-fill-k{height:100%;border-radius:2px;background:#3b82f6;}
.expl-pct{font-size:.63rem;color:#444;width:30px;text-align:right;}

/* ── Where to Watch ── */
.wtw-section{margin-top:.8rem;padding-top:.8rem;border-top:1px solid #1a1a1a;}
.wtw-label{font-size:.63rem;color:#444;text-transform:uppercase;letter-spacing:.14em;font-weight:700;margin-bottom:.7rem;}
.wtw-logos{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;}
.wtw-item{display:flex;flex-direction:column;align-items:center;gap:5px;cursor:pointer;}
.wtw-logo{width:50px;height:50px;border-radius:10px;object-fit:cover;border:1px solid #222;transition:transform .2s,box-shadow .2s;}
.wtw-logo:hover{transform:scale(1.08);box-shadow:0 4px 16px rgba(229,9,20,.3);}
.wtw-name{font-size:.54rem;color:#555;text-align:center;max-width:54px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.wtw-none{font-size:.78rem;color:#333;font-style:italic;}

/* ── Card provider strip ── */
.prov-strip{display:flex;gap:3px;margin-top:4px;flex-wrap:wrap;}
.prov-icon{width:18px;height:18px;border-radius:4px;object-fit:cover;border:1px solid #1e1e1e;}
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
               ("_tv_results", []), ("_tv_query", ""), ("_tv_similar", []), ("_tv_seed", None), ("_tv_sim_page", 0)]:
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
    det = get_movie_details_full(mv["id"], api_key)
    col_img, col_info = st.columns([1, 2])
    with col_img:
        poster = det.get("poster") or (f"https://image.tmdb.org/t/p/w342{mv['poster_path']}" if mv.get("poster_path") else None)
        if poster:
            st.markdown(f'<img src="{poster}" style="width:100%;border-radius:10px;">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-poster" style="height:280px;border-radius:10px;">🎬</div>', unsafe_allow_html=True)
    with col_info:
        title = mv.get("title", "")
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
    col_a, col_b, _ = st.columns([1, 1, 3])
    with col_a:
        iw = any(w["title"] == title for w in st.session_state.wishlist)
        if st.button("+ Wishlist" if not iw else "✓ Saved", key="dlg_mv_wish"):
            if not iw:
                st.session_state.wishlist.append({"title": title, "genres": "Movie"})
            st.rerun()
    with col_b:
        if st.button("✓ Watched", key="dlg_mv_seen"):
            if title not in st.session_state.watched:
                st.session_state.watched[title] = 0
            st.rerun()

# ── TV Show detail dialog ─────────────────────────────────────────────────────
@st.dialog("Series Details", width="large")
def show_tv_detail(show, api_key):
    det = get_tv_details(show["id"], api_key)
    col_img, col_info = st.columns([1, 2])
    with col_img:
        poster = det.get("poster") or (f"https://image.tmdb.org/t/p/w342{show['poster_path']}" if show.get("poster_path") else None)
        if poster:
            st.markdown(f'<img src="{poster}" style="width:100%;border-radius:10px;">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-poster" style="height:280px;border-radius:10px;">📺</div>', unsafe_allow_html=True)
    with col_info:
        st.markdown(f"### {show.get('name', '')}")
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
    col_a, col_b, _ = st.columns([1, 1, 3])
    with col_a:
        iw = any(w["title"] == show["name"] for w in st.session_state.wishlist)
        if st.button("+ Wishlist" if not iw else "✓ Saved", key="dlg_tv_wish"):
            if not iw:
                st.session_state.wishlist.append({"title": show["name"], "genres": "TV Show"})
            st.rerun()
    with col_b:
        if st.button("✓ Watched", key="dlg_tv_seen"):
            if show["name"] not in st.session_state.watched:
                st.session_state.watched[show["name"]] = 0
            st.rerun()

# ── Card / row helpers ────────────────────────────────────────────────────────
def card_html(title, genres, poster, tmdb_rating=None, score=None, in_wish=False, in_seen=False):
    tags = "".join(
        f'<span class="gtag">{g}</span>'
        for g in genres.split("|")[:3] if g and g != "(no genres listed)"
    )
    badges = ""
    if in_wish: badges += '<span class="ob ob-wish">🔖</span>'
    if in_seen: badges += '<span class="ob ob-seen">✓</span>'
    overlay   = f'<div class="overlay-badge">{badges}</div>' if badges else ""
    sim_html  = f'<span class="sim-badge"><span class="sim-dot"></span>{score*100:.0f}% match</span>' if score is not None else ""
    tmdb_html = f'<span class="tmdb-badge">★ {tmdb_rating:.1f}</span>' if tmdb_rating else ""
    img = f'<img class="movie-poster" src="{poster}">' if poster else '<div class="no-poster">🎬<div class="no-poster-lbl">No Poster</div></div>'
    return f"""
<div class="movie-card">
    {img}{overlay}
    <div class="card-body">
        <div class="card-title" title="{clean_title(title)}">{clean_title(title)}</div>
        <div class="genre-row">{tags}</div>
        <div class="card-badges">{sim_html}{tmdb_html}</div>
    </div>
</div>"""


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
                st.markdown(card_html(row["title"], row["genres"], poster, rating, score, iw, iv), unsafe_allow_html=True)

                k = f"{prefix}_{row_idx}"
                c1, c2 = st.columns(2)
                with c1:
                    lbl = "✓ Saved" if iw else "+ Wish"
                    if st.button(lbl, key=f"w{k}"):
                        if not iw:
                            st.session_state.wishlist.append({"title": row["title"], "genres": row["genres"]})
                        else:
                            st.session_state.wishlist = [x for x in st.session_state.wishlist if x["title"] != row["title"]]
                        st.rerun()
                with c2:
                    if st.button("✓ Watched", key=f"v{k}"):
                        if row["title"] not in st.session_state.watched:
                            st.session_state.watched[row["title"]] = 0
                        st.rerun()

                if tmdb_id and api_key:
                    t = get_trailer_url(tmdb_id, api_key)
                    if t:
                        st.link_button("▶ Trailer", t, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-title">Next<span class="sb-accent">Watch</span></div>
        <div class="sb-sub">Movie Discovery</div>
    </div>""", unsafe_allow_html=True)

    tmdb_api_key = st.secrets.get("TMDB_API_KEY", "")
    if not tmdb_api_key:
        tmdb_api_key = st.text_input("TMDb API Key", type="password",
                                      placeholder="Free key at themoviedb.org",
                                      label_visibility="collapsed")

    api_ok = bool(tmdb_api_key)
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-title">Dataset</div>
        <div class="stat-row"><span class="stat-label">Movies</span><span class="stat-value">{len(movies):,}</span></div>
        <div class="stat-row"><span class="stat-label">Ratings</span><span class="stat-value">{n_ratings:,}</span></div>
        <div class="stat-row"><span class="stat-label">Wishlist</span><span class="stat-value">{len(st.session_state.wishlist)}</span></div>
        <div class="stat-row"><span class="stat-label">Watched</span><span class="stat-value">{len(st.session_state.watched)}</span></div>
        <div class="stat-row" style="margin-top:7px;padding-top:7px;border-top:1px solid #1a1a1a;">
            <span class="stat-label">TMDb API</span>
            <span style="font-size:.7rem;font-weight:700;color:{'#22c55e' if api_ok else '#555'}">{'Connected' if api_ok else 'Not configured'}</span>
        </div>
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
tab_disc, tab_tv, tab_trend, tab_wish, tab_seen = st.tabs([
    "🎬 Movies",
    "📺 TV Shows",
    "🔥 Trending",
    f"🔖 Wishlist ({len(st.session_state.wishlist)})",
    f"✓ Watched ({len(st.session_state.watched)})",
])

# ══ TAB 1 — Movies ═══════════════════════════════════════════════════════════
with tab_disc:
    if not tmdb_api_key:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">🎬</div>
            <div class="empty-text">TMDb Key Required</div>
            <div class="empty-sub">Add your TMDb API key in the sidebar to search and browse movies.</div>
        </div>""", unsafe_allow_html=True)
    else:
        mv_q = st.text_input("", placeholder="Search a movie...",
                             label_visibility="collapsed", key="mv_search")

        if mv_q != st.session_state._mv_query:
            st.session_state._mv_query   = mv_q
            st.session_state._mv_results = search_movies_tmdb(mv_q, tmdb_api_key) if mv_q else []
            st.session_state._mv_similar = []
            st.session_state._mv_seed    = None

        if st.session_state._mv_results:
            results = st.session_state._mv_results
            chunks  = [results[i:i+4] for i in range(0, len(results), 4)]
            for chunk in chunks:
                cols = st.columns(4)
                for col, mv in zip(cols, chunk):
                    with col:
                        poster = f"https://image.tmdb.org/t/p/w342{mv['poster_path']}" if mv.get("poster_path") else None
                        vote   = mv.get("vote_average")
                        year   = (mv.get("release_date") or "")[:4]
                        tmdb_b = f'<span class="tmdb-badge">★ {vote:.1f}</span>' if vote else ""
                        img    = f'<img class="movie-poster" src="{poster}">' if poster else '<div class="no-poster">🎬<div class="no-poster-lbl">No Poster</div></div>'
                        st.markdown(f"""
                        <div class="movie-card">
                            {img}
                            <div class="card-body">
                                <div class="card-title">{mv['title']}</div>
                                <div class="genre-row"><span class="gtag">{year}</span></div>
                                <div class="card-badges">{tmdb_b}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            if st.button("Details", key=f"mvd_{mv['id']}"):
                                show_movie_detail(mv, tmdb_api_key)
                        with c2:
                            if st.button("Similar", key=f"mvs_{mv['id']}"):
                                st.session_state._mv_seed     = mv
                                st.session_state._mv_similar  = get_similar_movies(mv["id"], tmdb_api_key)
                                st.session_state._mv_sim_page = 0
                        with c3:
                            iw  = any(w["title"] == mv["title"] for w in st.session_state.wishlist)
                            lbl = "✓" if iw else "+ Wish"
                            if st.button(lbl, key=f"mvw_{mv['id']}"):
                                if not iw:
                                    st.session_state.wishlist.append({"title": mv["title"], "genres": "Movie"})
                                else:
                                    st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != mv["title"]]
                                st.rerun()
                        with c4:
                            t = get_trailer_url(mv["id"], tmdb_api_key)
                            if t:
                                st.link_button("▶", t, use_container_width=True)

        elif mv_q and not st.session_state._mv_results:
            st.caption("No results found.")

        if st.session_state._mv_similar:
            seed    = st.session_state._mv_seed
            similar = st.session_state._mv_similar
            PAGE    = 10
            page    = st.session_state._mv_sim_page
            total_p = max(1, (len(similar) + PAGE - 1) // PAGE)
            page    = min(page, total_p - 1)
            st.markdown(f"""
            <div class="sec-hdr" style="margin-top:1.5rem">
                <span class="sec-hdr-title">Similar to · <em>{seed['title']}</em></span>
                <div class="sec-hdr-line"></div>
                <span class="sec-hdr-count">{len(similar)} movies</span>
            </div>""", unsafe_allow_html=True)
            page_items = similar[page*PAGE:(page+1)*PAGE]
            sim_chunks = [page_items[i:i+5] for i in range(0, len(page_items), 5)]
            for chunk in sim_chunks:
                cols = st.columns(5)
                for col, mv in zip(cols, chunk):
                    with col:
                        poster = f"https://image.tmdb.org/t/p/w342{mv['poster_path']}" if mv.get("poster_path") else None
                        vote   = mv.get("vote_average")
                        year   = (mv.get("release_date") or "")[:4]
                        tmdb_b = f'<span class="tmdb-badge">★ {vote:.1f}</span>' if vote else ""
                        img    = f'<img class="movie-poster" src="{poster}">' if poster else '<div class="no-poster">🎬</div>'
                        st.markdown(f"""
                        <div class="movie-card">
                            {img}
                            <div class="card-body">
                                <div class="card-title">{mv['title']}</div>
                                <div class="genre-row"><span class="gtag">{year}</span></div>
                                <div class="card-badges">{tmdb_b}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        iw  = any(w["title"] == mv["title"] for w in st.session_state.wishlist)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("Details", key=f"mvsim_d_{mv['id']}"):
                                show_movie_detail(mv, tmdb_api_key)
                        with c2:
                            lbl = "✓" if iw else "+ Wish"
                            if st.button(lbl, key=f"mvsim_w_{mv['id']}"):
                                if not iw:
                                    st.session_state.wishlist.append({"title": mv["title"], "genres": "Movie"})
                                else:
                                    st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != mv["title"]]
                                st.rerun()
                        with c3:
                            t = get_trailer_url(mv["id"], tmdb_api_key)
                            if t:
                                st.link_button("▶", t, use_container_width=True)
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
                            poster = f"https://image.tmdb.org/t/p/w342{mv['poster_path']}" if mv.get("poster_path") else None
                            vote   = mv.get("vote_average")
                            year   = (mv.get("release_date") or "")[:4]
                            tmdb_b = f'<span class="tmdb-badge">★ {vote:.1f}</span>' if vote else ""
                            img    = f'<img class="movie-poster" src="{poster}">' if poster else '<div class="no-poster">🎬</div>'
                            st.markdown(f"""
                            <div class="movie-card">
                                {img}
                                <div class="card-body">
                                    <div class="card-title">{mv['title']}</div>
                                    <div class="genre-row"><span class="gtag">{year}</span></div>
                                    <div class="card-badges">{tmdb_b}</div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                            iw  = any(w["title"] == mv["title"] for w in st.session_state.wishlist)
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                if st.button("Details", key=f"mvtrd_{mv['id']}"):
                                    show_movie_detail(mv, tmdb_api_key)
                            with c2:
                                lbl = "✓" if iw else "+ Wish"
                                if st.button(lbl, key=f"mvtr_w_{mv['id']}"):
                                    if not iw:
                                        st.session_state.wishlist.append({"title": mv["title"], "genres": "Movie"})
                                    else:
                                        st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != mv["title"]]
                                    st.rerun()
                            with c3:
                                t = get_trailer_url(mv["id"], tmdb_api_key)
                                if t:
                                    st.link_button("▶", t, use_container_width=True)


# ══ TAB 2 — TV Shows ═════════════════════════════════════════════════════════
with tab_tv:
    if not tmdb_api_key:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">📺</div>
            <div class="empty-text">TMDb Key Required</div>
            <div class="empty-sub">Add your TMDb API key in the sidebar to search and browse TV shows.</div>
        </div>""", unsafe_allow_html=True)
    else:
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
            chunks  = [results[i:i+4] for i in range(0, len(results), 4)]
            for chunk in chunks:
                cols = st.columns(4)
                for col, show in zip(cols, chunk):
                    with col:
                        poster = f"https://image.tmdb.org/t/p/w342{show['poster_path']}" if show.get("poster_path") else None
                        vote   = show.get("vote_average")
                        year   = (show.get("first_air_date") or "")[:4]
                        tmdb_b = f'<span class="tmdb-badge">★ {vote:.1f}</span>' if vote else ""
                        img    = f'<img class="movie-poster" src="{poster}">' if poster else '<div class="no-poster">📺<div class="no-poster-lbl">No Poster</div></div>'
                        st.markdown(f"""
                        <div class="movie-card">
                            {img}
                            <div class="card-body">
                                <div class="card-title">{show['name']}</div>
                                <div class="genre-row"><span class="gtag">{year}</span></div>
                                <div class="card-badges">{tmdb_b}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            if st.button("Details", key=f"tvd_{show['id']}"):
                                show_tv_detail(show, tmdb_api_key)
                        with c2:
                            if st.button("Similar", key=f"tvs_{show['id']}"):
                                st.session_state._tv_seed     = show
                                st.session_state._tv_similar  = get_similar_tv(show["id"], tmdb_api_key)
                                st.session_state._tv_sim_page = 0
                        with c3:
                            iw = any(w["title"] == show["name"] for w in st.session_state.wishlist)
                            lbl = "✓" if iw else "+ Wish"
                            if st.button(lbl, key=f"tvw_{show['id']}"):
                                if not iw:
                                    st.session_state.wishlist.append({"title": show["name"], "genres": "TV Show"})
                                else:
                                    st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != show["name"]]
                                st.rerun()
                        with c4:
                            t = get_tv_trailer(show["id"], tmdb_api_key)
                            if t:
                                st.link_button("▶", t, use_container_width=True)

        elif tv_q and not st.session_state._tv_results:
            st.caption("No results found.")

        # ── Similar shows ────────────────────────────────────────────────────
        if st.session_state._tv_similar:
            seed    = st.session_state._tv_seed
            similar = st.session_state._tv_similar
            PAGE    = 10
            page    = min(st.session_state._tv_sim_page, max(0, (len(similar)-1)//PAGE))
            st.markdown(f"""
            <div class="sec-hdr" style="margin-top:1.5rem">
                <span class="sec-hdr-title">Similar to · <em>{seed['name']}</em></span>
                <div class="sec-hdr-line"></div>
                <span class="sec-hdr-count">{len(similar)} shows</span>
            </div>""", unsafe_allow_html=True)
            page_items = similar[page*PAGE:(page+1)*PAGE]
            sim_chunks = [page_items[i:i+5] for i in range(0, len(page_items), 5)]
            for chunk in sim_chunks:
                cols = st.columns(5)
                for col, show in zip(cols, chunk):
                    with col:
                        poster = f"https://image.tmdb.org/t/p/w342{show['poster_path']}" if show.get("poster_path") else None
                        vote   = show.get("vote_average")
                        year   = (show.get("first_air_date") or "")[:4]
                        tmdb_b = f'<span class="tmdb-badge">★ {vote:.1f}</span>' if vote else ""
                        img    = f'<img class="movie-poster" src="{poster}">' if poster else '<div class="no-poster">📺</div>'
                        st.markdown(f"""
                        <div class="movie-card">
                            {img}
                            <div class="card-body">
                                <div class="card-title">{show['name']}</div>
                                <div class="genre-row"><span class="gtag">{year}</span></div>
                                <div class="card-badges">{tmdb_b}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        iw  = any(w["title"] == show["name"] for w in st.session_state.wishlist)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("Details", key=f"tvsimd_{show['id']}"):
                                show_tv_detail(show, tmdb_api_key)
                        with c2:
                            lbl = "✓" if iw else "+ Wish"
                            if st.button(lbl, key=f"tvsim_w_{show['id']}"):
                                if not iw:
                                    st.session_state.wishlist.append({"title": show["name"], "genres": "TV Show"})
                                else:
                                    st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != show["name"]]
                                st.rerun()
                        with c3:
                            t = get_tv_trailer(show["id"], tmdb_api_key)
                            if t:
                                st.link_button("▶", t, use_container_width=True)
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
                            poster = f"https://image.tmdb.org/t/p/w342{show['poster_path']}" if show.get("poster_path") else None
                            vote   = show.get("vote_average")
                            year   = (show.get("first_air_date") or "")[:4]
                            tmdb_b = f'<span class="tmdb-badge">★ {vote:.1f}</span>' if vote else ""
                            img    = f'<img class="movie-poster" src="{poster}">' if poster else '<div class="no-poster">📺</div>'
                            st.markdown(f"""
                            <div class="movie-card">
                                {img}
                                <div class="card-body">
                                    <div class="card-title">{show['name']}</div>
                                    <div class="genre-row"><span class="gtag">{year}</span></div>
                                    <div class="card-badges">{tmdb_b}</div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                            iw  = any(w["title"] == show["name"] for w in st.session_state.wishlist)
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                if st.button("Details", key=f"tvtrd_{show['id']}"):
                                    show_tv_detail(show, tmdb_api_key)
                            with c2:
                                lbl = "✓" if iw else "+ Wish"
                                if st.button(lbl, key=f"tvtr_w_{show['id']}"):
                                    if not iw:
                                        st.session_state.wishlist.append({"title": show["name"], "genres": "TV Show"})
                                    else:
                                        st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != show["name"]]
                                    st.rerun()
                            with c3:
                                t = get_tv_trailer(show["id"], tmdb_api_key)
                                if t:
                                    st.link_button("▶", t, use_container_width=True)

# ══ TAB 3 — Trending ═════════════════════════════════════════════════════════
with tab_trend:
    if not tmdb_api_key:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">🔥</div>
            <div class="empty-text">TMDb Key Required</div>
            <div class="empty-sub">Add your key in the sidebar to see trending movies.</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Loading trending movies..."):
            trending = get_trending(tmdb_api_key)
        if trending:
            st.markdown("""
            <div class="sec-hdr">
                <span class="sec-hdr-title">Trending This Week</span>
                <div class="sec-hdr-line"></div>
            </div>""", unsafe_allow_html=True)
            chunks = [trending[i:i+5] for i in range(0, len(trending), 5)]
            for chunk in chunks:
                cols = st.columns(5)
                for col, m in zip(cols, chunk):
                    with col:
                        poster    = f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get("poster_path") else None
                        vote      = m.get("vote_average")
                        title_raw = m.get("title", "")
                        tmdb_html = f'<span class="tmdb-badge">★ {vote:.1f}</span>' if vote else ""
                        img       = f'<img class="movie-poster" src="{poster}">' if poster else '<div class="no-poster">🎬</div>'
                        st.markdown(f"""
                        <div class="trend-card">
                            {img}
                            <div class="card-body">
                                <div class="card-title">{title_raw}</div>
                                <div class="card-badges">{tmdb_html}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        matches = [t for t in movies["title"] if clean_title(t).lower() == title_raw.lower()]
                        if matches and st.button("Similar", key=f"tr_{title_raw}"):
                            with st.spinner("Computing..."):
                                st.session_state._trend_recs = recommend(matches[0], N_RECS)
                                st.session_state._trend_seed = matches[0]

        if "_trend_recs" in st.session_state and not st.session_state._trend_recs.empty:
            seed = st.session_state.get("_trend_seed", "")
            st.markdown(f"""
            <div class="sec-hdr" style="margin-top:2rem">
                <span class="sec-hdr-title">Similar to · <em>{clean_title(seed)}</em></span>
                <div class="sec-hdr-line"></div>
            </div>""", unsafe_allow_html=True)
            render_grid(st.session_state._trend_recs, tmdb_api_key, show_score=True, prefix="trend")

# ══ TAB 4 — Wishlist ═════════════════════════════════════════════════════════
with tab_wish:
    if not st.session_state.wishlist:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">🔖</div>
            <div class="empty-text">Your wishlist is empty</div>
            <div class="empty-sub">Click "+ Wish" on any movie to save it here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        c1, c2, _ = st.columns([1.5, 1, 5])
        df_wish = pd.DataFrame(st.session_state.wishlist)
        with c1:
            st.download_button("⬇ Export CSV", df_wish.to_csv(index=False),
                               "nextwatch_wishlist.csv", "text/csv", use_container_width=True)
        with c2:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.wishlist = []
                st.rerun()

        st.markdown(f"""
        <div class="sec-hdr" style="margin-top:1rem">
            <span class="sec-hdr-title">My Wishlist</span>
            <div class="sec-hdr-line"></div>
            <span class="sec-hdr-count">{len(st.session_state.wishlist)} movies</span>
        </div>""", unsafe_allow_html=True)

        chunks = [st.session_state.wishlist[i:i+5] for i in range(0, len(st.session_state.wishlist), 5)]
        for chunk in chunks:
            cols = st.columns(5)
            for col, item in zip(cols, chunk):
                with col:
                    poster, tmdb_id, rating = get_movie_info(item["title"], tmdb_api_key) if tmdb_api_key else (None, None, None)
                    st.markdown(card_html(item["title"], item["genres"], poster, rating, in_wish=True), unsafe_allow_html=True)
                    if st.button("✕ Remove", key=f"wr_{item['title']}"):
                        st.session_state.wishlist = [w for w in st.session_state.wishlist if w["title"] != item["title"]]
                        st.rerun()

# ══ TAB 5 — Watched ══════════════════════════════════════════════════════════
with tab_seen:
    if not st.session_state.watched:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">✓</div>
            <div class="empty-text">No movies marked as watched</div>
            <div class="empty-sub">Click "✓ Watched" on any movie to start your journal.</div>
        </div>""", unsafe_allow_html=True)
    else:
        c1, c2, _ = st.columns([1.5, 1, 5])
        df_seen = pd.DataFrame([
            {"title": t, "personal_rating": r,
             "genres": movies[movies["title"] == t]["genres"].iloc[0] if t in movies["title"].values else ""}
            for t, r in st.session_state.watched.items()
        ])
        with c1:
            st.download_button("⬇ Export CSV", df_seen.to_csv(index=False),
                               "nextwatch_watched.csv", "text/csv", use_container_width=True)
        with c2:
            if st.button("🗑 Clear all", use_container_width=True):
                st.session_state.watched = {}
                st.rerun()

        st.markdown(f"""
        <div class="sec-hdr" style="margin-top:1rem">
            <span class="sec-hdr-title">Watched Movies</span>
            <div class="sec-hdr-line"></div>
            <span class="sec-hdr-count">{len(st.session_state.watched)} movies</span>
        </div>""", unsafe_allow_html=True)

        for title, personal_rating in list(st.session_state.watched.items()):
            poster, tmdb_id, tmdb_rating = get_movie_info(title, tmdb_api_key) if tmdb_api_key else (None, None, None)
            img = f'<img class="list-thumb" src="{poster}">' if poster else '<div class="list-thumb-ph">🎬</div>'
            genres_row = movies[movies["title"] == title]["genres"].iloc[0] if title in movies["title"].values else ""
            genre_str  = " · ".join(g for g in genres_row.split("|")[:3] if g and g != "(no genres listed)")
            stars      = "★" * personal_rating + "☆" * (5 - personal_rating) if personal_rating else "☆☆☆☆☆"
            tmdb_html  = f'<span class="tmdb-badge">★ {tmdb_rating:.1f}</span>' if tmdb_rating else ""
            st.markdown(f"""
            <div class="list-row">
                {img}
                <div class="list-info">
                    <div class="list-title">{clean_title(title)}</div>
                    <div class="list-genres">{genre_str}</div>
                </div>
                <div class="list-meta"><span class="star-display">{stars}</span>{tmdb_html}</div>
            </div>""", unsafe_allow_html=True)

            ca, cb, _ = st.columns([1, 1, 5])
            with ca:
                new_r = st.select_slider("Rating", [0,1,2,3,4,5], value=personal_rating,
                                          format_func=lambda x: "★"*x or "—",
                                          key=f"rate_{title}", label_visibility="collapsed")
                if new_r != personal_rating:
                    st.session_state.watched[title] = new_r
                    st.rerun()
            with cb:
                if st.button("✕ Remove", key=f"sr_{title}"):
                    del st.session_state.watched[title]
                    st.rerun()

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
                render_grid(recs_from_seen, tmdb_api_key, show_score=True, prefix="seen")
