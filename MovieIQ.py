import ast
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MovieIQ | Film Success Studio", page_icon="🎞️", layout="wide")

# ---------------------------------------------------------------------------
# Custom look & feel (distinct indigo/gold identity, card-based layout)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0B0C14; }
    .hero {
        background: linear-gradient(120deg, #5B4FE9 0%, #2E2A6B 60%, #0B0C14 100%);
        padding: 2.2rem 2rem 1.6rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
    }
    .hero h1 { color: #FFFFFF; margin-bottom: 0.2rem; font-size: 2.1rem; }
    .hero p { color: #DCD9FF; margin: 0; font-size: 0.95rem; }
    .card {
        background-color: #171826;
        border: 1px solid #2A2C3B;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #171826;
        border-left: 4px solid #F2B705;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-box .value { font-size: 1.6rem; font-weight: 700; color: #F2B705; }
    .metric-box .label { font-size: 0.8rem; color: #B8B9C6; text-transform: uppercase; letter-spacing: 0.04em; }
    section[data-testid="stSidebar"] { background-color: #12131A; border-right: 1px solid #2A2C3B; }
    div[data-testid="stMetricValue"] { color: #F2B705; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    def extract_genre(g):
        try:
            parsed = ast.literal_eval(g)
            return parsed[0]["name"] if parsed else "Unknown"
        except (ValueError, SyntaxError):
            return "Unknown"

    df["genre"] = df["genres"].apply(extract_genre)
    return df


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


df = load_data()
bundle = load_model()
model = bundle["model"]
features = bundle["features"]
metrics = bundle["metrics"]
stats_results = bundle["stats_results"]

# ---------------------------------------------------------------------------
# Sidebar: branding + navigation + filters (instead of top tabs)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🎞️ MovieIQ")
st.sidebar.caption("Film Success Studio")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Explore the Data", "🧪 Statistical Tests", "🌲 Model Performance", "🎯 Predict a Movie"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filter the dataset**")
genres_available = sorted(df["genre"].unique())
selected_genres = st.sidebar.multiselect("Genre(s)", genres_available, default=genres_available)
min_vote = st.sidebar.slider("Minimum vote average", 0.0, 10.0, 0.0, 0.1)

filtered = df[df["genre"].isin(selected_genres) & (df["vote_average"] >= min_vote)]
st.sidebar.markdown(f"**{len(filtered)}** movies match")
st.sidebar.markdown("---")
st.sidebar.caption("Built by Mahendar Reddy Maram")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎞️ MovieIQ — Film Success Studio</h1>
        <p>Predicting whether a film clears its budget, powered by a Random Forest model.
        A movie is labeled <b>successful</b> when revenue&nbsp;&gt;&nbsp;budget.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "📊 Overview":
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Movies (filtered)", "Success Rate", "Avg Budget", "Avg Revenue"],
        [
            f"{len(filtered)}",
            f"{filtered['success'].mean() * 100:.1f}%" if len(filtered) else "—",
            f"${filtered['budget'].mean():,.0f}" if len(filtered) else "—",
            f"${filtered['revenue'].mean():,.0f}" if len(filtered) else "—",
        ],
    ):
        col.markdown(
            f'<div class="metric-box"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Filtered movie list**")
    st.dataframe(
        filtered[["title", "genre", "budget", "revenue", "popularity", "runtime", "vote_average", "success"]],
        use_container_width=True,
        height=350,
    )
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data", csv, "filtered_movies.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔍 Explore the Data":
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Budget vs Revenue**")
        st.image("assets/budget_vs_revenue.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Success Rate by Genre**")
        st.image("assets/success_by_genre.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Correlation Heatmap**")
    st.image("assets/correlation_heatmap.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "🧪 Statistical Tests":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**T-Test — vote_average vs success**")
    st.write(f"t-statistic = **{stats_results['t_stat']:.3f}**, p-value = **{stats_results['p_val_t']:.4f}**")
    if stats_results["p_val_t"] < 0.05:
        st.success("p < 0.05 → statistically significant difference in vote_average between successful and unsuccessful movies.")
    else:
        st.info("p ≥ 0.05 → no statistically significant difference in vote_average between the two groups.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Chi-Square Test — genre vs success**")
    st.write(f"chi² = **{stats_results['chi2']:.3f}**, p-value = **{stats_results['p_val_chi2']:.4f}**")
    if stats_results["p_val_chi2"] < 0.05:
        st.success("p < 0.05 → genre is significantly associated with success.")
    else:
        st.info("p ≥ 0.05 → no significant association found between genre and success in this dataset.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "🌲 Model Performance":
    c1, c2, c3 = st.columns(3)
    for col, label, value in zip(
        [c1, c2, c3],
        ["Accuracy", "Precision", "Recall"],
        [f"{metrics['accuracy']*100:.1f}%", f"{metrics['precision']*100:.1f}%", f"{metrics['recall']*100:.1f}%"],
    ):
        col.markdown(
            f'<div class="metric-box"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Confusion Matrix**")
        st.image("assets/confusion_matrix.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Feature Importance**")
        st.image("assets/feature_importance.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🎯 Predict a Movie":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Enter movie features to predict success**")
    budget_in = st.number_input("Budget (USD)", min_value=100000, max_value=500000000, value=100000000, step=100000)
    popularity_in = st.slider("Popularity", 0.0, 100.0, 50.0)
    runtime_in = st.slider("Runtime (minutes)", 60, 220, 120)
    vote_in = st.slider("Vote Average", 0.0, 10.0, 6.0)

    if st.button("🎬 Predict", type="primary"):
        X_new = pd.DataFrame([[budget_in, popularity_in, runtime_in, vote_in]], columns=features)
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0][1]
        if pred == 1:
            st.success(f"🎉 Predicted: **Success** (confidence {proba*100:.1f}%)")
        else:
            st.error(f"📉 Predicted: **Not Successful** (confidence {(1-proba)*100:.1f}%)")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("MovieIQ · Random Forest · Streamlit")
