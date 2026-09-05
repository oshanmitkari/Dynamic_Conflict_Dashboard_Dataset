
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Conflict Discourse Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme -----------------------------
COLORS = {
    "white": "#FFFFFF",
    "cream": "#FFF4CC",
    "yellow": "#FFE9A8",
    "blue": "#DDF3FF",
    "sky": "#BFE7FA",
    "pink": "#FADCE8",
    "pink2": "#F7C8D8",
    "text": "#263238",
    "muted": "#607D8B",
    "border": "#E6EDF2",
}
SENTIMENT_COLORS = {
    "Negative": COLORS["pink2"],
    "Neutral": COLORS["cream"],
    "Positive": COLORS["blue"],
}

# ----------------------------- Polished UI theme -----------------------------
st.markdown(
    """
    <style>
    /* Main canvas */
    .stApp { background: #FFFFFF !important; }
    .main .block-container {
        max-width: 1500px !important;
        padding: 1.25rem 2rem 3rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #EEF9FF 100%) !important;
        border-right: 1px solid #DCEAF1 !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem !important;
    }

    /* Typography */
    h1, h2, h3, h4, p, label, .stMarkdown {
        color: #263238 !important;
    }
    h1 { letter-spacing: -0.035em !important; }
    h2, h3 { letter-spacing: -0.02em !important; }

    /* Hero */
    .hero {
        background: linear-gradient(115deg, #FFFFFF 0%, #FFF7D8 45%, #E5F6FF 100%);
        border: 1px solid #DCE8EE;
        border-radius: 22px;
        padding: 30px 34px;
        margin: 0 0 20px 0;
        box-shadow: 0 8px 26px rgba(38,50,56,.055);
    }
    .hero h1 {
        font-size: 2.55rem !important;
        line-height: 1.08 !important;
        margin: 0 0 8px !important;
    }
    .hero p {
        font-size: 1.03rem !important;
        color: #607D8B !important;
        margin: 0 !important;
    }

    /* KPI cards */
    .kpi {
        background: #FFFFFF;
        border: 1px solid #DCE8EE;
        border-radius: 15px;
        padding: 15px 16px;
        min-height: 92px;
        box-shadow: 0 5px 18px rgba(38,50,56,.055);
    }
    .kpi-label {
        color: #607D8B !important;
        font-size: .78rem !important;
        font-weight: 650 !important;
    }
    .kpi-value {
        color: #263238 !important;
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        margin-top: 6px !important;
    }

    /* Insight / methodology cards */
    .insight {
        background: #F1FAFF;
        border: 1px solid #D6EDF8;
        border-left: 5px solid #BFE7FA;
        border-radius: 11px;
        padding: 11px 15px;
        margin: 7px 0;
    }
    .method {
        background: #FFF8DD;
        border: 1px solid #F0E1A4;
        border-radius: 12px;
        padding: 14px 16px;
    }

    /* Streamlit widgets */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #DCE8EE !important;
        border-radius: 14px !important;
        padding: 10px 14px !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #DCE8EE !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px !important;
        border: 1px solid #B8DDED !important;
        background: #E8F7FF !important;
        color: #263238 !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #DDF3FF !important;
        border-color: #9ED3E8 !important;
    }

    /* Prevent chart containers from becoming cramped */
    div[data-testid="stPlotlyChart"] {
        width: 100% !important;
    }

    /* Tabs / radio */
    div[role="radiogroup"] label {
        font-weight: 600 !important;
    }

    /* Mobile/tablet */
    @media (max-width: 900px) {
        .main .block-container { padding: 1rem !important; }
        .hero { padding: 22px !important; }
        .hero h1 { font-size: 2rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Data -----------------------------
BASE = Path(__file__).parent
RAW_PATH = BASE / "data" / "raw_dataset.csv"
CLEAN_PATH = BASE / "data" / "cleaned_dataset.csv"

@st.cache_data
def load_data():
    raw = pd.read_csv(RAW_PATH)
    clean = pd.read_csv(CLEAN_PATH)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    clean["Polarity"] = pd.to_numeric(clean["Polarity"], errors="coerce")
    clean["Subjectivity"] = pd.to_numeric(clean["Subjectivity"], errors="coerce")
    df = raw.copy()
    # Align by row position because the supplied files contain the same 10,500 records.
    for col in ["Index", "Clean Text", "Polarity", "Subjectivity", "Analysis", "Label"]:
        df[col] = clean[col].values
    return df, raw, clean

df, raw, clean = load_data()

# ----------------------------- Helpers -----------------------------
def fmt_num(x):
    if pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"

def kpi(label, value):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

def style_fig(fig, height=410):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF3F6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF3F6", zeroline=False)
    return fig

def export_csv(data):
    return data.to_csv(index=False).encode("utf-8")

# ----------------------------- Sidebar -----------------------------
st.sidebar.markdown("## 📊 Conflict Discourse")
st.sidebar.caption("Interactive Data Science Dashboard")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sentiment Analysis",
        "Temporal Analysis",
        "Engagement Analysis",
        "Topic Analysis",
        "Source Analysis",
        "Statistical Analysis",
        "Tweet Explorer",
        "Dataset & Methodology",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Global Filters")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d0, d1 = pd.Timestamp(min_date), pd.Timestamp(max_date)

sentiments = st.sidebar.multiselect(
    "Sentiment",
    sorted(df["Analysis"].dropna().unique()),
    default=sorted(df["Analysis"].dropna().unique()),
)
topics = st.sidebar.multiselect(
    "Search topic",
    sorted(df["Search"].dropna().unique()),
    default=[],
)
sources = st.sidebar.multiselect(
    "Source label",
    sorted(df["Source Label"].dropna().unique()),
    default=[],
)
like_min, like_max = int(df["Like Count"].min()), int(df["Like Count"].max())
rt_min, rt_max = int(df["Retweet Count"].min()), int(df["Retweet Count"].max())
likes = st.sidebar.slider("Minimum likes", like_min, like_max, like_min)
retweets = st.sidebar.slider("Minimum retweets", rt_min, rt_max, rt_min)

if st.sidebar.button("Reset filters", use_container_width=True):
    st.rerun()

mask = (
    df["Date"].between(d0, d1)
    & df["Analysis"].isin(sentiments)
    & (df["Like Count"] >= likes)
    & (df["Retweet Count"] >= retweets)
)
if topics:
    mask &= df["Search"].isin(topics)
if sources:
    mask &= df["Source Label"].isin(sources)
f = df.loc[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Records selected", f"{len(f):,}")
st.sidebar.caption("All analytics are recalculated from the selected records.")

# ----------------------------- Hero -----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Conflict Discourse Analytics</h1>
      <p>US–Israel–Iran Social Media Sentiment & Engagement Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(f"<div style=\"color:#607D8B;font-size:.88rem;margin:4px 0 18px\"><b>{len(f):,}</b> records selected • All charts update dynamically with the filters</div>", unsafe_allow_html=True)

# ----------------------------- Overview -----------------------------
if page == "Overview":
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">A high-level view of the selected portion of the curated dataset.</div>', unsafe_allow_html=True)

    avg_pol = f["Polarity"].mean()
    cols = st.columns(8)
    vals = [
        ("Total Posts", f"{len(f):,}"),
        ("Negative", f"{(f.Analysis=='Negative').sum():,}"),
        ("Neutral", f"{(f.Analysis=='Neutral').sum():,}"),
        ("Positive", f"{(f.Analysis=='Positive').sum():,}"),
        ("Avg Likes", fmt_num(f["Like Count"].mean())),
        ("Avg Retweets", fmt_num(f["Retweet Count"].mean())),
        ("Avg Replies", fmt_num(f["Reply Count"].mean())),
        ("Avg Polarity", f"{avg_pol:.3f}" if pd.notna(avg_pol) else "—"),
    ]
    for c, (lab, val) in zip(cols, vals):
        with c:
            kpi(lab, val)

    c1, c2 = st.columns(2)
    with c1:
        s = f["Analysis"].value_counts().reindex(["Negative","Neutral","Positive"]).fillna(0).reset_index()
        s.columns = ["Sentiment","Count"]
        fig = px.pie(s, names="Sentiment", values="Count", hole=.62, title="Sentiment Distribution",
                     color="Sentiment", color_discrete_map=SENTIMENT_COLORS)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c2:
        daily = f.groupby(["Date","Analysis"]).size().reset_index(name="Posts")
        fig = px.line(daily, x="Date", y="Posts", color="Analysis", markers=True,
                      title="Sentiment Volume Over Time", color_discrete_map=SENTIMENT_COLORS)
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        top = f["Search"].value_counts().head(12).sort_values().reset_index()
        top.columns = ["Search","Posts"]
        fig = px.bar(top, x="Posts", y="Search", orientation="h", title="Top Search Topics",
                     color_discrete_sequence=[COLORS["sky"]])
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    with c4:
        fig = px.scatter(
            f.sample(min(2500, len(f)), random_state=7) if len(f) > 2500 else f,
            x="Polarity", y="Like Count", size="Retweet Count", color="Analysis",
            hover_data=["Date","Search","Tweet"], opacity=.65,
            title="Polarity vs Likes",
            color_discrete_map=SENTIMENT_COLORS,
        )
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.markdown("### Key observations")
    if len(f):
        neg_pct = (f.Analysis.eq("Negative").mean() * 100)
        top_topic = f.Search.value_counts().idxmax()
        peak_date = f.Date.dt.date.value_counts().idxmax()
        top_eng_topic = f.groupby("Search")["Like Count"].mean().idxmax()
        for text in [
            f"Negative sentiment accounts for **{neg_pct:.1f}%** of the currently selected posts.",
            f"**{top_topic}** is the most frequent search topic in the current selection.",
            f"Post volume peaks on **{peak_date}** within the current selection.",
            f"**{top_eng_topic}** has the highest average likes among the selected topics.",
        ]:
            st.markdown(f'<div class="insight">• {text}</div>', unsafe_allow_html=True)

# ----------------------------- Sentiment -----------------------------
elif page == "Sentiment Analysis":
    st.markdown('<div class="section-title">Sentiment Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Distribution, intensity, subjectivity, and sentiment by topic/source.</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for c, (lab, val) in zip(cols, [
        ("Negative", (f.Analysis=="Negative").sum()),
        ("Neutral", (f.Analysis=="Neutral").sum()),
        ("Positive", (f.Analysis=="Positive").sum()),
        ("Mean Polarity", f.Polarity.mean()),
        ("Mean Subjectivity", f.Subjectivity.mean()),
    ]):
        with c:
            kpi(lab, f"{val:,.3f}" if isinstance(val, float) else f"{val:,}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(f, x="Polarity", nbins=30, title="Polarity Distribution")
        fig.add_vline(x=0, line_dash="dash", line_width=1)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.histogram(f, x="Subjectivity", nbins=30, title="Subjectivity Distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            f.sample(min(3000,len(f)), random_state=4) if len(f)>3000 else f,
            x="Polarity", y="Subjectivity", color="Analysis",
            hover_data=["Tweet","Search"], opacity=.65,
            title="Polarity vs Subjectivity", color_discrete_map=SENTIMENT_COLORS,
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        tab = pd.crosstab(f["Search"], f["Analysis"], normalize="index").reindex(columns=["Negative","Neutral","Positive"], fill_value=0)
        tab = tab.sort_values("Negative", ascending=False).head(15)
        fig = px.bar(tab.reset_index(), x="Search", y=["Negative","Neutral","Positive"],
                     title="Sentiment Composition by Topic", barmode="stack",
                     color_discrete_map=SENTIMENT_COLORS)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    src = pd.crosstab(f["Source Label"], f["Analysis"]).reindex(columns=["Negative","Neutral","Positive"], fill_value=0)
    fig = px.bar(src.reset_index(), x="Source Label", y=["Negative","Neutral","Positive"],
                 title="Sentiment by Source Label", barmode="stack",
                 color_discrete_map=SENTIMENT_COLORS)
    st.plotly_chart(style_fig(fig, 430), use_container_width=True)

# ----------------------------- Temporal -----------------------------
elif page == "Temporal Analysis":
    st.markdown('<div class="section-title">Temporal Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">How volume, sentiment, and engagement vary across the observed dates.</div>', unsafe_allow_html=True)

    daily = f.groupby("Date").agg(
        Posts=("Tweet","size"),
        Likes=("Like Count","sum"),
        Retweets=("Retweet Count","sum"),
        Replies=("Reply Count","sum"),
        AvgPolarity=("Polarity","mean"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.area(daily, x="Date", y="Posts", title="Daily Post Volume")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.line(daily, x="Date", y="AvgPolarity", markers=True, title="Average Polarity by Date")
        fig.add_hline(y=0, line_dash="dash", line_width=1)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    sent = f.groupby(["Date","Analysis"]).size().reset_index(name="Posts")
    fig = px.line(sent, x="Date", y="Posts", color="Analysis", markers=True,
                  title="Daily Sentiment Volume", color_discrete_map=SENTIMENT_COLORS)
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        e = daily.melt("Date", value_vars=["Likes","Retweets","Replies"], var_name="Metric", value_name="Count")
        fig = px.line(e, x="Date", y="Count", color="Metric", title="Engagement Over Time")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        heat = f.assign(Day=f.Date.dt.date).groupby("Day").size().reset_index(name="Posts")
        heat["Week"] = pd.to_datetime(heat["Day"]).dt.isocalendar().week.astype(int)
        heat["Weekday"] = pd.to_datetime(heat["Day"]).dt.day_name()
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = heat.pivot_table(index="Weekday", columns="Week", values="Posts", aggfunc="sum", fill_value=0).reindex(order)
        fig = px.imshow(pivot, aspect="auto", title="Posting Activity Heatmap",
                        color_continuous_scale=["#FFFFFF", COLORS["sky"], COLORS["blue"]])
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

# ----------------------------- Engagement -----------------------------
elif page == "Engagement Analysis":
    st.markdown('<div class="section-title">Engagement Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Observed likes, retweets, and replies across sentiment, topics, and individual posts.</div>', unsafe_allow_html=True)

    cols = st.columns(6)
    for c,(lab,val) in zip(cols,[
        ("Total Likes",f["Like Count"].sum()),
        ("Total Retweets",f["Retweet Count"].sum()),
        ("Total Replies",f["Reply Count"].sum()),
        ("Avg Likes",f["Like Count"].mean()),
        ("Avg Retweets",f["Retweet Count"].mean()),
        ("Avg Replies",f["Reply Count"].mean()),
    ]):
        with c: kpi(lab, fmt_num(val))

    c1,c2 = st.columns(2)
    with c1:
        av = f.groupby("Analysis")[["Like Count","Retweet Count","Reply Count"]].mean().reindex(["Negative","Neutral","Positive"]).reset_index()
        long = av.melt("Analysis", var_name="Metric", value_name="Average")
        fig = px.bar(long, x="Analysis", y="Average", color="Metric", barmode="group", title="Average Engagement by Sentiment")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.box(f, x="Analysis", y="Like Count", color="Analysis", title="Like Distribution by Sentiment",
                     color_discrete_map=SENTIMENT_COLORS)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("### Most engaged posts")
    rank = st.selectbox("Rank by", ["Like Count","Retweet Count","Reply Count"])
    cols_show = ["Date","Tweet","Like Count","Retweet Count","Reply Count","Analysis","Search","Source Label"]
    top_posts = f.sort_values(rank, ascending=False)[cols_show].head(20).copy()
    top_posts["Date"] = top_posts["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(top_posts, use_container_width=True, hide_index=True, height=520)

    st.caption("Engagement associations are descriptive; they should not be interpreted as causal effects.")

# ----------------------------- Topic -----------------------------
elif page == "Topic Analysis":
    st.markdown('<div class="section-title">Topic & Discourse Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">The Search field is treated as the topic/query category recorded in the dataset.</div>', unsafe_allow_html=True)

    topn = st.slider("Number of topics to display", 5, 20, 10)
    counts = f["Search"].value_counts().head(topn).sort_values().reset_index()
    counts.columns = ["Search","Posts"]
    fig = px.bar(counts, x="Posts", y="Search", orientation="h", title=f"Top {topn} Search Categories")
    st.plotly_chart(style_fig(fig, 450), use_container_width=True)

    topic_sent = pd.crosstab(f["Search"], f["Analysis"]).reindex(columns=["Negative","Neutral","Positive"], fill_value=0)
    topic_sent = topic_sent.loc[f["Search"].value_counts().head(topn).index]
    fig = px.bar(topic_sent.reset_index(), x="Search", y=["Negative","Neutral","Positive"],
                 title="Topic Sentiment Composition", barmode="stack",
                 color_discrete_map=SENTIMENT_COLORS)
    fig.update_xaxes(tickangle=-35)
    st.plotly_chart(style_fig(fig, 450), use_container_width=True)

    topic_metrics = f.groupby("Search").agg(
        Posts=("Tweet","size"),
        AvgLikes=("Like Count","mean"),
        AvgRetweets=("Retweet Count","mean"),
        AvgReplies=("Reply Count","mean"),
        AvgPolarity=("Polarity","mean"),
    ).sort_values("Posts", ascending=False)
    st.dataframe(topic_metrics.round(3), use_container_width=True)

    selected = st.selectbox("Explore one topic", ["All topics"] + sorted(f["Search"].dropna().unique()))
    if selected != "All topics":
        t = f[f.Search == selected]
        cols = st.columns(5)
        for c,(lab,val) in zip(cols,[
            ("Posts",len(t)),("Avg Likes",t["Like Count"].mean()),
            ("Avg Retweets",t["Retweet Count"].mean()),
            ("Avg Replies",t["Reply Count"].mean()),("Avg Polarity",t.Polarity.mean())
        ]):
            with c: kpi(lab, f"{val:,.3f}" if isinstance(val,float) else f"{val:,}")
        st.dataframe(
            t.sort_values("Like Count", ascending=False)[
                ["Date","Tweet","Like Count","Retweet Count","Reply Count","Analysis"]
            ].head(15),
            use_container_width=True, hide_index=True
        )

# ----------------------------- Source -----------------------------
elif page == "Source Analysis":
    st.markdown('<div class="section-title">Source Label Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Analysis of the Source Label field as recorded in the supplied dataset.</div>', unsafe_allow_html=True)

    counts = f["Source Label"].value_counts().sort_values().reset_index()
    counts.columns = ["Source Label","Posts"]
    fig = px.bar(counts, x="Posts", y="Source Label", orientation="h", title="Source Label Distribution")
    st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    sent = pd.crosstab(f["Source Label"], f["Analysis"]).reindex(columns=["Negative","Neutral","Positive"], fill_value=0)
    fig = px.bar(sent.reset_index(), x="Source Label", y=["Negative","Neutral","Positive"],
                 title="Sentiment by Source Label", barmode="stack",
                 color_discrete_map=SENTIMENT_COLORS)
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    metrics = f.groupby("Source Label").agg(
        Posts=("Tweet","size"),
        AvgLikes=("Like Count","mean"),
        AvgRetweets=("Retweet Count","mean"),
        AvgReplies=("Reply Count","mean"),
        AvgPolarity=("Polarity","mean"),
    ).sort_values("AvgLikes", ascending=False)
    st.dataframe(metrics.round(3), use_container_width=True)

    st.markdown(
        '<div class="method"><b>Interpretation note:</b> Source Label is analyzed exactly as recorded. '
        'It should not be treated as proof of the real-world platform, device, or collection mechanism unless '
        'the dataset documentation establishes that provenance.</div>',
        unsafe_allow_html=True,
    )

# ----------------------------- Statistics -----------------------------
elif page == "Statistical Analysis":
    st.markdown('<div class="section-title">Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Descriptive statistics and relationships among numeric variables.</div>', unsafe_allow_html=True)

    numeric = ["Reply Count","Retweet Count","Like Count","Polarity","Subjectivity"]
    desc = f[numeric].describe().T
    desc["median"] = f[numeric].median()
    desc["variance"] = f[numeric].var()
    desc["IQR"] = f[numeric].quantile(.75) - f[numeric].quantile(.25)
    desc["mode"] = f[numeric].mode().iloc[0]
    desc = desc[["count","mean","std","variance","min","25%","median","50%","75%","max","IQR","mode"]]
    desc.columns = ["Count","Mean","Std Dev","Variance","Min","Q1","Median","50%","Q3","Max","IQR","Mode"]
    st.dataframe(desc.round(4), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        metric = st.selectbox("Distribution", numeric)
        fig = px.box(f, y=metric, points="outliers", title=f"{metric} — Box Plot")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    with c2:
        fig = px.histogram(f, x=metric, nbins=35, title=f"{metric} — Histogram")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    corr = f[numeric].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale=["#FFFFFF", COLORS["pink"], COLORS["sky"]],
                    title="Correlation Matrix")
    st.plotly_chart(style_fig(fig, 500), use_container_width=True)
    st.caption("Correlation indicates association, not causation.")

# ----------------------------- Explorer -----------------------------
elif page == "Tweet Explorer":
    st.markdown('<div class="section-title">Tweet Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Search, inspect, sort, and export the currently filtered records.</div>', unsafe_allow_html=True)

    query = st.text_input("Search tweet text", placeholder="Type a word, phrase, topic, or keyword...")
    show = f.copy()
    if query.strip():
        q = query.strip().lower()
        show = show[show["Tweet"].str.lower().str.contains(re.escape(q), na=False)]

    page_size = st.selectbox("Rows per page", [10,25,50,100], index=1)
    max_pages = max(1, int(np.ceil(len(show)/page_size)))
    pno = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1)
    subset = show.iloc[(pno-1)*page_size:pno*page_size].copy()

    cols = ["Date","Tweet","Like Count","Retweet Count","Reply Count","Search","Source Label","Polarity","Subjectivity","Analysis"]
    subset["Date"] = subset["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(subset[cols], use_container_width=True, hide_index=True, height=560)

    st.download_button(
        "⬇️ Export filtered records as CSV",
        data=export_csv(show),
        file_name="filtered_conflict_discourse_data.csv",
        mime="text/csv",
    )

# ----------------------------- Dataset -----------------------------
else:
    st.markdown('<div class="section-title">Dataset & Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Transparent documentation of the supplied data and analytical workflow.</div>', unsafe_allow_html=True)

    health = {
        "Raw rows": len(raw),
        "Cleaned rows": len(clean),
        "Raw columns": raw.shape[1],
        "Cleaned columns": clean.shape[1],
        "Date coverage": f"{raw.Date.min().date()} → {raw.Date.max().date()}",
        "Unique dates": raw.Date.dt.date.nunique(),
        "Search categories": raw["Search"].nunique(),
        "Source labels": raw["Source Label"].nunique(),
        "Languages": raw["Language"].nunique(),
        "Exact duplicate rows": int(raw.duplicated().sum()),
        "Duplicate tweet texts": int(raw["Tweet"].duplicated().sum()),
    }
    cols = st.columns(5)
    for i,(lab,val) in enumerate(health.items()):
        with cols[i%5]:
            kpi(lab, str(val))

    st.markdown("### Field definitions")
    definitions = pd.DataFrame([
        ["Date","Raw","Date associated with the record."],
        ["Tweet","Raw","Original text field supplied with the dataset."],
        ["Reply Count","Raw","Reply count recorded for the post."],
        ["Retweet Count","Raw","Retweet count recorded for the post."],
        ["Like Count","Raw","Like count recorded for the post."],
        ["Language","Raw","Language value recorded in the dataset."],
        ["Source Label","Raw","Source label recorded in the dataset."],
        ["Search","Raw","Search/query category associated with the record."],
        ["Clean Text","Cleaned","Text after the supplied cleaning process."],
        ["Polarity","Cleaned","Sentiment polarity score supplied in the cleaned dataset."],
        ["Subjectivity","Cleaned","Subjectivity score supplied in the cleaned dataset."],
        ["Analysis","Cleaned","Categorical sentiment supplied in the cleaned dataset."],
        ["Label","Cleaned","Numeric sentiment label supplied in the cleaned dataset."],
    ], columns=["Field","Layer","Meaning"])
    st.dataframe(definitions, use_container_width=True, hide_index=True)

    st.markdown("### Analytical pipeline")
    st.markdown(
        """
        <div class="method">
        <b>Raw Dataset</b> → Validation → Supplied Clean Text / Sentiment Fields → 
        Descriptive Statistics → Temporal Analysis → Engagement Analysis → Topic & Source Analysis → 
        Interactive Visualization → Export
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dataset integrity notes")
    st.write(
        "The supplied raw and cleaned files both contain 10,500 records. "
        "The dashboard preserves the original files and computes dashboard metrics dynamically. "
        "The raw file contains no exact duplicate rows, although the Tweet text itself is repeated across records; "
        "this is reported rather than silently removed."
    )
    st.warning(
        "This dashboard describes patterns within the supplied curated dataset. "
        "It should not be presented as a representative sample of all social-media users or as evidence of geopolitical causality."
    )

st.markdown("---")
st.caption("Conflict Discourse Analytics • Curated 10,500-record dataset • Built for transparent exploratory data analysis")
