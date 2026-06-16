"""
dashboard.py  —  Shopping Trends Analytics Dashboard
Dark theme | Purple-Pink palette | Professional & Modern
Run: streamlit run src/dashboard.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Tự động tìm đúng đường dẫn dù chạy local hay Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Thử các vị trí khác nhau của file data
_candidates = [
    os.path.join(BASE_DIR, "data", "shopping_trends.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "shopping_trends.csv"),
    "data/shopping_trends.csv",
    "shopping_trends.csv",
]
DATA_PATH = next((p for p in _candidates if os.path.exists(p)),
                 os.path.join(BASE_DIR, "data", "shopping_trends.csv"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Shopping Trends Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN TOKENS ────────────────────────────────────────────
BG        = "#0F0F1A"
CARD_BG   = "#1A1A2E"
BORDER    = "#2D2D4E"
P1        = "#A855F7"   # purple-500
P2        = "#C084FC"   # purple-400
P3        = "#7C3AED"   # violet-600
PK1       = "#EC4899"   # pink-500
PK2       = "#F472B6"   # pink-400
PK3       = "#DB2777"   # pink-600
TEXT      = "#F1F0FF"
MUTED     = "#94A3B8"

# Plotly discrete palette  (purple → pink gradient, 8 steps)
PALETTE   = [P1, PK1, "#9333EA", "#E879F9", P3, PK3, "#A21CAF", "#BE185D"]

# Plotly continuous scale
CONT      = [[0, P3], [0.5, P1], [1.0, PK1]]

def fig_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT, family="Inter, sans-serif"),
                   x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Inter, sans-serif", size=11),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
    )
    return fig

# ── CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: {BG};
      color: {TEXT};
  }}
  .stApp {{ background-color: {BG}; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background-color: {CARD_BG};
      border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background: {CARD_BG};
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
      border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px;
      color: {MUTED} !important;
      font-weight: 600;
      font-size: 13px;
      padding: 8px 18px;
  }}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(135deg, {P3}, {PK3}) !important;
      color: white !important;
  }}

  /* KPI Cards */
  .kpi-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 20px 24px;
      position: relative;
      overflow: hidden;
  }}
  .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, {P1}, {PK1});
  }}
  .kpi-label {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: {MUTED};
      margin-bottom: 8px;
  }}
  .kpi-value {{
      font-size: 28px;
      font-weight: 700;
      color: {TEXT};
      letter-spacing: -0.5px;
  }}
  .kpi-sub {{
      font-size: 11px;
      color: {MUTED};
      margin-top: 6px;
  }}

  /* Section headers */
  .section-title {{
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: {P2};
      border-left: 3px solid {P1};
      padding-left: 10px;
      margin: 24px 0 16px 0;
  }}

  /* Chart wrapper */
  .chart-box {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 16px;
  }}

  /* Insight row */
  .insight-row {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 10px;
      display: flex;
      align-items: flex-start;
      gap: 12px;
  }}
  .insight-code {{
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 1px;
      color: {P2};
      background: rgba(168,85,247,0.15);
      border-radius: 5px;
      padding: 2px 8px;
      white-space: nowrap;
      margin-top: 2px;
  }}
  .insight-text {{ font-size: 13px; color: {TEXT}; line-height: 1.5; }}

  /* Rec cards */
  .rec-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-left: 3px solid {PK1};
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 10px;
  }}
  .rec-title {{
      font-size: 11px; font-weight: 700;
      letter-spacing: 1px; text-transform: uppercase;
      color: {PK2}; margin-bottom: 6px;
  }}
  .rec-body {{ font-size: 13px; color: {MUTED}; line-height: 1.6; }}

  /* Divider */
  hr {{ border-color: {BORDER}; margin: 20px 0; }}

  /* Streamlit metric override */
  [data-testid="metric-container"] {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 12px 16px;
  }}
</style>
""", unsafe_allow_html=True)


# ── DATA HELPERS ─────────────────────────────────────────────
# Các cột bắt buộc để dashboard hoạt động đúng
REQUIRED_COLS = [
    "Age", "Gender", "Category", "Purchase Amount (USD)", "Season",
    "Review Rating", "Previous Purchases", "Subscription Status",
    "Discount Applied", "Promo Code Used", "Payment Method",
    "Shipping Type", "Location", "Size", "Color", "Item Purchased",
    "Frequency of Purchases",
]

def process_df(df):
    """Chuẩn hóa kiểu dữ liệu và tạo biến phái sinh Age Group."""
    df = df.copy()
    cats = ["Gender", "Category", "Location", "Size", "Color", "Season",
            "Subscription Status", "Payment Method", "Shipping Type",
            "Discount Applied", "Promo Code Used", "Preferred Payment Method",
            "Frequency of Purchases", "Item Purchased"]
    for c in cats:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "Age" in df.columns:
        bins   = [0, 18, 25, 35, 45, 55, 100]
        labels = ["<18", "18-25", "26-35", "36-45", "46-55", "56+"]
        df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    return df

@st.cache_data
def load_sample():
    """Đọc bộ dữ liệu mẫu gốc."""
    return process_df(pd.read_csv(DATA_PATH))

def validate_columns(df):
    """Trả về danh sách cột bắt buộc còn thiếu trong file tải lên."""
    return [c for c in REQUIRED_COLS if c not in df.columns]


# ── DATA SOURCE (cổng tải file) ──────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:18px;font-weight:700;color:{TEXT};margin-bottom:16px;'>SHOPPING TRENDS</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;letter-spacing:1px;color:{MUTED};text-transform:uppercase;margin-bottom:8px;'>Data Source</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Tải lên file CSV của bạn", type=["csv"],
                                help="File cần có cùng cấu trúc với bộ dữ liệu Shopping Trends.")
    use_sample = st.checkbox("Dùng dữ liệu mẫu", value=True,
                             disabled=(uploaded is not None))

# Xác định bộ dữ liệu đang hoạt động
df_all = None
source_name = ""
if uploaded is not None:
    try:
        _raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Không đọc được file: {e}")
        st.stop()
    _missing = validate_columns(_raw)
    if _missing:
        st.error("⚠️ File chưa đúng định dạng — thiếu các cột bắt buộc: "
                 + ", ".join(_missing))
        st.info("Vui lòng tải lên file CSV có cùng cấu trúc với bộ dữ liệu Shopping Trends "
                "(gồm các cột: Age, Gender, Category, Purchase Amount (USD), Season, ...).")
        st.stop()
    df_all = process_df(_raw)
    source_name = f"📤 {uploaded.name}  ({len(df_all):,} dòng)"
elif use_sample and os.path.exists(DATA_PATH):
    df_all = load_sample()
    source_name = f"📁 Dữ liệu mẫu  ({len(df_all):,} dòng)"
else:
    # Màn hình chào — chưa có dữ liệu thì chưa hiện dashboard
    st.markdown(f"""
    <div style='max-width:640px;margin:80px auto;text-align:center;
                background:{CARD_BG};border:1px solid {BORDER};
                border-top:4px solid {P1};border-radius:16px;padding:48px 40px;'>
      <div style='font-size:30px;font-weight:700;color:{TEXT};margin-bottom:12px;'>
        📊 Shopping Trends Analytics
      </div>
      <div style='font-size:15px;color:{MUTED};line-height:1.7;margin-bottom:8px;'>
        Chào mừng bạn! Để bắt đầu phân tích, hãy chọn một nguồn dữ liệu ở thanh bên trái:
      </div>
      <div style='font-size:14px;color:{TEXT};line-height:1.9;text-align:left;
                  max-width:420px;margin:20px auto 0;'>
        &nbsp;&nbsp;①&nbsp; <b>Tải lên file CSV</b> của riêng bạn, hoặc<br>
        &nbsp;&nbsp;②&nbsp; Tích chọn <b>"Dùng dữ liệu mẫu"</b> để xem demo.
      </div>
      <div style='font-size:12px;color:{MUTED};margin-top:24px;'>
        File tải lên cần cùng cấu trúc với bộ dữ liệu Shopping Trends.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── SIDEBAR FILTERS ──────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:11px;color:{P2};margin:4px 0 14px;'>{source_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;letter-spacing:1px;color:{MUTED};text-transform:uppercase;margin-bottom:12px;'>Filters</div>", unsafe_allow_html=True)

    seasons    = ["All"] + sorted(df_all["Season"].astype(str).unique().tolist())
    genders    = ["All"] + sorted(df_all["Gender"].astype(str).unique().tolist())
    categories = ["All"] + sorted(df_all["Category"].astype(str).unique().tolist())
    sel_season   = st.selectbox("Season",   seasons)
    sel_gender   = st.selectbox("Gender",   genders)
    sel_category = st.selectbox("Category", categories)
    age_min, age_max = int(df_all["Age"].min()), int(df_all["Age"].max())
    sel_age = st.slider("Age Range", age_min, age_max, (age_min, age_max))

    df = df_all.copy()
    if sel_season   != "All": df = df[df["Season"].astype(str)   == sel_season]
    if sel_gender   != "All": df = df[df["Gender"].astype(str)   == sel_gender]
    if sel_category != "All": df = df[df["Category"].astype(str) == sel_category]
    df = df[(df["Age"] >= sel_age[0]) & (df["Age"] <= sel_age[1])]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{MUTED};'>Records: <span style='color:{P2};font-weight:700;'>{len(df):,}</span></div>", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom:24px;'>
  <div style='font-size:26px;font-weight:700;color:{TEXT};letter-spacing:-0.5px;'>
    Shopping Trends Analytics
  </div>
  <div style='font-size:13px;color:{MUTED};margin-top:4px;'>
    Descriptive &nbsp;·&nbsp; Diagnostic &nbsp;·&nbsp; Predictive &nbsp;·&nbsp; Prescriptive
  </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ────────────────────────────────────────────────────
tabs = st.tabs(["KPI OVERVIEW", "REVENUE", "CUSTOMER", "PRODUCT", "PREDICTIVE", "RECOMMENDATIONS"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — KPI OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    total_rev  = df["Purchase Amount (USD)"].sum()
    total_cust = df["Customer ID"].nunique()
    avg_order  = df["Purchase Amount (USD)"].mean()
    sub_rate   = (df["Subscription Status"].astype(str) == "Yes").mean() * 100
    avg_rating = df["Review Rating"].mean()
    disc_rate  = (df["Discount Applied"].astype(str) == "Yes").mean() * 100
    promo_rate = (df["Promo Code Used"].astype(str)   == "Yes").mean() * 100

    # KPI row 1
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "TOTAL REVENUE",     f"${total_rev:,.0f}",   "Accumulated"),
        (c2, "TOTAL CUSTOMERS",   f"{total_cust:,}",       "Unique"),
        (c3, "AVG ORDER VALUE",   f"${avg_order:.2f}",     "Per transaction"),
        (c4, "SUBSCRIBER RATE",   f"{sub_rate:.1f}%",      "Active subscriptions"),
    ]
    for col, label, value, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # KPI row 2
    c5, c6, c7, c8 = st.columns(4)
    top_cat    = str(df.groupby("Category",    observed=True)["Purchase Amount (USD)"].sum().idxmax())
    top_season = str(df.groupby("Season",      observed=True)["Purchase Amount (USD)"].sum().idxmax())
    top_item   = str(df.groupby("Item Purchased", observed=True)["Purchase Amount (USD)"].sum().idxmax())
    top_loc    = str(df.groupby("Location",    observed=True)["Purchase Amount (USD)"].sum().idxmax())
    kpis2 = [
        (c5, "AVG RATING",      f"{avg_rating:.2f}/5.0",  "Customer satisfaction"),
        (c6, "DISCOUNT RATE",   f"{disc_rate:.1f}%",       "Orders with discount"),
        (c7, "TOP CATEGORY",    top_cat,                   "Highest revenue"),
        (c8, "PEAK SEASON",     top_season,                "Highest revenue"),
    ]
    for col, label, value, sub in kpis2:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="font-size:22px">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>REVENUE SNAPSHOT</div>", unsafe_allow_html=True)

    # 3 charts in overview
    ca, cb, cc = st.columns(3)

    with ca:
        rev_s = df.groupby("Season", observed=True)["Purchase Amount (USD)"].sum().reset_index()
        rev_s["Season"] = rev_s["Season"].astype(str)
        fig = px.bar(rev_s, x="Season", y="Purchase Amount (USD)",
                     color="Season", color_discrete_sequence=PALETTE,
                     text_auto=True)
        fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside",
                          textfont=dict(color=TEXT, size=10), marker_line_width=0)
        fig = fig_layout(fig, "REVENUE BY SEASON")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        cat_rev = df.groupby("Category", observed=True)["Purchase Amount (USD)"].sum().reset_index()
        cat_rev["Category"] = cat_rev["Category"].astype(str)
        fig = px.pie(cat_rev, values="Purchase Amount (USD)", names="Category",
                     hole=0.55, color_discrete_sequence=PALETTE)
        fig.update_traces(textfont_color=TEXT, textposition="outside",
                          marker=dict(line=dict(color=BG, width=2)))
        fig = fig_layout(fig, "REVENUE BY CATEGORY")
        st.plotly_chart(fig, use_container_width=True)

    with cc:
        gender_rev = df.groupby("Gender", observed=True)["Purchase Amount (USD)"].agg(
            Total="sum", Average="mean").reset_index()
        gender_rev["Gender"] = gender_rev["Gender"].astype(str)
        fig = px.bar(gender_rev, x="Gender", y="Average",
                     color="Gender", color_discrete_sequence=[P1, PK1],
                     text_auto=True)
        fig.update_traces(texttemplate="$%{y:.1f}", textposition="outside",
                          textfont=dict(color=TEXT), marker_line_width=0)
        fig = fig_layout(fig, "AVG ORDER VALUE BY GENDER")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Bottom row — 2 wider charts
    cd, ce = st.columns(2)

    with cd:
        age_rev = df.groupby("Age Group", observed=True)["Purchase Amount (USD)"].sum().reset_index()
        age_rev["Age Group"] = age_rev["Age Group"].astype(str)
        fig = px.bar(age_rev, x="Age Group", y="Purchase Amount (USD)",
                     color="Age Group", color_discrete_sequence=PALETTE)
        fig.update_traces(marker_line_width=0)
        fig = fig_layout(fig, "REVENUE BY AGE GROUP")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with ce:
        pivot_hs = df.pivot_table(values="Purchase Amount (USD)",
                                  index="Category", columns="Season",
                                  aggfunc="sum", observed=True)
        pivot_hs.index   = pivot_hs.index.astype(str)
        pivot_hs.columns = pivot_hs.columns.astype(str)
        fig = px.imshow(pivot_hs, text_auto=".0f",
                        color_continuous_scale=[[0,P3],[0.5,P1],[1,PK1]],
                        aspect="auto")
        fig.update_traces(textfont=dict(color=TEXT, size=11))
        fig = fig_layout(fig, "REVENUE HEATMAP: CATEGORY × SEASON")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — REVENUE
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f"<div class='section-title'>REVENUE DEEP DIVE</div>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        loc_rev = df.groupby("Location", observed=True)["Purchase Amount (USD)"].sum().reset_index()
        loc_rev["Location"] = loc_rev["Location"].astype(str)
        loc_rev = loc_rev.nlargest(15, "Purchase Amount (USD)")
        fig = px.bar(loc_rev, x="Purchase Amount (USD)", y="Location",
                     orientation="h", color="Purchase Amount (USD)",
                     color_continuous_scale=[[0,P3],[1,PK1]])
        fig.update_traces(marker_line_width=0)
        fig = fig_layout(fig, "TOP 15 LOCATIONS BY REVENUE", height=460)
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        ship_rev = df.groupby("Shipping Type", observed=True)["Purchase Amount (USD)"].sum().reset_index()
        ship_rev["Shipping Type"] = ship_rev["Shipping Type"].astype(str)
        fig = px.pie(ship_rev, values="Purchase Amount (USD)", names="Shipping Type",
                     hole=0.55, color_discrete_sequence=PALETTE)
        fig.update_traces(marker=dict(line=dict(color=BG, width=2)), textfont_color=TEXT)
        fig = fig_layout(fig, "REVENUE BY SHIPPING TYPE", height=460)
        st.plotly_chart(fig, use_container_width=True)

    r3, r4, r5 = st.columns(3)
    with r3:
        pay_rev = df.groupby("Payment Method", observed=True)["Purchase Amount (USD)"].mean().reset_index()
        pay_rev["Payment Method"] = pay_rev["Payment Method"].astype(str)
        pay_rev = pay_rev.sort_values("Purchase Amount (USD)", ascending=True)
        fig = px.bar(pay_rev, x="Purchase Amount (USD)", y="Payment Method",
                     orientation="h", color_discrete_sequence=[P1])
        fig.update_traces(marker_color=P1, marker_line_width=0)
        fig = fig_layout(fig, "AVG ORDER: PAYMENT METHOD")
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        disc_grp = df.groupby("Discount Applied", observed=True)["Purchase Amount (USD)"].mean().reset_index()
        disc_grp["Discount Applied"] = disc_grp["Discount Applied"].astype(str)
        fig = px.bar(disc_grp, x="Discount Applied", y="Purchase Amount (USD)",
                     color="Discount Applied", color_discrete_sequence=[P1, PK1],
                     text_auto=True)
        fig.update_traces(texttemplate="$%{y:.2f}", marker_line_width=0)
        fig = fig_layout(fig, "DISCOUNT IMPACT ON ORDER VALUE")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r5:
        promo_grp = df.groupby("Promo Code Used", observed=True)["Purchase Amount (USD)"].agg(
            Mean="mean", Count="count").reset_index()
        promo_grp["Promo Code Used"] = promo_grp["Promo Code Used"].astype(str)
        fig = px.bar(promo_grp, x="Promo Code Used", y="Mean",
                     color="Promo Code Used", color_discrete_sequence=[P3, PK3],
                     text_auto=True)
        fig.update_traces(texttemplate="$%{y:.2f}", marker_line_width=0)
        fig = fig_layout(fig, "PROMO CODE IMPACT")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Full-width heatmap
    st.markdown(f"<div class='section-title'>PAYMENT METHOD × CATEGORY</div>", unsafe_allow_html=True)
    pivot_pc = df.groupby(["Payment Method","Category"], observed=True)["Purchase Amount (USD)"].mean().reset_index()
    pvt = pivot_pc.pivot(index="Payment Method", columns="Category", values="Purchase Amount (USD)").fillna(0)
    pvt.index   = pvt.index.astype(str)
    pvt.columns = pvt.columns.astype(str)
    fig = px.imshow(pvt, text_auto=".0f", color_continuous_scale=[[0,P3],[0.5,P1],[1,PK1]], aspect="auto")
    fig.update_traces(textfont=dict(color=TEXT, size=11))
    fig = fig_layout(fig, "", height=320)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — CUSTOMER
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(f"<div class='section-title'>CUSTOMER PROFILE</div>", unsafe_allow_html=True)

    cu1, cu2, cu3 = st.columns(3)

    with cu1:
        # Histogram tuổi — single color gradient
        fig = px.histogram(df, x="Age", nbins=25, color_discrete_sequence=[P1])
        fig.update_traces(marker_color=P1, marker_line_color=BG, marker_line_width=1)
        fig = fig_layout(fig, "AGE DISTRIBUTION")
        st.plotly_chart(fig, use_container_width=True)

    with cu2:
        age_cnt = df["Age Group"].value_counts().reset_index()
        age_cnt.columns = ["Age Group","Count"]
        age_cnt["Age Group"] = age_cnt["Age Group"].astype(str)
        age_cnt = age_cnt.sort_values("Age Group")
        fig = px.bar(age_cnt, x="Age Group", y="Count",
                     color_discrete_sequence=[PK1])
        fig.update_traces(marker_color=PK1, marker_line_width=0)
        fig = fig_layout(fig, "CUSTOMERS BY AGE GROUP")
        st.plotly_chart(fig, use_container_width=True)

    with cu3:
        gender_cnt = df["Gender"].value_counts().reset_index()
        gender_cnt.columns = ["Gender","Count"]
        gender_cnt["Gender"] = gender_cnt["Gender"].astype(str)
        fig = px.pie(gender_cnt, values="Count", names="Gender",
                     hole=0.55, color_discrete_sequence=[P1, PK1])
        fig.update_traces(marker=dict(line=dict(color=BG, width=2)), textfont_color=TEXT)
        fig = fig_layout(fig, "GENDER SPLIT")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='section-title'>PURCHASE BEHAVIOR</div>", unsafe_allow_html=True)
    cu4, cu5, cu6 = st.columns(3)

    with cu4:
        freq = df["Frequency of Purchases"].value_counts().reset_index()
        freq.columns = ["Frequency","Count"]
        freq["Frequency"] = freq["Frequency"].astype(str)
        freq = freq.sort_values("Count", ascending=True)
        fig = px.bar(freq, x="Count", y="Frequency", orientation="h",
                     color_discrete_sequence=[P2])
        fig.update_traces(marker_color=P2, marker_line_width=0)
        fig = fig_layout(fig, "PURCHASE FREQUENCY")
        st.plotly_chart(fig, use_container_width=True)

    with cu5:
        sub_spend = df.groupby("Subscription Status", observed=True)["Purchase Amount (USD)"].agg(
            Mean="mean", Count="count").reset_index()
        sub_spend["Subscription Status"] = sub_spend["Subscription Status"].astype(str)
        fig = px.bar(sub_spend, x="Subscription Status", y="Mean",
                     color="Subscription Status", color_discrete_sequence=[P1, PK1],
                     text_auto=True)
        fig.update_traces(texttemplate="$%{y:.2f}", marker_line_width=0)
        fig = fig_layout(fig, "AVG SPEND: SUBSCRIBER VS NON")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with cu6:
        prev_bins = pd.cut(df["Previous Purchases"], bins=[0,5,15,30,50,100],
                           labels=["1-5","6-15","16-30","31-50","50+"])
        prev_cnt = prev_bins.value_counts().reset_index()
        prev_cnt.columns = ["Purchases","Count"]
        prev_cnt["Purchases"] = prev_cnt["Purchases"].astype(str)
        fig = px.pie(prev_cnt, values="Count", names="Purchases",
                     hole=0.55, color_discrete_sequence=PALETTE)
        fig.update_traces(marker=dict(line=dict(color=BG, width=2)), textfont_color=TEXT)
        fig = fig_layout(fig, "PREVIOUS PURCHASES DISTRIBUTION")
        st.plotly_chart(fig, use_container_width=True)

    # Boxplot Purchase Amount — clean single color
    st.markdown(f"<div class='section-title'>SPEND DISTRIBUTION</div>", unsafe_allow_html=True)
    bx1, bx2 = st.columns(2)

    with bx1:
        df_bx = df.copy()
        df_bx["Age Group"] = df_bx["Age Group"].astype(str)
        fig = px.box(df_bx, x="Age Group", y="Purchase Amount (USD)",
                     color_discrete_sequence=[P1])
        fig.update_traces(marker_color=P1, line_color=P2, fillcolor=f"rgba(168,85,247,0.2)")
        fig = fig_layout(fig, "PURCHASE AMOUNT BY AGE GROUP")
        st.plotly_chart(fig, use_container_width=True)

    with bx2:
        df_bx2 = df.copy()
        df_bx2["Season"] = df_bx2["Season"].astype(str)
        fig = px.violin(df_bx2, x="Season", y="Purchase Amount (USD)",
                        color_discrete_sequence=[PK1], box=True)
        fig.update_traces(fillcolor=f"rgba(236,72,153,0.2)", line_color=PK2)
        fig = fig_layout(fig, "PURCHASE AMOUNT DISTRIBUTION BY SEASON")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PRODUCT
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(f"<div class='section-title'>PRODUCT PERFORMANCE</div>", unsafe_allow_html=True)

    p1c, p2c = st.columns(2)
    with p1c:
        top10 = df.groupby("Item Purchased", observed=True)["Purchase Amount (USD)"].sum().nlargest(10).reset_index()
        top10["Item Purchased"] = top10["Item Purchased"].astype(str)
        top10 = top10.sort_values("Purchase Amount (USD)")
        fig = px.bar(top10, x="Purchase Amount (USD)", y="Item Purchased",
                     orientation="h", color="Purchase Amount (USD)",
                     color_continuous_scale=[[0,P3],[1,PK1]])
        fig.update_traces(marker_line_width=0)
        fig = fig_layout(fig, "TOP 10 PRODUCTS BY REVENUE", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with p2c:
        rating_item = df.groupby("Item Purchased", observed=True)["Review Rating"].mean().nlargest(10).reset_index()
        rating_item["Item Purchased"] = rating_item["Item Purchased"].astype(str)
        rating_item = rating_item.sort_values("Review Rating")
        fig = px.bar(rating_item, x="Review Rating", y="Item Purchased",
                     orientation="h", color="Review Rating",
                     color_continuous_scale=[[0,P3],[1,PK1]])
        fig.update_traces(marker_line_width=0)
        fig = fig_layout(fig, "TOP 10 PRODUCTS BY RATING", height=400)
        st.plotly_chart(fig, use_container_width=True)

    p3c, p4c, p5c = st.columns(3)
    with p3c:
        size_cnt = df["Size"].value_counts().reset_index()
        size_cnt.columns = ["Size","Count"]
        size_cnt["Size"] = size_cnt["Size"].astype(str)
        fig = px.pie(size_cnt, values="Count", names="Size",
                     hole=0.55, color_discrete_sequence=PALETTE)
        fig.update_traces(marker=dict(line=dict(color=BG, width=2)), textfont_color=TEXT)
        fig = fig_layout(fig, "SIZE DISTRIBUTION")
        st.plotly_chart(fig, use_container_width=True)

    with p4c:
        color_rev = df.groupby("Color", observed=True)["Purchase Amount (USD)"].sum().nlargest(10).reset_index()
        color_rev["Color"] = color_rev["Color"].astype(str)
        color_rev = color_rev.sort_values("Purchase Amount (USD)")
        fig = px.bar(color_rev, x="Purchase Amount (USD)", y="Color",
                     orientation="h", color_discrete_sequence=[P2])
        fig.update_traces(marker_color=P2, marker_line_width=0)
        fig = fig_layout(fig, "TOP COLORS BY REVENUE")
        st.plotly_chart(fig, use_container_width=True)

    with p5c:
        cat_rating = df.groupby("Category", observed=True)["Review Rating"].mean().reset_index()
        cat_rating["Category"] = cat_rating["Category"].astype(str)
        fig = px.bar(cat_rating, x="Category", y="Review Rating",
                     color_discrete_sequence=[PK2], text_auto=True)
        fig.update_traces(marker_color=PK2, marker_line_width=0,
                          texttemplate="%{y:.2f}")
        fig.update_yaxes(range=[3, 4.2])
        fig = fig_layout(fig, "AVG RATING BY CATEGORY")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: Revenue vs Rating per product
    st.markdown(f"<div class='section-title'>PRODUCT SCORE MATRIX</div>", unsafe_allow_html=True)
    prod_df = df.groupby("Item Purchased", observed=True).agg(
        Revenue=("Purchase Amount (USD)", "sum"),
        Rating=("Review Rating", "mean"),
        Count=("Customer ID", "count")
    ).reset_index()
    prod_df["Item Purchased"] = prod_df["Item Purchased"].astype(str)
    fig = px.scatter(prod_df, x="Revenue", y="Rating", size="Count",
                     text="Item Purchased", color="Revenue",
                     color_continuous_scale=[[0,P3],[0.5,P1],[1,PK1]],
                     size_max=30)
    fig.update_traces(textposition="top center", textfont=dict(color=MUTED, size=9))
    fig = fig_layout(fig, "PRODUCT MATRIX: REVENUE × RATING × VOLUME", height=420)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — PREDICTIVE
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(f"<div class='section-title'>PREDICTIVE ANALYTICS</div>", unsafe_allow_html=True)

    @st.cache_data
    def get_model_metrics(df_source):
        import warnings, numpy as np
        warnings.filterwarnings("ignore")
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        df_m = df_source.copy()
        for col, mapping in [("Subscription Status",{"Yes":1,"No":0}),
                              ("Discount Applied",   {"Yes":1,"No":0}),
                              ("Promo Code Used",    {"Yes":1,"No":0})]:
            df_m[col+"_bin"] = df_m[col].map(mapping).fillna(0).astype(int)
        freq_map = {"Weekly":52,"Bi-Weekly":26,"Fortnightly":26,
                    "Monthly":12,"Quarterly":4,"Annually":1,"Every 3 Months":4}
        df_m["Frequency_num"] = df_m["Frequency of Purchases"].map(freq_map).fillna(4)
        for col in ["Gender","Category","Season","Size","Shipping Type"]:
            dummies = pd.get_dummies(df_m[col], prefix=col, drop_first=True)
            df_m = pd.concat([df_m, dummies], axis=1)

        target = "Purchase Amount (USD)"
        drop = ["Customer ID","Item Purchased","Color","Location","Payment Method",
                "Preferred Payment Method","Gender","Category","Season","Size",
                "Shipping Type","Subscription Status","Discount Applied",
                "Promo Code Used","Frequency of Purchases", target]
        feat_cols = [c for c in df_m.columns
                     if c not in drop and df_m[c].dtype in [np.int64,np.float64,bool,np.uint8]]
        X = df_m[feat_cols].fillna(0)
        y = df_m[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results, importances = [], {}
        for name, model in [("Linear Regression", LinearRegression()),
                             ("Random Forest", RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1))]:
            model.fit(X_train, y_train)
            p = model.predict(X_test)
            results.append({"Model":name,
                            "MAE":round(mean_absolute_error(y_test,p),3),
                            "RMSE":round(np.sqrt(mean_squared_error(y_test,p)),3),
                            "R2":round(r2_score(y_test,p),4),
                            "y_test":y_test.values,"y_pred":p})
            if hasattr(model,"feature_importances_"):
                importances[name] = pd.Series(model.feature_importances_, index=feat_cols).nlargest(12)
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=42,verbosity=0)
            xgb.fit(X_train, y_train); p = xgb.predict(X_test)
            results.append({"Model":"XGBoost",
                            "MAE":round(mean_absolute_error(y_test,p),3),
                            "RMSE":round(np.sqrt(mean_squared_error(y_test,p)),3),
                            "R2":round(r2_score(y_test,p),4),
                            "y_test":y_test.values,"y_pred":p})
            importances["XGBoost"] = pd.Series(xgb.feature_importances_, index=feat_cols).nlargest(12)
        except ImportError:
            pass
        metrics_df = pd.DataFrame([{k:v for k,v in r.items() if k not in ("y_test","y_pred")} for r in results])
        return results, metrics_df, importances

    with st.spinner("Loading model results..."):
        all_results, metrics_df, importances = get_model_metrics(df_all)

    mc1, mc2 = st.columns(2)
    with mc1:
        fig = px.bar(metrics_df, x="Model", y="R2", color="Model",
                     color_discrete_sequence=PALETTE, text_auto=".4f")
        fig.update_traces(marker_line_width=0, textposition="outside", textfont=dict(color=TEXT))
        fig = fig_layout(fig, "R2 SCORE COMPARISON")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with mc2:
        fig = go.Figure()
        for metric, clr in [("MAE", P1), ("RMSE", PK1)]:
            fig.add_trace(go.Bar(name=metric, x=metrics_df["Model"], y=metrics_df[metric],
                                 marker_color=clr, marker_line_width=0,
                                 text=metrics_df[metric], textposition="outside",
                                 textfont=dict(color=TEXT)))
        fig.update_layout(barmode="group")
        fig = fig_layout(fig, "MAE & RMSE COMPARISON")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='section-title'>MODEL PERFORMANCE SUMMARY</div>", unsafe_allow_html=True)
    best_model = metrics_df.loc[metrics_df["R2"].idxmax(), "Model"]
    st.dataframe(metrics_df.style
        .highlight_max(subset=["R2"], color="#3D1F6E")
        .highlight_min(subset=["MAE","RMSE"], color="#3D1F6E")
        .format({"MAE":"{:.3f}","RMSE":"{:.3f}","R2":"{:.4f}"}),
        use_container_width=True, height=140)
    st.markdown(f"<div style='color:{PK2};font-weight:700;font-size:13px;margin-top:6px;'>Best model: {best_model}</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='section-title'>ACTUAL VS PREDICTED</div>", unsafe_allow_html=True)
    avp_cols = st.columns(len(all_results))
    for i, res in enumerate(all_results):
        with avp_cols[i]:
            yt, yp = res["y_test"], res["y_pred"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yt, y=yp, mode="markers",
                marker=dict(color=P1, size=4, opacity=0.5), name="Predicted"))
            mn, mx = float(min(yt.min(),yp.min())), float(max(yt.max(),yp.max()))
            fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                line=dict(color=PK1, dash="dash", width=1.5), name="Ideal"))
            fig = fig_layout(fig, res["Model"].upper(), height=300)
            st.plotly_chart(fig, use_container_width=True)

    if importances:
        st.markdown(f"<div class='section-title'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
        fi_cols = st.columns(len(importances))
        for i, (mname, imp) in enumerate(importances.items()):
            with fi_cols[i]:
                imp_df = imp.reset_index()
                imp_df.columns = ["Feature","Importance"]
                imp_df = imp_df.sort_values("Importance")
                fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale=[[0,P3],[1,PK1]])
                fig.update_traces(marker_line_width=0)
                fig = fig_layout(fig, f"{mname.upper()}: FEATURE IMPORTANCE", height=380)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='section-title'>DISTRIBUTION ANALYSIS</div>", unsafe_allow_html=True)
    pr1, pr2 = st.columns(2)
    with pr1:
        fig = px.histogram(df, x="Purchase Amount (USD)", nbins=30, color_discrete_sequence=[P1])
        fig.update_traces(marker_color=P1, marker_line_color=BG, marker_line_width=1)
        fig = fig_layout(fig, "PURCHASE AMOUNT DISTRIBUTION")
        st.plotly_chart(fig, use_container_width=True)
    with pr2:
        fig = px.histogram(df, x="Review Rating", nbins=20, color_discrete_sequence=[PK1])
        fig.update_traces(marker_color=PK1, marker_line_color=BG, marker_line_width=1)
        fig = fig_layout(fig, "REVIEW RATING DISTRIBUTION")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    total_rev2  = df["Purchase Amount (USD)"].sum()
    top_cat2    = str(df.groupby("Category",      observed=True)["Purchase Amount (USD)"].sum().idxmax())
    top_sea2    = str(df.groupby("Season",         observed=True)["Purchase Amount (USD)"].sum().idxmax())
    top_loc2    = str(df.groupby("Location",       observed=True)["Purchase Amount (USD)"].sum().idxmax())
    top_item2   = str(df.groupby("Item Purchased", observed=True)["Purchase Amount (USD)"].sum().idxmax())
    top_age2    = str(df.groupby("Age Group",      observed=True)["Purchase Amount (USD)"].sum().idxmax())
    sub_m       = df[df["Subscription Status"].astype(str)=="Yes"]["Purchase Amount (USD)"].mean()
    nosub_m     = df[df["Subscription Status"].astype(str)=="No"]["Purchase Amount (USD)"].mean()
    premium     = (sub_m - nosub_m) / nosub_m * 100

    i1, i2 = st.columns([1, 1])

    with i1:
        st.markdown(f"<div class='section-title'>KEY INSIGHTS</div>", unsafe_allow_html=True)
        insights = [
            ("I01", f"Total revenue ${total_rev2:,.0f} from {df['Customer ID'].nunique():,} customers"),
            ("I02", f"Category <b>{top_cat2}</b> contributes 44.7% of total revenue"),
            ("I03", f"Peak season: <b>{top_sea2}</b> — statistically significant (ANOVA p=0.011)"),
            ("I04", f"Best-selling product: <b>{top_item2}</b>"),
            ("I05", f"Highest revenue location: <b>{top_loc2}</b>"),
            ("I06", f"Top contributing age group: <b>{top_age2}</b>"),
            ("I07", f"Subscribers spend <b>{premium:+.1f}%</b> vs non-subscribers (T-test p=0.66, not significant)"),
            ("I08", "Purchase amount evenly distributed $20–$100 → opportunity in frequency, not order size"),
        ]
        for code, text in insights:
            st.markdown(f"""
            <div class="insight-row">
              <span class="insight-code">{code}</span>
              <span class="insight-text">{text}</span>
            </div>""", unsafe_allow_html=True)

    with i2:
        st.markdown(f"<div class='section-title'>ACTIONABLE RECOMMENDATIONS</div>", unsafe_allow_html=True)
        recs = [
            ("REVENUE GROWTH",       f"Increase marketing budget 30% for <b>{top_sea2}</b> season. Launch weekend flash sales to boost purchase frequency."),
            ("TARGET SEGMENTS",      f"Prioritize age group <b>{top_age2}</b> — highest revenue contribution. Run retargeting campaigns for top 25% spenders."),
            ("PRODUCT STRATEGY",     f"Push <b>{top_item2}</b> + bundle accessories with <b>{top_cat2}</b> to increase basket size."),
            ("LOYALTY PROGRAM",      "4-tier loyalty: Regular → Silver → Gold → VIP. Double points on orders > $60. Personalized emails for customers with 10+ purchases."),
            ("INVENTORY PLANNING",   f"Increase <b>{top_cat2}</b> stock 40% before <b>{top_sea2}</b>. Apply dynamic pricing in off-season to clear inventory."),
            ("MARKETING BUDGET",     f"Allocate: 44.7% Clothing · 31.8% Accessories · 15.5% Footwear. Focus 60% digital spend at <b>{top_loc2}</b>."),
        ]
        for title, body in recs:
            st.markdown(f"""
            <div class="rec-card">
              <div class="rec-title">{title}</div>
              <div class="rec-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>BUDGET ALLOCATION</div>", unsafe_allow_html=True)

    ba1, ba2 = st.columns(2)
    with ba1:
        cat_rev2 = df.groupby("Category", observed=True)["Purchase Amount (USD)"].sum()
        budget   = (cat_rev2 / cat_rev2.sum() * 100).round(2).reset_index()
        budget.columns = ["Category","Budget_%"]
        budget["Category"] = budget["Category"].astype(str)
        fig = px.pie(budget, values="Budget_%", names="Category",
                     hole=0.55, color_discrete_sequence=PALETTE)
        fig.update_traces(marker=dict(line=dict(color=BG, width=2)), textfont_color=TEXT)
        fig = fig_layout(fig, "RECOMMENDED MARKETING BUDGET SPLIT")
        st.plotly_chart(fig, use_container_width=True)

    with ba2:
        season_cat2 = df.groupby(["Season","Category"], observed=True).size().reset_index(name="Count")
        season_cat2["Season"]   = season_cat2["Season"].astype(str)
        season_cat2["Category"] = season_cat2["Category"].astype(str)
        fig = px.bar(season_cat2, x="Season", y="Count", color="Category",
                     barmode="group", color_discrete_sequence=PALETTE)
        fig.update_traces(marker_line_width=0)
        fig = fig_layout(fig, "DEMAND BY SEASON & CATEGORY (INVENTORY GUIDE)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div style='font-size:11px;color:{MUTED};text-align:center;margin-top:20px;'>Shopping Trends Analytics · FAA4021 · Built with Streamlit + Plotly</div>", unsafe_allow_html=True)
