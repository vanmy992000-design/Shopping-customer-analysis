"""
descriptive_analysis.py
========================
Module phân tích mô tả (Descriptive Analytics) cho dữ liệu shopping_trends.

Bao gồm:
- Thống kê cơ bản (tổng KH, tổng doanh thu, ...)
- Doanh thu theo giới tính, nhóm tuổi, danh mục, mùa, địa điểm
- Top 10 sản phẩm bán chạy & địa điểm doanh thu cao nhất
- Tần suất mua hàng theo Subscription Status
- Trực quan hóa: Bar Chart, Pie Chart, Histogram, Boxplot, Heatmap
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Không dùng GUI backend (dùng cho lưu file)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np

# ============================================================
# PATHS & CẤU HÌNH
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Style biểu đồ
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = "Set2"
FIG_DPI = 150


def _save(fig, filename: str):
    """Lưu biểu đồ vào thư mục output/charts."""
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] Lưu: {filename}")
    return path


# ============================================================
# THỐNG KÊ TỔNG HỢP
# ============================================================
def basic_stats(df: pd.DataFrame) -> dict:
    """Tính các chỉ số KPI cơ bản."""
    stats = {
        "total_customers"   : df["Customer ID"].nunique(),
        "total_revenue"     : df["Purchase Amount (USD)"].sum(),
        "avg_purchase"      : df["Purchase Amount (USD)"].mean().round(2),
        "median_purchase"   : df["Purchase Amount (USD)"].median(),
        "total_categories"  : df["Category"].nunique(),
        "total_locations"   : df["Location"].nunique(),
        "subscriber_rate"   : (df["Subscription Status"] == "Yes").mean() * 100,
    }
    print("\n[INFO] KPI Tổng hợp:")
    for k, v in stats.items():
        print(f"  {k:25s}: {v:,.2f}" if isinstance(v, float) else f"  {k:25s}: {v:,}")
    return stats


# ============================================================
# DOANH THU THEO GIỚI TÍNH
# ============================================================
def revenue_by_gender(df: pd.DataFrame):
    """Tính và vẽ biểu đồ doanh thu theo giới tính."""
    gender_rev = df.groupby("Gender", observed=True)["Purchase Amount (USD)"].agg(
        Total="sum", Average="mean", Count="count"
    ).reset_index().round(2)
    print("\n[INFO] Doanh thu theo Giới tính:")
    print(gender_rev.to_string(index=False))

    # Pie Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pie(
        gender_rev["Total"], labels=gender_rev["Gender"],
        autopct="%1.1f%%", startangle=90,
        colors=sns.color_palette(PALETTE, len(gender_rev))
    )
    axes[0].set_title("Tổng Doanh Thu theo Giới Tính")

    # Bar Chart
    bars = axes[1].bar(
        gender_rev["Gender"], gender_rev["Average"],
        color=sns.color_palette(PALETTE, len(gender_rev))
    )
    axes[1].set_title("Doanh Thu Trung Bình theo Giới Tính")
    axes[1].set_ylabel("USD")
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"${bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=10
        )
    fig.suptitle("Phân Tích Doanh Thu theo Giới Tính", fontsize=14, fontweight="bold")
    _save(fig, "01_revenue_by_gender.png")
    return gender_rev


# ============================================================
# DOANH THU THEO NHÓM TUỔI
# ============================================================
def revenue_by_age_group(df: pd.DataFrame):
    """Tính và vẽ biểu đồ doanh thu theo nhóm tuổi."""
    age_rev = df.groupby("Age Group", observed=True)["Purchase Amount (USD)"].agg(
        Total="sum", Average="mean", Count="count"
    ).reset_index().round(2)
    print("\n[INFO] Doanh thu theo Nhóm Tuổi:")
    print(age_rev.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette(PALETTE, len(age_rev))
    bars = ax.bar(age_rev["Age Group"].astype(str), age_rev["Total"], color=colors)
    ax.set_title("Tổng Doanh Thu theo Nhóm Tuổi", fontsize=13, fontweight="bold")
    ax.set_xlabel("Nhóm Tuổi")
    ax.set_ylabel("Tổng Doanh Thu (USD)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"${bar.get_height():,.0f}",
            ha="center", va="bottom", fontsize=9
        )
    _save(fig, "02_revenue_by_age_group.png")
    return age_rev


# ============================================================
# DOANH THU THEO DANH MỤC SẢN PHẨM
# ============================================================
def revenue_by_category(df: pd.DataFrame):
    """Doanh thu theo Category."""
    cat_rev = df.groupby("Category", observed=True)["Purchase Amount (USD)"].agg(
        Total="sum", Average="mean", Count="count"
    ).reset_index().sort_values("Total", ascending=False).round(2)
    print("\n[INFO] Doanh thu theo Danh mục:")
    print(cat_rev.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = sns.color_palette(PALETTE, len(cat_rev))

    # Bar chart
    axes[0].barh(cat_rev["Category"].astype(str), cat_rev["Total"], color=colors)
    axes[0].set_title("Tổng Doanh Thu theo Danh Mục")
    axes[0].set_xlabel("USD")
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Pie chart
    axes[1].pie(
        cat_rev["Total"], labels=cat_rev["Category"].astype(str),
        autopct="%1.1f%%", startangle=140, colors=colors
    )
    axes[1].set_title("Tỷ Trọng Doanh Thu theo Danh Mục")
    fig.suptitle("Phân Tích Doanh Thu theo Danh Mục Sản Phẩm", fontsize=13, fontweight="bold")
    _save(fig, "03_revenue_by_category.png")
    return cat_rev


# ============================================================
# DOANH THU THEO MÙA
# ============================================================
def revenue_by_season(df: pd.DataFrame):
    """Doanh thu theo mùa (Season)."""
    season_rev = df.groupby("Season", observed=True)["Purchase Amount (USD)"].agg(
        Total="sum", Average="mean", Count="count"
    ).reset_index().sort_values("Total", ascending=False).round(2)
    print("\n[INFO] Doanh thu theo Mùa:")
    print(season_rev.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette(PALETTE, len(season_rev))
    bars = ax.bar(season_rev["Season"].astype(str), season_rev["Total"], color=colors)
    ax.set_title("Tổng Doanh Thu theo Mùa", fontsize=13, fontweight="bold")
    ax.set_ylabel("Tổng Doanh Thu (USD)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"${bar.get_height():,.0f}",
            ha="center", va="bottom", fontsize=10
        )
    _save(fig, "04_revenue_by_season.png")
    return season_rev


# ============================================================
# TOP 10 SẢN PHẨM BÁN CHẠY
# ============================================================
def top10_items(df: pd.DataFrame):
    """Top 10 sản phẩm bán chạy nhất theo doanh thu."""
    top10 = (
        df.groupby("Item Purchased", observed=True)["Purchase Amount (USD)"]
        .agg(Total="sum", Count="count")
        .reset_index()
        .sort_values("Total", ascending=False)
        .head(10)
    )
    print("\n[INFO] Top 10 sản phẩm bán chạy:")
    print(top10.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = sns.color_palette("Spectral", 10)
    bars = ax.barh(top10["Item Purchased"].astype(str)[::-1], top10["Total"][::-1], color=colors)
    ax.set_title("Top 10 Sản Phẩm Bán Chạy (theo Doanh Thu)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tổng Doanh Thu (USD)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _save(fig, "05_top10_items.png")
    return top10


# ============================================================
# TOP 10 ĐỊA ĐIỂM DOANH THU CAO NHẤT
# ============================================================
def top10_locations(df: pd.DataFrame):
    """Top 10 địa điểm có doanh thu cao nhất."""
    top_loc = (
        df.groupby("Location", observed=True)["Purchase Amount (USD)"]
        .sum()
        .reset_index()
        .sort_values("Purchase Amount (USD)", ascending=False)
        .head(10)
    )
    print("\n[INFO] Top 10 địa điểm doanh thu cao nhất:")
    print(top_loc.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = sns.color_palette("Blues_d", 10)
    ax.barh(
        top_loc["Location"].astype(str)[::-1],
        top_loc["Purchase Amount (USD)"][::-1],
        color=colors
    )
    ax.set_title("Top 10 Địa Điểm Doanh Thu Cao Nhất", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tổng Doanh Thu (USD)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _save(fig, "06_top10_locations.png")
    return top_loc


# ============================================================
# TẦN SUẤT MUA HÀNG THEO SUBSCRIPTION STATUS
# ============================================================
def purchase_frequency_by_subscription(df: pd.DataFrame):
    """Tần suất mua hàng theo Subscription Status."""
    freq_sub = df.groupby(
        ["Subscription Status", "Frequency of Purchases"], observed=True
    ).size().reset_index(name="Count")
    print("\n[INFO] Tần suất mua hàng theo Subscription Status:")
    print(freq_sub.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 5))
    freq_pivot = freq_sub.pivot(
        index="Frequency of Purchases", columns="Subscription Status", values="Count"
    ).fillna(0)
    freq_pivot.plot(kind="bar", ax=ax, color=sns.color_palette(PALETTE, 2))
    ax.set_title("Tần Suất Mua Hàng theo Subscription Status", fontsize=13, fontweight="bold")
    ax.set_ylabel("Số Khách Hàng")
    ax.set_xlabel("Tần Suất Mua Hàng")
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Subscription")
    _save(fig, "07_frequency_by_subscription.png")
    return freq_sub


# ============================================================
# HISTOGRAM - PHÂN PHỐI PURCHASE AMOUNT
# ============================================================
def histogram_purchase_amount(df: pd.DataFrame):
    """Vẽ histogram phân phối giá trị mua hàng."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram tổng
    axes[0].hist(
        df["Purchase Amount (USD)"], bins=30,
        color=sns.color_palette(PALETTE)[0], edgecolor="white"
    )
    axes[0].set_title("Phân Phối Purchase Amount (Tổng)")
    axes[0].set_xlabel("USD")
    axes[0].set_ylabel("Số lượng")

    # Histogram theo giới tính
    for gender, grp in df.groupby("Gender", observed=True):
        axes[1].hist(
            grp["Purchase Amount (USD)"], bins=25,
            alpha=0.6, label=str(gender), edgecolor="white"
        )
    axes[1].set_title("Phân Phối Purchase Amount theo Giới Tính")
    axes[1].set_xlabel("USD")
    axes[1].legend()

    fig.suptitle("Histogram - Phân Phối Giá Trị Mua Hàng", fontsize=13, fontweight="bold")
    _save(fig, "08_histogram_purchase_amount.png")


# ============================================================
# BOXPLOT - PURCHASE AMOUNT THEO CATEGORY
# ============================================================
def boxplot_by_category(df: pd.DataFrame):
    """Boxplot giá trị mua hàng theo danh mục sản phẩm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Theo Category
    df_box = df.copy()
    df_box["Category"] = df_box["Category"].astype(str)
    sns.boxplot(
        data=df_box, x="Category", y="Purchase Amount (USD)",
        palette=PALETTE, ax=axes[0]
    )
    axes[0].set_title("Purchase Amount theo Danh Mục")
    axes[0].set_xlabel("Danh Mục")

    # Theo Season
    df_box["Season"] = df_box["Season"].astype(str)
    sns.boxplot(
        data=df_box, x="Season", y="Purchase Amount (USD)",
        palette="pastel", ax=axes[1]
    )
    axes[1].set_title("Purchase Amount theo Mùa")
    axes[1].set_xlabel("Mùa")

    fig.suptitle("Boxplot - Purchase Amount theo Danh Mục & Mùa", fontsize=13, fontweight="bold")
    _save(fig, "09_boxplot_category_season.png")


# ============================================================
# HEATMAP - DOANH THU THEO CATEGORY × SEASON
# ============================================================
def heatmap_category_season(df: pd.DataFrame):
    """Heatmap doanh thu theo Danh mục × Mùa."""
    pivot = df.pivot_table(
        values="Purchase Amount (USD)",
        index="Category",
        columns="Season",
        aggfunc="sum",
        observed=True
    )
    pivot.index = pivot.index.astype(str)
    pivot.columns = pivot.columns.astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "USD"}
    )
    ax.set_title("Heatmap: Doanh Thu theo Danh Mục × Mùa", fontsize=13, fontweight="bold")
    _save(fig, "10_heatmap_category_season.png")
    return pivot


# ============================================================
# HÀM TỔNG HỢP
# ============================================================
def run_descriptive(df: pd.DataFrame) -> dict:
    """Chạy toàn bộ phân tích mô tả và trả về dict kết quả."""
    print("\n" + "=" * 50)
    print("  BƯỚC 2: PHÂN TÍCH MÔ TẢ (DESCRIPTIVE)")
    print("=" * 50)

    results = {}
    results["stats"]        = basic_stats(df)
    results["gender"]       = revenue_by_gender(df)
    results["age_group"]    = revenue_by_age_group(df)
    results["category"]     = revenue_by_category(df)
    results["season"]       = revenue_by_season(df)
    results["top10_items"]  = top10_items(df)
    results["top10_locs"]   = top10_locations(df)
    results["freq_sub"]     = purchase_frequency_by_subscription(df)

    # Biểu đồ bổ sung
    histogram_purchase_amount(df)
    boxplot_by_category(df)
    results["heatmap"]      = heatmap_category_season(df)

    print("\n[INFO] Phân tích mô tả hoàn tất!\n")
    return results


if __name__ == "__main__":
    from data_cleaning import run_cleaning
    df = run_cleaning()
    run_descriptive(df)
