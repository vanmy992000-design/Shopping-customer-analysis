"""
diagnostic_analysis.py
=======================
Module phân tích chẩn đoán (Diagnostic Analytics).

Trả lời các câu hỏi:
- Vì sao doanh thu tăng/giảm?
- Nhóm khách hàng đóng góp doanh thu nhiều nhất?
- Giới tính nào chi tiêu nhiều hơn?
- Danh mục sản phẩm nào hoạt động tốt nhất?
- Mùa nào tạo doanh thu cao nhất?
- Subscriber có chi tiêu nhiều hơn không?
- Có sự khác biệt doanh thu giữa các khu vực không?

Kỹ thuật:
- Correlation Analysis
- Pivot Table Analysis
- Group By Analysis
- T-test (Subscription)
- ANOVA (Season, Category)
- Heatmap Correlation, Clustered Bar Chart, Comparative Boxplot
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "output", "reports")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
PALETTE = "Set2"
FIG_DPI = 150


def _save(fig, filename: str):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] Lưu: {filename}")
    return path


# ============================================================
# CORRELATION ANALYSIS
# ============================================================
def correlation_analysis(df: pd.DataFrame):
    """
    Phân tích tương quan giữa các biến số.
    Tạo heatmap correlation.
    """
    # Encode biến phân loại để tính correlation
    df_enc = df.copy()
    for col in ["Gender", "Season", "Category", "Subscription Status",
                "Discount Applied", "Promo Code Used", "Frequency of Purchases"]:
        if col in df_enc.columns:
            df_enc[col + "_enc"] = df_enc[col].astype(str).astype("category").cat.codes

    num_cols = [
        "Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases",
        "Gender_enc", "Season_enc", "Category_enc",
        "Subscription Status_enc", "Discount Applied_enc",
        "Promo Code Used_enc", "Frequency of Purchases_enc"
    ]
    num_cols = [c for c in num_cols if c in df_enc.columns]
    corr_matrix = df_enc[num_cols].corr().round(3)

    # Đổi tên cột ngắn hơn
    rename_map = {
        "Purchase Amount (USD)": "Purchase Amt",
        "Review Rating": "Review",
        "Previous Purchases": "Prev Purchases",
        "Gender_enc": "Gender",
        "Season_enc": "Season",
        "Category_enc": "Category",
        "Subscription Status_enc": "Subscribed",
        "Discount Applied_enc": "Discount",
        "Promo Code Used_enc": "Promo",
        "Frequency of Purchases_enc": "Frequency"
    }
    corr_matrix.rename(index=rename_map, columns=rename_map, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        ax=ax, vmin=-1, vmax=1, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Heatmap Tương Quan giữa các Biến", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    _save(fig, "11_correlation_heatmap.png")

    print("\n[INFO] Top tương quan với Purchase Amount:")
    target_corr = corr_matrix["Purchase Amt"].drop("Purchase Amt").sort_values(key=abs, ascending=False)
    print(target_corr.to_string())
    return corr_matrix


# ============================================================
# PIVOT TABLE ANALYSIS
# ============================================================
def pivot_analysis(df: pd.DataFrame):
    """
    Phân tích Pivot Table:
    - Doanh thu trung bình theo Gender × Season
    - Doanh thu trung bình theo Age Group × Category
    """
    # Pivot 1: Gender × Season
    pivot_gender_season = df.pivot_table(
        values="Purchase Amount (USD)",
        index="Gender",
        columns="Season",
        aggfunc="mean",
        observed=True
    ).round(2)
    pivot_gender_season.index = pivot_gender_season.index.astype(str)
    pivot_gender_season.columns = pivot_gender_season.columns.astype(str)
    print("\n[INFO] Pivot: Doanh thu TB theo Gender × Season:")
    print(pivot_gender_season.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.heatmap(
        pivot_gender_season, annot=True, fmt=".1f",
        cmap="Blues", ax=axes[0], linewidths=0.5
    )
    axes[0].set_title("Doanh Thu TB: Gender × Season")

    # Pivot 2: Age Group × Category
    pivot_age_cat = df.pivot_table(
        values="Purchase Amount (USD)",
        index="Age Group",
        columns="Category",
        aggfunc="mean",
        observed=True
    ).round(2)
    pivot_age_cat.index = pivot_age_cat.index.astype(str)
    pivot_age_cat.columns = pivot_age_cat.columns.astype(str)

    sns.heatmap(
        pivot_age_cat, annot=True, fmt=".1f",
        cmap="Oranges", ax=axes[1], linewidths=0.5
    )
    axes[1].set_title("Doanh Thu TB: Age Group × Category")
    fig.suptitle("Pivot Table Analysis", fontsize=13, fontweight="bold")
    _save(fig, "12_pivot_table_analysis.png")

    return pivot_gender_season, pivot_age_cat


# ============================================================
# CLUSTERED BAR CHART: DOANH THU THEO CATEGORY × SUBSCRIPTION
# ============================================================
def clustered_bar_subscription(df: pd.DataFrame):
    """Clustered Bar Chart: Doanh thu theo Category × Subscription Status."""
    grp = df.groupby(
        ["Category", "Subscription Status"], observed=True
    )["Purchase Amount (USD)"].mean().reset_index()
    grp["Category"] = grp["Category"].astype(str)
    grp["Subscription Status"] = grp["Subscription Status"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot_sub = grp.pivot(index="Category", columns="Subscription Status", values="Purchase Amount (USD)")
    pivot_sub.plot(kind="bar", ax=ax, color=sns.color_palette(PALETTE, 2), edgecolor="white")
    ax.set_title("Doanh Thu TB theo Danh Mục & Subscription Status",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Doanh Thu Trung Bình (USD)")
    ax.set_xlabel("Danh Mục")
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="Subscription")
    _save(fig, "13_clustered_bar_subscription.png")
    return grp


# ============================================================
# COMPARATIVE BOXPLOT: SUBSCRIPTION VS NON-SUBSCRIPTION
# ============================================================
def comparative_boxplot_subscription(df: pd.DataFrame):
    """Comparative Boxplot: Subscriber vs Non-subscriber."""
    df_plot = df.copy()
    df_plot["Subscription Status"] = df_plot["Subscription Status"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot theo Subscription
    sns.boxplot(
        data=df_plot, x="Subscription Status",
        y="Purchase Amount (USD)", palette=PALETTE, ax=axes[0]
    )
    axes[0].set_title("Purchase Amount: Subscriber vs Non-Subscriber")
    axes[0].set_ylabel("USD")

    # Boxplot theo Season × Subscription
    df_plot["Season"] = df_plot["Season"].astype(str)
    sns.boxplot(
        data=df_plot, x="Season", y="Purchase Amount (USD)",
        hue="Subscription Status", palette=PALETTE, ax=axes[1]
    )
    axes[1].set_title("Purchase Amount: Season × Subscription Status")
    axes[1].set_ylabel("USD")
    axes[1].legend(title="Subscription")

    fig.suptitle("Comparative Boxplot - Subscription Analysis", fontsize=13, fontweight="bold")
    _save(fig, "14_comparative_boxplot_subscription.png")


# ============================================================
# T-TEST: SUBSCRIBER VS NON-SUBSCRIBER
# ============================================================
def ttest_subscription(df: pd.DataFrame) -> dict:
    """T-test kiểm định sự khác biệt doanh thu giữa subscriber và non-subscriber."""
    sub_yes = df[df["Subscription Status"].astype(str) == "Yes"]["Purchase Amount (USD)"].dropna()
    sub_no  = df[df["Subscription Status"].astype(str) == "No"]["Purchase Amount (USD)"].dropna()

    t_stat, p_value = stats.ttest_ind(sub_yes, sub_no, equal_var=False)  # Welch's t-test
    result = {
        "mean_subscriber"    : sub_yes.mean().round(2),
        "mean_non_subscriber": sub_no.mean().round(2),
        "t_statistic"        : round(t_stat, 4),
        "p_value"            : round(p_value, 4),
        "significant"        : p_value < 0.05
    }
    print("\n[INFO] T-Test: Subscriber vs Non-Subscriber")
    print(f"  Mean Subscriber    : ${result['mean_subscriber']}")
    print(f"  Mean Non-Subscriber: ${result['mean_non_subscriber']}")
    print(f"  T-statistic        : {result['t_statistic']}")
    print(f"  P-value            : {result['p_value']}")
    print(f"  Kết luận           : {'Có' if result['significant'] else 'Không có'} sự khác biệt có ý nghĩa (α=0.05)")
    return result


# ============================================================
# ANOVA: DOANH THU THEO MÙA
# ============================================================
def anova_season(df: pd.DataFrame) -> dict:
    """One-way ANOVA kiểm định sự khác biệt doanh thu theo mùa."""
    groups = [
        grp["Purchase Amount (USD)"].dropna().values
        for _, grp in df.groupby("Season", observed=True)
    ]
    f_stat, p_value = stats.f_oneway(*groups)
    result = {
        "f_statistic": round(f_stat, 4),
        "p_value"    : round(p_value, 4),
        "significant": p_value < 0.05
    }
    print("\n[INFO] ANOVA: Doanh thu theo Mùa")
    print(f"  F-statistic: {result['f_statistic']}")
    print(f"  P-value    : {result['p_value']}")
    print(f"  Kết luận   : {'Có' if result['significant'] else 'Không có'} sự khác biệt doanh thu giữa các mùa (α=0.05)")
    return result


# ============================================================
# ANOVA: DOANH THU THEO DANH MỤC
# ============================================================
def anova_category(df: pd.DataFrame) -> dict:
    """One-way ANOVA kiểm định sự khác biệt doanh thu theo danh mục."""
    groups = [
        grp["Purchase Amount (USD)"].dropna().values
        for _, grp in df.groupby("Category", observed=True)
    ]
    f_stat, p_value = stats.f_oneway(*groups)
    result = {
        "f_statistic": round(f_stat, 4),
        "p_value"    : round(p_value, 4),
        "significant": p_value < 0.05
    }
    print("\n[INFO] ANOVA: Doanh thu theo Danh mục")
    print(f"  F-statistic: {result['f_statistic']}")
    print(f"  P-value    : {result['p_value']}")
    print(f"  Kết luận   : {'Có' if result['significant'] else 'Không có'} sự khác biệt doanh thu giữa các danh mục (α=0.05)")
    return result


# ============================================================
# HÀM TỔNG HỢP
# ============================================================
def run_diagnostic(df: pd.DataFrame) -> dict:
    """Chạy toàn bộ phân tích chẩn đoán."""
    print("\n" + "=" * 50)
    print("  BƯỚC 3: PHÂN TÍCH CHẨN ĐOÁN (DIAGNOSTIC)")
    print("=" * 50)

    results = {}
    results["correlation"]  = correlation_analysis(df)
    results["pivot"]        = pivot_analysis(df)
    results["clustered_bar"]= clustered_bar_subscription(df)
    comparative_boxplot_subscription(df)
    results["ttest"]        = ttest_subscription(df)
    results["anova_season"] = anova_season(df)
    results["anova_cat"]    = anova_category(df)

    print("\n[INFO] Phân tích chẩn đoán hoàn tất!\n")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_cleaning import run_cleaning
    df = run_cleaning()
    run_diagnostic(df)
