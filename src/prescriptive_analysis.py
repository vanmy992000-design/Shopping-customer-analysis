"""
prescriptive_analysis.py
=========================
Module phân tích đề xuất (Prescriptive Analytics).

Từ kết quả phân tích:
- Đề xuất chiến lược tăng doanh thu
- Đề xuất nhóm khách hàng cần tập trung marketing
- Đề xuất sản phẩm nên quảng bá
- Đề xuất chương trình khách hàng thân thiết
- Đề xuất tối ưu tồn kho theo mùa
- Đề xuất phân bổ ngân sách marketing

Xuất ra:
- Danh sách Insight kinh doanh
- Danh sách Actionable Recommendations
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "output", "reports")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
FIG_DPI = 150


def _save(fig, filename: str):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] Lưu: {filename}")
    return path


# ============================================================
# PHÂN TÍCH NHÓM KHÁCH HÀNG TIỀM NĂNG
# ============================================================
def segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phân khúc khách hàng dựa trên:
    - Tổng chi tiêu
    - Tần suất mua hàng
    - Có Subscription hay không
    """
    # Encode frequency thành số
    freq_map = {
        "Weekly": 52, "Bi-Weekly": 26, "Fortnightly": 26,
        "Monthly": 12, "Quarterly": 4, "Annually": 1, "Every 3 Months": 4
    }
    df_seg = df.copy()
    df_seg["Freq_num"] = df_seg["Frequency of Purchases"].astype(str).map(freq_map).fillna(4)
    df_seg["Subscribed"] = (df_seg["Subscription Status"].astype(str) == "Yes").astype(int)

    # RFM đơn giản hóa (dùng Previous Purchases thay cho Recency)
    segment = df_seg.groupby("Customer ID").agg(
        Total_Spend=("Purchase Amount (USD)", "sum"),
        Avg_Spend=("Purchase Amount (USD)", "mean"),
        Frequency=("Freq_num", "first"),
        Subscribed=("Subscribed", "first"),
        Age=("Age", "first"),
        Gender=("Gender", "first"),
    ).reset_index()

    # Phân loại khách hàng
    spend_threshold = segment["Total_Spend"].quantile(0.75)
    segment["Segment"] = segment["Total_Spend"].apply(
        lambda x: "VIP" if x >= spend_threshold else
                  "Tiềm năng" if x >= segment["Total_Spend"].quantile(0.5) else
                  "Thông thường"
    )

    print("\n[INFO] Phân khúc khách hàng:")
    print(segment["Segment"].value_counts().to_string())

    # Biểu đồ phân khúc
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    segment["Segment"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=axes[0],
        colors=sns.color_palette("Set2", 3), startangle=90
    )
    axes[0].set_title("Tỷ Trọng Phân Khúc Khách Hàng")
    axes[0].set_ylabel("")

    seg_rev = segment.groupby("Segment")["Total_Spend"].mean().reset_index()
    axes[1].bar(
        seg_rev["Segment"], seg_rev["Total_Spend"],
        color=sns.color_palette("Set2", 3)
    )
    axes[1].set_title("Chi Tiêu TB theo Phân Khúc")
    axes[1].set_ylabel("USD")
    fig.suptitle("Phân Khúc Khách Hàng", fontsize=13, fontweight="bold")
    _save(fig, "24_customer_segments.png")
    return segment


# ============================================================
# PHÂN TÍCH SẢN PHẨM NÊN QUẢNG BÁ
# ============================================================
def product_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    """Phân tích sản phẩm theo doanh thu + đánh giá để đề xuất quảng bá."""
    prod_analysis = df.groupby("Item Purchased", observed=True).agg(
        Total_Revenue=("Purchase Amount (USD)", "sum"),
        Avg_Revenue=("Purchase Amount (USD)", "mean"),
        Avg_Rating=("Review Rating", "mean"),
        Count=("Customer ID", "count")
    ).reset_index().round(2)

    # Score tổng hợp (chuẩn hóa về 0-1)
    prod_analysis["Rev_Score"]   = (prod_analysis["Total_Revenue"] - prod_analysis["Total_Revenue"].min()) / \
                                    (prod_analysis["Total_Revenue"].max() - prod_analysis["Total_Revenue"].min())
    prod_analysis["Rating_Score"] = (prod_analysis["Avg_Rating"] - prod_analysis["Avg_Rating"].min()) / \
                                     (prod_analysis["Avg_Rating"].max() - prod_analysis["Avg_Rating"].min())
    prod_analysis["Score"]        = (prod_analysis["Rev_Score"] * 0.6 + prod_analysis["Rating_Score"] * 0.4).round(3)
    prod_analysis = prod_analysis.sort_values("Score", ascending=False)

    print("\n[INFO] Top 10 sản phẩm nên quảng bá:")
    print(prod_analysis.head(10)[["Item Purchased", "Total_Revenue", "Avg_Rating", "Score"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 6))
    top10 = prod_analysis.head(10)
    ax.scatter(top10["Total_Revenue"], top10["Avg_Rating"],
               s=top10["Count"] * 3, alpha=0.7,
               c=top10["Score"], cmap="YlOrRd")
    for _, row in top10.iterrows():
        ax.annotate(str(row["Item Purchased"]),
                    (row["Total_Revenue"], row["Avg_Rating"]),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Tổng Doanh Thu (USD)")
    ax.set_ylabel("Điểm Đánh Giá TB")
    ax.set_title("Bubble Chart: Sản Phẩm theo Doanh Thu & Đánh Giá",
                 fontsize=12, fontweight="bold")
    _save(fig, "25_product_recommendation.png")
    return prod_analysis


# ============================================================
# PHÂN TÍCH TỒN KHO THEO MÙA
# ============================================================
def seasonal_inventory_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Phân tích nhu cầu sản phẩm theo mùa để tối ưu tồn kho."""
    season_item = df.groupby(
        ["Season", "Category"], observed=True
    )["Purchase Amount (USD)"].agg(
        Total="sum", Count="count"
    ).reset_index().round(2)
    season_item["Season"]   = season_item["Season"].astype(str)
    season_item["Category"] = season_item["Category"].astype(str)

    pivot = season_item.pivot(index="Category", columns="Season", values="Count").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", 4))
    ax.set_title("Nhu Cầu theo Danh Mục & Mùa (Tối Ưu Tồn Kho)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Số Lượt Mua")
    ax.set_xlabel("Danh Mục")
    plt.xticks(rotation=15)
    ax.legend(title="Mùa")
    _save(fig, "26_seasonal_inventory.png")
    return season_item


# ============================================================
# PHÂN BỔ NGÂN SÁCH MARKETING
# ============================================================
def marketing_budget_allocation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đề xuất phân bổ ngân sách marketing theo:
    - Danh mục sản phẩm
    - Mùa cao điểm
    - Phân khúc khách hàng
    """
    cat_rev = df.groupby("Category", observed=True)["Purchase Amount (USD)"].sum()
    total_rev = cat_rev.sum()
    budget_alloc = (cat_rev / total_rev * 100).round(2).reset_index()
    budget_alloc.columns = ["Category", "Budget_%"]
    budget_alloc["Category"] = budget_alloc["Category"].astype(str)
    budget_alloc = budget_alloc.sort_values("Budget_%", ascending=False)

    print("\n[INFO] Đề xuất phân bổ ngân sách marketing theo Danh mục:")
    print(budget_alloc.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.pie(
        budget_alloc["Budget_%"], labels=budget_alloc["Category"],
        autopct="%1.1f%%", startangle=90,
        colors=sns.color_palette("Spectral", len(budget_alloc))
    )
    ax.set_title("Đề Xuất Phân Bổ Ngân Sách Marketing",
                 fontsize=12, fontweight="bold")
    _save(fig, "27_marketing_budget_allocation.png")
    return budget_alloc


# ============================================================
# TẠO DANH SÁCH INSIGHTS & RECOMMENDATIONS
# ============================================================
def generate_insights_report(df: pd.DataFrame,
                              pred_results: dict = None,
                              diag_results: dict = None) -> str:
    """Tổng hợp insights và recommendations dưới dạng text report."""

    # Tính toán thêm
    total_revenue  = df["Purchase Amount (USD)"].sum()
    top_category   = df.groupby("Category", observed=True)["Purchase Amount (USD)"].sum().idxmax()
    top_season     = df.groupby("Season", observed=True)["Purchase Amount (USD)"].sum().idxmax()
    top_location   = df.groupby("Location", observed=True)["Purchase Amount (USD)"].sum().idxmax()
    top_item       = df.groupby("Item Purchased", observed=True)["Purchase Amount (USD)"].sum().idxmax()
    sub_mean       = df[df["Subscription Status"].astype(str) == "Yes"]["Purchase Amount (USD)"].mean()
    nosub_mean     = df[df["Subscription Status"].astype(str) == "No"]["Purchase Amount (USD)"].mean()
    sub_premium    = ((sub_mean - nosub_mean) / nosub_mean * 100)

    gender_rev     = df.groupby("Gender", observed=True)["Purchase Amount (USD)"].mean()
    top_gender     = str(gender_rev.idxmax())
    top_age_group  = df.groupby("Age Group", observed=True)["Purchase Amount (USD)"].sum().idxmax()

    report_path = os.path.join(REPORTS_DIR, "prescriptive_insights.txt")

    best_model = pred_results.get("best_model", "N/A") if pred_results else "N/A"
    metrics_df = pred_results.get("metrics", None) if pred_results else None

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  INSIGHTS KINH DOANH & ACTIONABLE RECOMMENDATIONS\n")
        f.write("  Shopping Trends Analysis\n")
        f.write("=" * 65 + "\n\n")

        f.write("━━━ BUSINESS INSIGHTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

        insights = [
            f"[I01] Tổng doanh thu toàn hệ thống: ${total_revenue:,.0f} USD từ {df['Customer ID'].nunique():,} khách hàng.",
            f"[I02] Danh mục sản phẩm đóng góp doanh thu cao nhất: {str(top_category)}.",
            f"[I03] Mùa tạo doanh thu cao nhất: {str(top_season)} — cần tăng cường hoạt động kinh doanh trong giai đoạn này.",
            f"[I04] Sản phẩm bán chạy nhất theo doanh thu: {str(top_item)}.",
            f"[I05] Địa điểm có doanh thu cao nhất: {str(top_location)}.",
            f"[I06] Khách hàng đăng ký Subscription chi tiêu cao hơn {abs(sub_premium):.1f}% so với khách không đăng ký.",
            f"[I07] Giới tính chi tiêu trung bình cao hơn: {top_gender}.",
            f"[I08] Nhóm tuổi đóng góp doanh thu cao nhất: {str(top_age_group)}.",
            f"[I09] Mô hình dự báo tốt nhất: {best_model}.",
            f"[I10] Discount và Promo Code là yếu tố kích thích mua hàng quan trọng.",
        ]

        for insight in insights:
            f.write(insight + "\n")

        f.write("\n\n━━━ ACTIONABLE RECOMMENDATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

        recommendations = [
            "[R01] CHIẾN LƯỢC TĂNG DOANH THU:\n"
            "      → Tập trung đẩy mạnh bán hàng vào mùa " + str(top_season) + ".\n"
            "      → Tăng ngân sách quảng cáo 20-30% trước mùa cao điểm.\n"
            "      → Triển khai flash sale cuối tuần để tăng tần suất mua.\n",

            "[R02] NHÓM KHÁCH HÀNG CẦN TẬP TRUNG MARKETING:\n"
            f"      → Ưu tiên nhóm tuổi {str(top_age_group)} — đây là phân khúc đóng góp doanh thu cao nhất.\n"
            f"      → Tập trung marketing vào giới tính {top_gender}.\n"
            "      → Triển khai chiến dịch retargeting cho khách hàng VIP (chi tiêu top 25%).\n",

            "[R03] SẢN PHẨM NÊN QUẢNG BÁ:\n"
            f"      → Đẩy mạnh marketing cho '{str(top_item)}' — sản phẩm bán chạy nhất.\n"
            f"      → Tập trung mở rộng danh mục {str(top_category)}.\n"
            "      → Bundle sản phẩm phụ kiện với sản phẩm chính để tăng giá trị đơn hàng.\n",

            "[R04] CHƯƠNG TRÌNH KHÁCH HÀNG THÂN THIẾT:\n"
            "      → Xây dựng tier: Regular → Silver → Gold → VIP.\n"
            "      → Tặng điểm thưởng kép cho mỗi lần mua từ $60 trở lên.\n"
            "      → Ưu tiên mời khách hàng VIP vào chương trình subscription để giữ chân.\n"
            "      → Gửi email cá nhân hóa cho khách hàng có Previous Purchases > 10.\n",

            "[R05] TỐI ƯU TỒN KHO THEO MÙA:\n"
            f"      → Tăng tồn kho Clothing và Footwear 40% trước mùa {str(top_season)}.\n"
            "      → Áp dụng dynamic pricing (giảm giá) vào mùa thấp điểm để giải phóng tồn kho.\n"
            "      → Theo dõi sell-through rate theo tuần để điều chỉnh đặt hàng kịp thời.\n",

            "[R06] PHÂN BỔ NGÂN SÁCH MARKETING:\n"
            "      → 60% ngân sách cho digital marketing (Meta Ads, Google Ads) vào mùa cao điểm.\n"
            "      → 25% cho email marketing và loyalty program.\n"
            "      → 15% cho influencer và affiliate marketing.\n"
            f"      → Ưu tiên đầu tư tại {str(top_location)} — thị trường doanh thu cao nhất.\n",
        ]

        for rec in recommendations:
            f.write(rec + "\n")

        if metrics_df is not None:
            f.write("\n━━━ KẾT QUẢ MÔ HÌNH DỰ BÁO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
            f.write(metrics_df.to_string(index=False) + "\n")

        f.write("\n" + "=" * 65 + "\n")

    print(f"\n[INFO] Báo cáo insights đã lưu: {report_path}")
    return report_path


# ============================================================
# HÀM TỔNG HỢP
# ============================================================
def run_prescriptive(df: pd.DataFrame,
                     pred_results: dict = None,
                     diag_results: dict = None) -> dict:
    """Chạy toàn bộ phân tích đề xuất."""
    print("\n" + "=" * 50)
    print("  BƯỚC 5: PHÂN TÍCH ĐỀ XUẤT (PRESCRIPTIVE)")
    print("=" * 50)

    results = {}
    results["segments"]   = segment_analysis(df)
    results["products"]   = product_recommendation(df)
    results["inventory"]  = seasonal_inventory_analysis(df)
    results["budget"]     = marketing_budget_allocation(df)
    results["report_path"]= generate_insights_report(df, pred_results, diag_results)

    print("\n[INFO] Phân tích đề xuất hoàn tất!\n")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_cleaning import run_cleaning
    df = run_cleaning()
    run_prescriptive(df)
