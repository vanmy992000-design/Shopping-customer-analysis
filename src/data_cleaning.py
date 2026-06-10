"""
data_cleaning.py
================
Module làm sạch và kiểm tra chất lượng dữ liệu từ file shopping_trends.csv.

Chức năng:
- Kiểm tra kiểu dữ liệu
- Kiểm tra giá trị null
- Kiểm tra dữ liệu trùng lặp
- Chuyển đổi kiểu dữ liệu
- Xuất báo cáo chất lượng dữ liệu
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "shopping_trends.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "output", "reports")


# ============================================================
# HÀM ĐỌC DỮ LIỆU
# ============================================================
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu từ file CSV và trả về DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Đọc dữ liệu thành công: {df.shape[0]} dòng, {df.shape[1]} cột")
    return df


# ============================================================
# KIỂM TRA KIỂU DỮ LIỆU
# ============================================================
def check_dtypes(df: pd.DataFrame) -> pd.Series:
    """Kiểm tra và in ra kiểu dữ liệu của từng cột."""
    print("\n[INFO] Kiểu dữ liệu các cột:")
    print(df.dtypes.to_string())
    return df.dtypes


# ============================================================
# KIỂM TRA GIÁ TRỊ NULL
# ============================================================
def check_null(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra số lượng và tỷ lệ giá trị null trong từng cột."""
    null_count = df.isnull().sum()
    null_pct = (null_count / len(df) * 100).round(2)
    null_report = pd.DataFrame({
        "Null Count": null_count,
        "Null (%)": null_pct
    })
    print("\n[INFO] Báo cáo giá trị null:")
    print(null_report[null_report["Null Count"] > 0].to_string() or "  Không có giá trị null!")
    return null_report


# ============================================================
# KIỂM TRA DỮ LIỆU TRÙNG LẶP
# ============================================================
def check_duplicates(df: pd.DataFrame) -> int:
    """Kiểm tra và báo cáo số dòng dữ liệu trùng lặp."""
    dup_count = df.duplicated().sum()
    print(f"\n[INFO] Số dòng trùng lặp: {dup_count}")
    return dup_count


# ============================================================
# CHUYỂN ĐỔI KIỂU DỮ LIỆU
# ============================================================
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển đổi kiểu dữ liệu cần thiết:
    - Các cột phân loại dạng string chuyển sang category
    - Đảm bảo số nguyên đúng kiểu
    """
    df = df.copy()

    # Các cột phân loại
    categorical_cols = [
        "Gender", "Category", "Location", "Size", "Color",
        "Season", "Subscription Status", "Payment Method",
        "Shipping Type", "Discount Applied", "Promo Code Used",
        "Preferred Payment Method", "Frequency of Purchases", "Item Purchased"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Đảm bảo kiểu số
    df["Purchase Amount (USD)"] = pd.to_numeric(df["Purchase Amount (USD)"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")
    df["Previous Purchases"] = pd.to_numeric(df["Previous Purchases"], errors="coerce").astype("Int64")

    # Tạo cột nhóm tuổi (Age Group)
    bins = [0, 18, 25, 35, 45, 55, 100]
    labels = ["<18", "18-25", "26-35", "36-45", "46-55", "56+"]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

    print("\n[INFO] Chuyển đổi kiểu dữ liệu hoàn tất.")
    print(f"  - {len(categorical_cols)} cột phân loại → category")
    print("  - Tạo cột 'Age Group'")
    return df


# ============================================================
# XUẤT BÁO CÁO CHẤT LƯỢNG DỮ LIỆU
# ============================================================
def export_quality_report(df: pd.DataFrame, null_report: pd.DataFrame, dup_count: int) -> str:
    """Xuất báo cáo chất lượng dữ liệu ra file text."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "data_quality_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("       BÁO CÁO CHẤT LƯỢNG DỮ LIỆU - SHOPPING TRENDS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Tổng số dòng     : {df.shape[0]}\n")
        f.write(f"Tổng số cột      : {df.shape[1]}\n")
        f.write(f"Số dòng trùng lặp: {dup_count}\n\n")

        f.write("--- Kiểu dữ liệu các cột ---\n")
        f.write(df.dtypes.to_string() + "\n\n")

        f.write("--- Báo cáo giá trị Null ---\n")
        null_info = null_report[null_report["Null Count"] > 0]
        if null_info.empty:
            f.write("Không có giá trị null trong dữ liệu.\n\n")
        else:
            f.write(null_info.to_string() + "\n\n")

        f.write("--- Thống kê mô tả (cột số) ---\n")
        f.write(df.describe(include=[np.number]).round(3).to_string() + "\n\n")

        f.write("--- Thống kê cột phân loại ---\n")
        cat_cols = df.select_dtypes(include="category").columns
        for col in cat_cols:
            f.write(f"\n{col} ({df[col].nunique()} unique):\n")
            f.write(df[col].value_counts().head(10).to_string() + "\n")

    print(f"\n[INFO] Báo cáo chất lượng dữ liệu đã lưu: {report_path}")
    return report_path


# ============================================================
# HÀM TỔNG HỢP: CHẠY TOÀN BỘ QUY TRÌNH CLEANING
# ============================================================
def run_cleaning() -> pd.DataFrame:
    """Chạy toàn bộ quy trình làm sạch dữ liệu và trả về DataFrame đã xử lý."""
    print("=" * 50)
    print("  BƯỚC 1: LÀM SẠCH DỮ LIỆU")
    print("=" * 50)

    # Bước 1: Đọc dữ liệu
    df = load_data()

    # Bước 2: Kiểm tra kiểu dữ liệu
    check_dtypes(df)

    # Bước 3: Kiểm tra null
    null_report = check_null(df)

    # Bước 4: Kiểm tra trùng lặp
    dup_count = check_duplicates(df)

    # Bước 5: Chuyển đổi kiểu dữ liệu
    df = convert_dtypes(df)

    # Bước 6: Xuất báo cáo
    export_quality_report(df, null_report, dup_count)

    print("\n[INFO] Làm sạch dữ liệu hoàn tất!\n")
    return df


if __name__ == "__main__":
    df_clean = run_cleaning()
    print(df_clean.head())
