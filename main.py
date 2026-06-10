"""
main.py
=======
File khởi chạy chính của project phân tích Shopping Trends.

Thứ tự thực thi:
1. Data Cleaning          → Làm sạch & kiểm tra chất lượng dữ liệu
2. Descriptive Analytics  → Phân tích mô tả
3. Diagnostic Analytics   → Phân tích chẩn đoán
4. Predictive Analytics   → Phân tích dự báo
5. Prescriptive Analytics → Đề xuất chiến lược kinh doanh
6. (Optional) Dashboard   → Chạy riêng bằng: streamlit run src/dashboard.py

Cách chạy:
    python main.py
    python main.py --skip-predict   (bỏ qua predictive để chạy nhanh hơn)
"""

import sys
import os
import time
import argparse

# Thêm src/ vào Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))


def print_banner():
    """In banner chào mừng."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       SHOPPING TRENDS - PHÂN TÍCH DỮ LIỆU TOÀN DIỆN         ║
║       FAA4021 | Phân Tích Doanh Nghiệp                       ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Shopping Trends Data Analysis")
    parser.add_argument(
        "--skip-predict", action="store_true",
        help="Bỏ qua bước Predictive Analytics (chạy nhanh hơn)"
    )
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()
    start_time = time.time()

    try:
        # --------------------------------------------------------
        # BƯỚC 1: DATA CLEANING
        # --------------------------------------------------------
        from data_cleaning import run_cleaning
        df = run_cleaning()

        # --------------------------------------------------------
        # BƯỚC 2: DESCRIPTIVE ANALYTICS
        # --------------------------------------------------------
        from descriptive_analysis import run_descriptive
        desc_results = run_descriptive(df)

        # --------------------------------------------------------
        # BƯỚC 3: DIAGNOSTIC ANALYTICS
        # --------------------------------------------------------
        from diagnostic_analysis import run_diagnostic
        diag_results = run_diagnostic(df)

        # --------------------------------------------------------
        # BƯỚC 4: PREDICTIVE ANALYTICS
        # --------------------------------------------------------
        pred_results = None
        if not args.skip_predict:
            from predictive_analysis import run_predictive
            pred_results = run_predictive(df)
        else:
            print("\n[SKIP] Bỏ qua Predictive Analytics (--skip-predict)")

        # --------------------------------------------------------
        # BƯỚC 5: PRESCRIPTIVE ANALYTICS
        # --------------------------------------------------------
        from prescriptive_analysis import run_prescriptive
        presc_results = run_prescriptive(df, pred_results, diag_results)

        # --------------------------------------------------------
        # TỔNG KẾT
        # --------------------------------------------------------
        elapsed = time.time() - start_time
        print("\n" + "=" * 65)
        print("  ✅ PHÂN TÍCH HOÀN TẤT!")
        print(f"  ⏱️  Thời gian thực thi: {elapsed:.1f} giây")
        print("=" * 65)
        print(f"\n📂 Kết quả lưu tại:")
        print(f"   → {os.path.join(BASE_DIR, 'output', 'charts')}   (biểu đồ)")
        print(f"   → {os.path.join(BASE_DIR, 'output', 'reports')}  (báo cáo)")
        if pred_results:
            print(f"   → {os.path.join(BASE_DIR, 'output', 'models')}   (mô hình)")
        print("\n🚀 Để xem Dashboard, chạy lệnh:")
        print(f"   streamlit run {os.path.join(BASE_DIR, 'src', 'dashboard.py')}")
        print()

    except Exception as e:
        print(f"\n[ERROR] Đã xảy ra lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
