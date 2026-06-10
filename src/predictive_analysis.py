"""
predictive_analysis.py
=======================
Module phân tích dự báo (Predictive Analytics).

Mục tiêu: Dự đoán Purchase Amount (USD).

Các bước:
1. Feature Engineering
2. One-Hot Encoding
3. Train/Test Split
4. Linear Regression
5. Random Forest Regressor
6. XGBoost Regressor
7. Đánh giá: MAE, RMSE, R² Score
8. So sánh mô hình
9. Trực quan hóa: Actual vs Predicted, Feature Importance, Residual Plot
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost không khả dụng. Bỏ qua.")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
FIG_DPI = 150


def _save(fig, filename: str):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] Lưu: {filename}")
    return path


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo và chuẩn hóa các features cho mô hình dự báo.
    """
    df_fe = df.copy()

    # Chuyển Subscription Status và Discount/Promo sang 0/1
    binary_cols = {
        "Subscription Status": {"Yes": 1, "No": 0},
        "Discount Applied"   : {"Yes": 1, "No": 0},
        "Promo Code Used"    : {"Yes": 1, "No": 0},
    }
    for col, mapping in binary_cols.items():
        if col in df_fe.columns:
            df_fe[col + "_bin"] = df_fe[col].astype(str).map(mapping).fillna(0).astype(int)

    # Mã hóa Frequency of Purchases thành số
    freq_map = {
        "Weekly": 52, "Bi-Weekly": 26, "Fortnightly": 26,
        "Monthly": 12, "Quarterly": 4, "Annually": 1, "Every 3 Months": 4
    }
    df_fe["Frequency_num"] = df_fe["Frequency of Purchases"].astype(str).map(freq_map).fillna(4)

    # One-Hot Encoding cho các biến phân loại chính
    ohe_cols = ["Gender", "Category", "Season", "Size", "Shipping Type"]
    for col in ohe_cols:
        if col in df_fe.columns:
            dummies = pd.get_dummies(df_fe[col].astype(str), prefix=col, drop_first=True)
            df_fe = pd.concat([df_fe, dummies], axis=1)

    print("[INFO] Feature engineering hoàn tất.")
    return df_fe


# ============================================================
# CHUẨN BỊ X, y
# ============================================================
def prepare_features(df_fe: pd.DataFrame):
    """Tách features và target variable."""
    target = "Purchase Amount (USD)"

    # Chọn các cột features
    drop_cols = [
        "Customer ID", "Item Purchased", "Color", "Location",
        "Payment Method", "Preferred Payment Method",
        "Gender", "Category", "Season", "Size", "Shipping Type",
        "Subscription Status", "Discount Applied", "Promo Code Used",
        "Frequency of Purchases", "Age Group",
        target
    ]
    drop_cols = [c for c in drop_cols if c in df_fe.columns]

    feature_cols = [c for c in df_fe.columns if c not in drop_cols]
    # Chỉ giữ cột số
    feature_cols = [c for c in feature_cols if df_fe[c].dtype in [np.int64, np.float64, int, float, bool, np.uint8]]

    X = df_fe[feature_cols].fillna(0)
    y = df_fe[target]

    print(f"[INFO] Số features: {X.shape[1]}")
    print(f"[INFO] Features: {list(X.columns)}")
    return X, y, feature_cols


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """Chia dữ liệu train/test theo tỷ lệ 80/20."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu")
    return X_train, X_test, y_train, y_test


# ============================================================
# HÀM ĐÁNH GIÁ MÔ HÌNH
# ============================================================
def evaluate_model(name: str, y_test, y_pred) -> dict:
    """Tính MAE, RMSE, R² Score."""
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R²   : {r2:.4f}")
    return {"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ============================================================
# BIỂU ĐỒ ACTUAL VS PREDICTED
# ============================================================
def plot_actual_vs_predicted(name: str, y_test, y_pred, filename: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=15, label="Dự đoán")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Lý tưởng")
    ax.set_xlabel("Actual (USD)")
    ax.set_ylabel("Predicted (USD)")
    ax.set_title(f"{name}: Actual vs Predicted", fontsize=12, fontweight="bold")
    ax.legend()
    _save(fig, filename)


# ============================================================
# BIỂU ĐỒ RESIDUAL PLOT
# ============================================================
def plot_residuals(name: str, y_test, y_pred, filename: str):
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter residual
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="coral", s=15)
    axes[0].axhline(0, color="navy", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted (USD)")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"{name}: Residual Plot")

    # Histogram residual
    axes[1].hist(residuals, bins=30, color="coral", edgecolor="white")
    axes[1].axvline(0, color="navy", linestyle="--")
    axes[1].set_xlabel("Residuals")
    axes[1].set_title(f"{name}: Phân Phối Residuals")

    fig.suptitle(f"Residual Analysis - {name}", fontsize=12, fontweight="bold")
    _save(fig, filename)


# ============================================================
# BIỂU ĐỒ FEATURE IMPORTANCE
# ============================================================
def plot_feature_importance(name: str, model, feature_cols: list, filename: str):
    if not hasattr(model, "feature_importances_"):
        return
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(importance))
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
    ax.set_title(f"{name}: Top 15 Feature Importance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    _save(fig, filename)


# ============================================================
# BIỂU ĐỒ SO SÁNH MÔ HÌNH
# ============================================================
def plot_model_comparison(metrics_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["MAE", "RMSE", "R2"]
    colors = sns.color_palette("Set2", len(metrics_df))

    for i, metric in enumerate(metrics):
        bars = axes[i].bar(metrics_df["Model"], metrics_df[metric], color=colors)
        axes[i].set_title(f"So Sánh {metric}")
        axes[i].set_ylabel(metric)
        for bar in bars:
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9
            )
    plt.xticks(rotation=15)
    fig.suptitle("So Sánh Hiệu Suất Các Mô Hình Dự Báo", fontsize=13, fontweight="bold")
    _save(fig, "20_model_comparison.png")


# ============================================================
# HÀM TỔNG HỢP
# ============================================================
def run_predictive(df: pd.DataFrame) -> dict:
    """Chạy toàn bộ quy trình phân tích dự báo."""
    print("\n" + "=" * 50)
    print("  BƯỚC 4: PHÂN TÍCH DỰ BÁO (PREDICTIVE)")
    print("=" * 50)

    # Feature Engineering
    df_fe = feature_engineering(df)
    X, y, feature_cols = prepare_features(df_fe)
    X_train, X_test, y_train, y_test = split_data(X, y)

    metrics_list = []
    models_trained = {}

    # --------------------------------------------------------
    # Mô hình 1: Linear Regression
    # --------------------------------------------------------
    print("\n[INFO] Huấn luyện Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    metrics_list.append(evaluate_model("Linear Regression", y_test, y_pred_lr))
    plot_actual_vs_predicted("Linear Regression", y_test, y_pred_lr, "15_lr_actual_vs_pred.png")
    plot_residuals("Linear Regression", y_test, y_pred_lr, "16_lr_residuals.png")
    models_trained["Linear Regression"] = lr
    joblib.dump(lr, os.path.join(MODELS_DIR, "linear_regression.pkl"))

    # --------------------------------------------------------
    # Mô hình 2: Random Forest
    # --------------------------------------------------------
    print("\n[INFO] Huấn luyện Random Forest (n_estimators=100)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics_list.append(evaluate_model("Random Forest", y_test, y_pred_rf))
    plot_actual_vs_predicted("Random Forest", y_test, y_pred_rf, "17_rf_actual_vs_pred.png")
    plot_residuals("Random Forest", y_test, y_pred_rf, "18_rf_residuals.png")
    plot_feature_importance("Random Forest", rf, feature_cols, "19_rf_feature_importance.png")
    models_trained["Random Forest"] = rf
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))

    # --------------------------------------------------------
    # Mô hình 3: XGBoost
    # --------------------------------------------------------
    if XGBOOST_AVAILABLE:
        print("\n[INFO] Huấn luyện XGBoost...")
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                           random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        metrics_list.append(evaluate_model("XGBoost", y_test, y_pred_xgb))
        plot_actual_vs_predicted("XGBoost", y_test, y_pred_xgb, "21_xgb_actual_vs_pred.png")
        plot_residuals("XGBoost", y_test, y_pred_xgb, "22_xgb_residuals.png")
        plot_feature_importance("XGBoost", xgb, feature_cols, "23_xgb_feature_importance.png")
        models_trained["XGBoost"] = xgb
        joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.pkl"))

    # --------------------------------------------------------
    # So sánh mô hình
    # --------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_list)
    print("\n[INFO] Bảng So Sánh Mô Hình:")
    print(metrics_df.to_string(index=False))
    plot_model_comparison(metrics_df)

    # Chọn mô hình tốt nhất theo R²
    best_idx  = metrics_df["R2"].idxmax()
    best_model_name = metrics_df.loc[best_idx, "Model"]
    print(f"\n[INFO] Mô hình tốt nhất: {best_model_name} (R²={metrics_df.loc[best_idx,'R2']})")

    print("\n[INFO] Phân tích dự báo hoàn tất!\n")
    return {
        "metrics"   : metrics_df,
        "best_model": best_model_name,
        "models"    : models_trained,
        "feature_cols": feature_cols
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_cleaning import run_cleaning
    df = run_cleaning()
    run_predictive(df)
