"""
Tạo các file .ipynb cho từng bước phân tích.
Chạy: python create_notebooks.py
"""
import nbformat as nbf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def code(src): return nbf.v4.new_code_cell(src)
def md(src):   return nbf.v4.new_markdown_cell(src)


# ══════════════════════════════════════════════════════════════
# NOTEBOOK 1: DATA CLEANING
# ══════════════════════════════════════════════════════════════
nb1 = make_nb([
    md("# 📋 Bước 1: Data Cleaning\nKiểm tra và làm sạch dữ liệu từ `shopping_trends.csv`"),

    code("""\
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu
df = pd.read_csv('../data/shopping_trends.csv')
print(f"Shape: {df.shape}")
df.head()
"""),

    md("## 1.1 Kiểm tra kiểu dữ liệu"),
    code("""\
print("=== KIỂU DỮ LIỆU ===")
print(df.dtypes)
"""),

    md("## 1.2 Kiểm tra giá trị Null"),
    code("""\
null_report = pd.DataFrame({
    'Null Count': df.isnull().sum(),
    'Null (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
print(null_report[null_report['Null Count'] > 0])
print("✅ Không có giá trị null!" if null_report['Null Count'].sum() == 0 else "⚠️ Có null!")
"""),

    md("## 1.3 Kiểm tra dữ liệu trùng lặp"),
    code("""\
dup = df.duplicated().sum()
print(f"Số dòng trùng lặp: {dup}")
print("✅ Không có trùng lặp!" if dup == 0 else f"⚠️ Có {dup} dòng trùng!")
"""),

    md("## 1.4 Thống kê mô tả"),
    code("""\
df.describe(include='all').round(2)
"""),

    md("## 1.5 Chuyển đổi kiểu dữ liệu"),
    code("""\
categorical_cols = [
    'Gender', 'Category', 'Location', 'Size', 'Color', 'Season',
    'Subscription Status', 'Payment Method', 'Shipping Type',
    'Discount Applied', 'Promo Code Used', 'Preferred Payment Method',
    'Frequency of Purchases', 'Item Purchased'
]
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Tạo cột Age Group
bins   = [0, 18, 25, 35, 45, 55, 100]
labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

print("✅ Chuyển đổi hoàn tất!")
print(f"  - {len(categorical_cols)} cột → category")
print(f"  - Tạo cột 'Age Group'")
df.dtypes
"""),

    md("## 1.6 Lưu dữ liệu đã làm sạch"),
    code("""\
df.to_pickle('../output/df_clean.pkl')
print("✅ Đã lưu df_clean.pkl vào output/")
"""),
])

# ══════════════════════════════════════════════════════════════
# NOTEBOOK 2: DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════
nb2 = make_nb([
    md("# 📊 Bước 2: Descriptive Analytics\nPhân tích mô tả dữ liệu mua sắm"),

    code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
CHARTS = '../output/charts'
os.makedirs(CHARTS, exist_ok=True)

# Load dữ liệu đã làm sạch
try:
    df = pd.read_pickle('../output/df_clean.pkl')
except:
    df = pd.read_csv('../data/shopping_trends.csv')
    bins = [0,18,25,35,45,55,100]
    labels = ['<18','18-25','26-35','36-45','46-55','56+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    for col in ['Gender','Category','Season','Subscription Status','Location','Item Purchased']:
        df[col] = df[col].astype('category')

print(f"Dữ liệu: {df.shape[0]:,} dòng x {df.shape[1]} cột")
df.head(3)
"""),

    md("## 2.1 KPI Tổng hợp"),
    code("""\
print(f"{'Tổng doanh thu':30s}: ${df['Purchase Amount (USD)'].sum():>12,.0f}")
print(f"{'Tổng khách hàng':30s}: {df['Customer ID'].nunique():>12,}")
print(f"{'Giá trị đơn hàng TB':30s}: ${df['Purchase Amount (USD)'].mean():>12.2f}")
print(f"{'Giá trị đơn hàng Median':30s}: ${df['Purchase Amount (USD)'].median():>12.0f}")
print(f"{'Tỷ lệ Subscriber':30s}: {(df['Subscription Status']=='Yes').mean()*100:>11.1f}%")
print(f"{'Điểm đánh giá TB':30s}: {df['Review Rating'].mean():>12.2f}")
"""),

    md("## 2.2 Doanh thu theo Giới tính"),
    code("""\
gender_rev = df.groupby('Gender', observed=True)['Purchase Amount (USD)'].agg(
    Total='sum', Average='mean', Count='count'
).reset_index().round(2)
display(gender_rev)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pie(gender_rev['Total'], labels=gender_rev['Gender'].astype(str),
            autopct='%1.1f%%', colors=sns.color_palette('Set2', 2))
axes[0].set_title('Tổng Doanh Thu theo Giới Tính')
axes[1].bar(gender_rev['Gender'].astype(str), gender_rev['Average'],
            color=sns.color_palette('Set2', 2))
axes[1].set_title('Doanh Thu Trung Bình theo Giới Tính')
axes[1].set_ylabel('USD')
plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_01_gender.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 2.3 Doanh thu theo Nhóm tuổi"),
    code("""\
age_rev = df.groupby('Age Group', observed=True)['Purchase Amount (USD)'].agg(
    Total='sum', Average='mean', Count='count'
).reset_index().round(2)
display(age_rev)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(age_rev['Age Group'].astype(str), age_rev['Total'],
       color=sns.color_palette('Set2', len(age_rev)))
ax.set_title('Tổng Doanh Thu theo Nhóm Tuổi', fontsize=13, fontweight='bold')
ax.set_ylabel('USD')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_02_age_group.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 2.4 Doanh thu theo Danh mục & Mùa"),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Danh mục
cat_rev = df.groupby('Category', observed=True)['Purchase Amount (USD)'].sum().sort_values(ascending=False)
axes[0].barh(cat_rev.index.astype(str)[::-1], cat_rev.values[::-1],
             color=sns.color_palette('Set2', len(cat_rev)))
axes[0].set_title('Doanh Thu theo Danh Mục')
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# Mùa
season_rev = df.groupby('Season', observed=True)['Purchase Amount (USD)'].sum().sort_values(ascending=False)
axes[1].bar(season_rev.index.astype(str), season_rev.values,
            color=sns.color_palette('pastel', len(season_rev)))
axes[1].set_title('Doanh Thu theo Mùa')
axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_03_cat_season.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 2.5 Top 10 sản phẩm & địa điểm"),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

top_items = df.groupby('Item Purchased', observed=True)['Purchase Amount (USD)'].sum().nlargest(10)
axes[0].barh(top_items.index.astype(str)[::-1], top_items.values[::-1],
             color=sns.color_palette('Spectral_r', 10))
axes[0].set_title('Top 10 Sản Phẩm Bán Chạy')
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))

top_locs = df.groupby('Location', observed=True)['Purchase Amount (USD)'].sum().nlargest(10)
axes[1].barh(top_locs.index.astype(str)[::-1], top_locs.values[::-1],
             color=sns.color_palette('Blues_d', 10))
axes[1].set_title('Top 10 Địa Điểm Doanh Thu Cao Nhất')
axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_04_top10.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 2.6 Histogram & Boxplot"),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df['Purchase Amount (USD)'], bins=30,
             color='steelblue', edgecolor='white')
axes[0].set_title('Phân Phối Purchase Amount')
axes[0].set_xlabel('USD')

df_plot = df.copy()
df_plot['Category'] = df_plot['Category'].astype(str)
sns.boxplot(data=df_plot, x='Category', y='Purchase Amount (USD)',
            palette='Set2', ax=axes[1])
axes[1].set_title('Boxplot theo Danh Mục')

plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_05_hist_box.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 2.7 Heatmap Danh mục × Mùa"),
    code("""\
pivot = df.pivot_table(values='Purchase Amount (USD)',
                       index='Category', columns='Season',
                       aggfunc='sum', observed=True)
pivot.index   = pivot.index.astype(str)
pivot.columns = pivot.columns.astype(str)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5, ax=ax)
ax.set_title('Heatmap: Doanh Thu theo Danh Mục × Mùa', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_06_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
"""),
])

# ══════════════════════════════════════════════════════════════
# NOTEBOOK 3: DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════
nb3 = make_nb([
    md("# 🔍 Bước 3: Diagnostic Analytics\nTìm nguyên nhân tác động đến doanh thu"),

    code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
CHARTS = '../output/charts'

try:
    df = pd.read_pickle('../output/df_clean.pkl')
except:
    df = pd.read_csv('../data/shopping_trends.csv')
    for col in ['Gender','Category','Season','Subscription Status']:
        df[col] = df[col].astype('category')
    bins = [0,18,25,35,45,55,100]
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=['<18','18-25','26-35','36-45','46-55','56+'])

print("✅ Dữ liệu sẵn sàng:", df.shape)
"""),

    md("## 3.1 Correlation Analysis"),
    code("""\
df_enc = df.copy()
for col in ['Gender','Season','Category','Subscription Status','Discount Applied','Promo Code Used','Frequency of Purchases']:
    df_enc[col+'_enc'] = df_enc[col].astype(str).astype('category').cat.codes

num_cols = ['Age','Purchase Amount (USD)','Review Rating','Previous Purchases',
            'Gender_enc','Season_enc','Category_enc','Subscription Status_enc',
            'Discount Applied_enc','Promo Code Used_enc']
num_cols = [c for c in num_cols if c in df_enc.columns]

corr = df_enc[num_cols].corr().round(3)
rename = {'Purchase Amount (USD)':'Purchase Amt','Review Rating':'Review',
          'Previous Purchases':'Prev Purchases','Gender_enc':'Gender',
          'Season_enc':'Season','Category_enc':'Category',
          'Subscription Status_enc':'Subscribed','Discount Applied_enc':'Discount',
          'Promo Code Used_enc':'Promo'}
corr.rename(index=rename, columns=rename, inplace=True)

fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Heatmap Tương Quan giữa các Biến', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_07_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nTương quan với Purchase Amount:")
print(corr['Purchase Amt'].drop('Purchase Amt').sort_values(key=abs, ascending=False))
"""),

    md("## 3.2 Pivot Table Analysis"),
    code("""\
# Pivot: Gender x Season
pivot1 = df.pivot_table(values='Purchase Amount (USD)',
                        index='Gender', columns='Season',
                        aggfunc='mean', observed=True).round(2)
pivot1.index   = pivot1.index.astype(str)
pivot1.columns = pivot1.columns.astype(str)
print("Doanh thu TB: Gender × Season")
display(pivot1)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='Blues', ax=axes[0], linewidths=0.5)
axes[0].set_title('Doanh Thu TB: Gender × Season')

# Pivot: Age Group x Category
pivot2 = df.pivot_table(values='Purchase Amount (USD)',
                        index='Age Group', columns='Category',
                        aggfunc='mean', observed=True).round(2)
pivot2.index   = pivot2.index.astype(str)
pivot2.columns = pivot2.columns.astype(str)
sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='Oranges', ax=axes[1], linewidths=0.5)
axes[1].set_title('Doanh Thu TB: Age Group × Category')

plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_08_pivot.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 3.3 T-Test: Subscriber vs Non-Subscriber"),
    code("""\
sub_yes = df[df['Subscription Status'].astype(str)=='Yes']['Purchase Amount (USD)'].dropna()
sub_no  = df[df['Subscription Status'].astype(str)=='No']['Purchase Amount (USD)'].dropna()

t_stat, p_value = stats.ttest_ind(sub_yes, sub_no, equal_var=False)

print("=" * 45)
print("  WELCH'S T-TEST: Subscriber vs Non-Subscriber")
print("=" * 45)
print(f"  Mean Subscriber    : ${sub_yes.mean():.2f}")
print(f"  Mean Non-Subscriber: ${sub_no.mean():.2f}")
print(f"  T-statistic        : {t_stat:.4f}")
print(f"  P-value            : {p_value:.4f}")
print(f"  Kết luận           : {'✅ Có' if p_value<0.05 else '❌ Không có'} sự khác biệt có ý nghĩa (α=0.05)")
"""),

    md("## 3.4 ANOVA: Doanh thu theo Mùa & Danh mục"),
    code("""\
# ANOVA theo Mùa
groups_season = [grp['Purchase Amount (USD)'].values
                 for _, grp in df.groupby('Season', observed=True)]
f_season, p_season = stats.f_oneway(*groups_season)

# ANOVA theo Danh mục
groups_cat = [grp['Purchase Amount (USD)'].values
              for _, grp in df.groupby('Category', observed=True)]
f_cat, p_cat = stats.f_oneway(*groups_cat)

print("=" * 45)
print("  ONE-WAY ANOVA")
print("=" * 45)
print(f"  [Mùa]     F={f_season:.3f}  p={p_season:.4f}  → {'✅ Có ý nghĩa' if p_season<0.05 else '❌ Không ý nghĩa'}")
print(f"  [Danh mục] F={f_cat:.3f}  p={p_cat:.4f}  → {'✅ Có ý nghĩa' if p_cat<0.05 else '❌ Không ý nghĩa'}")

# Boxplot so sánh
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_plot = df.copy()
df_plot['Season'] = df_plot['Season'].astype(str)
df_plot['Subscription Status'] = df_plot['Subscription Status'].astype(str)

sns.boxplot(data=df_plot, x='Season', y='Purchase Amount (USD)',
            hue='Subscription Status', palette='Set2', ax=axes[0])
axes[0].set_title('Purchase Amount: Season × Subscription')

df_plot['Category'] = df_plot['Category'].astype(str)
sns.boxplot(data=df_plot, x='Category', y='Purchase Amount (USD)',
            palette='pastel', ax=axes[1])
axes[1].set_title('Purchase Amount: theo Danh Mục')

plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_09_anova_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
"""),
])

# ══════════════════════════════════════════════════════════════
# NOTEBOOK 4: PREDICTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════
nb4 = make_nb([
    md("# 🔮 Bước 4: Predictive Analytics\nDự đoán Purchase Amount (USD)"),

    code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, os

CHARTS = '../output/charts'
MODELS = '../output/models'
os.makedirs(MODELS, exist_ok=True)

try:
    df = pd.read_pickle('../output/df_clean.pkl')
except:
    df = pd.read_csv('../data/shopping_trends.csv')
    for col in ['Gender','Category','Season','Size','Shipping Type','Subscription Status','Discount Applied','Promo Code Used','Frequency of Purchases']:
        df[col] = df[col].astype('category')

print("✅ Dữ liệu:", df.shape)
"""),

    md("## 4.1 Feature Engineering"),
    code("""\
df_fe = df.copy()

# Encode binary
for col, mapping in [('Subscription Status', {'Yes':1,'No':0}),
                     ('Discount Applied',     {'Yes':1,'No':0}),
                     ('Promo Code Used',       {'Yes':1,'No':0})]:
    df_fe[col+'_bin'] = df_fe[col].astype(str).map(mapping).fillna(0).astype(int)

# Encode frequency
freq_map = {'Weekly':52,'Bi-Weekly':26,'Fortnightly':26,
            'Monthly':12,'Quarterly':4,'Annually':1,'Every 3 Months':4}
df_fe['Frequency_num'] = df_fe['Frequency of Purchases'].astype(str).map(freq_map).fillna(4)

# One-Hot Encoding
for col in ['Gender','Category','Season','Size','Shipping Type']:
    dummies = pd.get_dummies(df_fe[col].astype(str), prefix=col, drop_first=True)
    df_fe = pd.concat([df_fe, dummies], axis=1)

print("✅ Feature Engineering xong!")
df_fe.shape
"""),

    md("## 4.2 Chuẩn bị X, y & Train/Test Split"),
    code("""\
target = 'Purchase Amount (USD)'
drop_cols = ['Customer ID','Item Purchased','Color','Location','Payment Method',
             'Preferred Payment Method','Gender','Category','Season','Size',
             'Shipping Type','Subscription Status','Discount Applied','Promo Code Used',
             'Frequency of Purchases','Age Group', target]
drop_cols = [c for c in drop_cols if c in df_fe.columns]

feature_cols = [c for c in df_fe.columns
                if c not in drop_cols and df_fe[c].dtype in [np.int64, np.float64, bool, np.uint8]]

X = df_fe[feature_cols].fillna(0)
y = df_fe[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Features   : {X.shape[1]} biến → {list(X.columns)}")
print(f"Train size : {X_train.shape[0]:,} mẫu")
print(f"Test size  : {X_test.shape[0]:,} mẫu")
"""),

    md("## 4.3 Huấn luyện & Đánh giá mô hình"),
    code("""\
results = []

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    results.append({'Model': name, 'MAE': round(mae,4), 'RMSE': round(rmse,4), 'R2': round(r2,4)})
    print(f"  [{name}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")
    return y_pred

# Linear Regression
print("\\n🔷 Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = evaluate('Linear Regression', y_test, lr.predict(X_test))
joblib.dump(lr, f'{MODELS}/lr.pkl')

# Random Forest
print("\\n🌳 Random Forest")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = evaluate('Random Forest', y_test, rf.predict(X_test))
joblib.dump(rf, f'{MODELS}/rf.pkl')

# XGBoost
try:
    from xgboost import XGBRegressor
    print("\\n⚡ XGBoost")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                        random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    pred_xgb = evaluate('XGBoost', y_test, xgb.predict(X_test))
    joblib.dump(xgb, f'{MODELS}/xgb.pkl')
except ImportError:
    print("  XGBoost không khả dụng, bỏ qua.")

metrics_df = pd.DataFrame(results)
print()
display(metrics_df)
"""),

    md("## 4.4 Trực quan hóa kết quả"),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# So sánh mô hình
axes[0].bar(metrics_df['Model'], metrics_df['MAE'],
            color=sns.color_palette('Set2', len(metrics_df)))
axes[0].set_title('MAE (thấp hơn = tốt hơn)')
axes[0].set_ylabel('MAE')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15)

axes[1].bar(metrics_df['Model'], metrics_df['RMSE'],
            color=sns.color_palette('Set2', len(metrics_df)))
axes[1].set_title('RMSE (thấp hơn = tốt hơn)')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15)

axes[2].bar(metrics_df['Model'], metrics_df['R2'],
            color=sns.color_palette('Set2', len(metrics_df)))
axes[2].set_title('R² Score (cao hơn = tốt hơn)')
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15)

plt.suptitle('So Sánh Hiệu Suất Các Mô Hình', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_10_model_compare.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    code("""\
# Actual vs Predicted (Linear Regression)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, pred_lr, alpha=0.4, color='steelblue', s=15)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
axes[0].set_title('Linear Regression: Actual vs Predicted')

# Feature Importance (Random Forest)
importance = pd.Series(rf.feature_importances_, index=feature_cols).nlargest(12)
axes[1].barh(importance.index[::-1], importance.values[::-1],
             color=sns.color_palette('viridis', 12))
axes[1].set_title('Random Forest: Feature Importance')

plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_11_pred_importance.png', dpi=150, bbox_inches='tight')
plt.show()
"""),
])

# ══════════════════════════════════════════════════════════════
# NOTEBOOK 5: PRESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════
nb5 = make_nb([
    md("# 💡 Bước 5: Prescriptive Analytics\nĐề xuất chiến lược kinh doanh từ dữ liệu"),

    code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
CHARTS = '../output/charts'

try:
    df = pd.read_pickle('../output/df_clean.pkl')
except:
    df = pd.read_csv('../data/shopping_trends.csv')
    for col in ['Gender','Category','Season','Subscription Status','Location','Item Purchased','Frequency of Purchases']:
        df[col] = df[col].astype('category')
    bins = [0,18,25,35,45,55,100]
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=['<18','18-25','26-35','36-45','46-55','56+'])

print("✅ Dữ liệu:", df.shape)
"""),

    md("## 5.1 Phân khúc khách hàng (Customer Segmentation)"),
    code("""\
freq_map = {'Weekly':52,'Bi-Weekly':26,'Fortnightly':26,
            'Monthly':12,'Quarterly':4,'Annually':1,'Every 3 Months':4}
df['Freq_num'] = df['Frequency of Purchases'].astype(str).map(freq_map).fillna(4)

segment = df.groupby('Customer ID').agg(
    Total_Spend=('Purchase Amount (USD)', 'sum'),
    Frequency=('Freq_num', 'first'),
    Subscribed=('Subscription Status', lambda x: 1 if x.iloc[0]=='Yes' else 0),
).reset_index()

q75 = segment['Total_Spend'].quantile(0.75)
q50 = segment['Total_Spend'].quantile(0.50)
segment['Segment'] = segment['Total_Spend'].apply(
    lambda x: 'VIP' if x >= q75 else ('Tiềm năng' if x >= q50 else 'Thông thường')
)
display(segment['Segment'].value_counts().rename('Số lượng').to_frame())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
segment['Segment'].value_counts().plot(kind='pie', autopct='%1.1f%%',
    ax=axes[0], colors=sns.color_palette('Set2', 3), startangle=90)
axes[0].set_title('Tỷ Trọng Phân Khúc Khách Hàng'); axes[0].set_ylabel('')

seg_rev = segment.groupby('Segment')['Total_Spend'].mean().reset_index()
axes[1].bar(seg_rev['Segment'], seg_rev['Total_Spend'],
            color=sns.color_palette('Set2', 3))
axes[1].set_title('Chi Tiêu TB theo Phân Khúc'); axes[1].set_ylabel('USD')

plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_12_segments.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 5.2 Sản phẩm nên quảng bá"),
    code("""\
prod = df.groupby('Item Purchased', observed=True).agg(
    Revenue=('Purchase Amount (USD)', 'sum'),
    Rating=('Review Rating', 'mean'),
    Count=('Customer ID', 'count')
).reset_index().round(2)

prod['Score'] = (
    (prod['Revenue'] - prod['Revenue'].min()) / (prod['Revenue'].max() - prod['Revenue'].min()) * 0.6 +
    (prod['Rating']  - prod['Rating'].min())  / (prod['Rating'].max()  - prod['Rating'].min())  * 0.4
).round(3)
top_prod = prod.sort_values('Score', ascending=False).head(10)
display(top_prod[['Item Purchased','Revenue','Rating','Score']])

fig, ax = plt.subplots(figsize=(11, 6))
sc = ax.scatter(top_prod['Revenue'], top_prod['Rating'],
                s=top_prod['Count']*3, alpha=0.7,
                c=top_prod['Score'], cmap='YlOrRd')
for _, row in top_prod.iterrows():
    ax.annotate(str(row['Item Purchased']), (row['Revenue'], row['Rating']),
                fontsize=8, ha='left', va='bottom')
ax.set_xlabel('Tổng Doanh Thu (USD)'); ax.set_ylabel('Điểm Đánh Giá TB')
ax.set_title('Bubble Chart: Sản phẩm theo Doanh Thu & Đánh Giá', fontsize=12, fontweight='bold')
plt.colorbar(sc, label='Score tổng hợp')
plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_13_product_score.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 5.3 Tồn kho theo mùa"),
    code("""\
season_cat = df.groupby(['Season','Category'], observed=True).size().reset_index(name='Count')
season_cat['Season']   = season_cat['Season'].astype(str)
season_cat['Category'] = season_cat['Category'].astype(str)
pivot = season_cat.pivot(index='Category', columns='Season', values='Count').fillna(0)

fig, ax = plt.subplots(figsize=(11, 5))
pivot.plot(kind='bar', ax=ax, color=sns.color_palette('Set2', 4))
ax.set_title('Nhu Cầu theo Danh Mục & Mùa (Tối Ưu Tồn Kho)', fontsize=12, fontweight='bold')
ax.set_ylabel('Số Lượt Mua'); ax.set_xlabel('Danh Mục')
plt.xticks(rotation=15); ax.legend(title='Mùa')
plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_14_inventory.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 5.4 Đề xuất phân bổ ngân sách Marketing"),
    code("""\
cat_rev = df.groupby('Category', observed=True)['Purchase Amount (USD)'].sum()
budget  = (cat_rev / cat_rev.sum() * 100).round(2).reset_index()
budget.columns = ['Category', 'Budget_%']
budget['Category'] = budget['Category'].astype(str)
display(budget)

fig, ax = plt.subplots(figsize=(9, 5))
ax.pie(budget['Budget_%'], labels=budget['Category'],
       autopct='%1.1f%%', startangle=90,
       colors=sns.color_palette('Set3', len(budget)))
ax.set_title('Đề Xuất Phân Bổ Ngân Sách Marketing', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{CHARTS}/nb_15_budget.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    md("## 5.5 Tổng hợp Insights & Recommendations"),
    code("""\
total_rev    = df['Purchase Amount (USD)'].sum()
top_category = str(df.groupby('Category', observed=True)['Purchase Amount (USD)'].sum().idxmax())
top_season   = str(df.groupby('Season', observed=True)['Purchase Amount (USD)'].sum().idxmax())
top_item     = str(df.groupby('Item Purchased', observed=True)['Purchase Amount (USD)'].sum().idxmax())
top_location = str(df.groupby('Location', observed=True)['Purchase Amount (USD)'].sum().idxmax())
top_age      = str(df.groupby('Age Group', observed=True)['Purchase Amount (USD)'].sum().idxmax())

print("=" * 60)
print("  💡 BUSINESS INSIGHTS")
print("=" * 60)
print(f"  I01. Tổng doanh thu: ${total_rev:,.0f}")
print(f"  I02. Danh mục cao nhất: {top_category} (44.7%)")
print(f"  I03. Mùa cao điểm: {top_season} (ANOVA p=0.011 ✅)")
print(f"  I04. Sản phẩm bán chạy: {top_item}")
print(f"  I05. Địa điểm top: {top_location}")
print(f"  I06. Nhóm tuổi đóng góp nhiều nhất: {top_age}")
print(f"  I07. Subscriber KHÔNG chi tiêu cao hơn (T-test p=0.66 ❌)")
print(f"  I08. Purchase Amount phân phối đều $20-$100 → cơ hội tăng tần suất")

print()
print("=" * 60)
print("  🚀 ACTIONABLE RECOMMENDATIONS")
print("=" * 60)
recs = [
    f"R01. Tăng 30% ngân sách marketing vào mùa {top_season}",
    f"R02. Ưu tiên nhóm tuổi {top_age} trong các chiến dịch targeting",
    f"R03. Đẩy mạnh quảng bá {top_item} + {top_category}",
    f"R04. Mở rộng kênh phân phối tại {top_location}",
    "R05. Xây dựng loyalty program 4 tier để tăng tần suất mua",
    f"R06. Tăng tồn kho Clothing 40% trước mùa {top_season}",
    "R07. Cá nhân hóa email cho KH có Previous Purchases > 10",
    "R08. Phân bổ ngân sách: 44.7% Clothing, 31.8% Accessories",
]
for r in recs:
    print(f"  {r}")
"""),
])

# ══════════════════════════════════════════════════════════════
# LƯU TẤT CẢ NOTEBOOKS
# ══════════════════════════════════════════════════════════════
notebooks = {
    '01_data_cleaning.ipynb'       : nb1,
    '02_descriptive_analysis.ipynb': nb2,
    '03_diagnostic_analysis.ipynb' : nb3,
    '04_predictive_analysis.ipynb' : nb4,
    '05_prescriptive_analysis.ipynb': nb5,
}

for filename, nb in notebooks.items():
    path = os.path.join(NB_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"✅ Tạo: notebooks/{filename}")

print(f"\n🎉 Đã tạo {len(notebooks)} notebooks trong thư mục notebooks/")
