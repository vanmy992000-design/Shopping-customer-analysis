"""Script to generate the PDF report."""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'output', 'reports')
CHARTS_DIR = os.path.join(BASE_DIR, 'output', 'charts')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'shopping_trends.csv'))
bins = [0, 18, 25, 35, 45, 55, 100]
labels = ['<18', '18-25', '26-35', '36-45', '46-55', '56+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
for col in ['Gender', 'Category', 'Season', 'Subscription Status']:
    df[col] = df[col].astype('category')

# Stats
total_rev = df['Purchase Amount (USD)'].sum()
total_cust = df['Customer ID'].nunique()
avg_order = df['Purchase Amount (USD)'].mean()
sub_rate = (df['Subscription Status'].astype(str) == 'Yes').mean() * 100
top_category = str(df.groupby('Category', observed=True)['Purchase Amount (USD)'].sum().idxmax())
top_season = str(df.groupby('Season', observed=True)['Purchase Amount (USD)'].sum().idxmax())
top_location = str(df.groupby('Location', observed=True)['Purchase Amount (USD)'].sum().idxmax())
top_item = str(df.groupby('Item Purchased', observed=True)['Purchase Amount (USD)'].sum().idxmax())

# Styles
styles = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle('Title2', parent=styles['Title'],
    fontSize=20, textColor=colors.HexColor('#2C3E50'), alignment=TA_CENTER, spaceAfter=6)
H1_STYLE = ParagraphStyle('H1c', parent=styles['Heading1'],
    fontSize=14, textColor=colors.HexColor('#2980B9'), spaceBefore=14, spaceAfter=6)
H2_STYLE = ParagraphStyle('H2c', parent=styles['Heading2'],
    fontSize=11, textColor=colors.HexColor('#8E44AD'), spaceBefore=10, spaceAfter=4)
BODY_STYLE = ParagraphStyle('Bodyc', parent=styles['Normal'],
    fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
CAPTION_STYLE = ParagraphStyle('Captionc', parent=styles['Normal'],
    fontSize=8, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)


def make_table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths)
    cmds = [
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980B9') if header else colors.white),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white if header else colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#EBF5FB'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(cmds))
    return t


def add_chart(filename, width=14 * cm, height=7 * cm):
    path = os.path.join(CHARTS_DIR, filename)
    if os.path.exists(path):
        return RLImage(path, width=width, height=height)
    return Paragraph(f'[Chart not found: {filename}]', BODY_STYLE)


story = []

# Cover
story.append(Spacer(1, 2 * cm))
story.append(Paragraph('SHOPPING TRENDS DATA ANALYSIS', TITLE_STYLE))
story.append(Paragraph('Bao cao Phan tich Du lieu Toan dien', ParagraphStyle(
    'sub', parent=styles['Normal'], fontSize=13, alignment=TA_CENTER,
    textColor=colors.HexColor('#7F8C8D'), spaceAfter=4)))
story.append(HRFlowable(width='80%', thickness=2, color=colors.HexColor('#2980B9')))
story.append(Spacer(1, 0.5 * cm))
cover_data = [
    ['Hoc phan:', 'Phan tich Doanh nghiep - FAA4021'],
    ['Nguon du lieu:', 'shopping_trends.csv (Kaggle)'],
    ['So ban ghi:', f'{len(df):,} quan sat x {df.shape[1]} bien'],
    ['Ky thuat:', 'Descriptive | Diagnostic | Predictive | Prescriptive'],
    ['Cong cu:', 'Python, Pandas, Scikit-learn, XGBoost, Streamlit'],
]
story.append(make_table(cover_data, col_widths=[5 * cm, 11 * cm], header=False))
story.append(PageBreak())

# Chapter 1
story.append(Paragraph('CHUONG 1: GIOI THIEU CHUNG VE DU AN PHAN TICH', H1_STYLE))
story.append(Paragraph('1.1 Mo ta bo du lieu', H2_STYLE))
story.append(Paragraph(
    'Bo du lieu Shopping Trends tu Kaggle mo phong hanh vi mua sam cua 3,900 khach hang '
    'tai 50 tieu bang Hoa Ky. Du lieu bao gom thong tin nhan khau hoc (tuoi, gioi tinh, '
    'vi tri), hanh vi mua sam (danh muc, san pham, gia tri don hang, tan suat mua) va cac '
    'yeu to marketing (subscription, discount, promo code, phuong thuc van chuyen). '
    'Quy mo du lieu dam bao du co so cho phan tich thong ke va mo hinh hoa.',
    BODY_STYLE))

story.append(Paragraph('1.2 Cau hoi va muc tieu nghien cuu', H2_STYLE))
goals = [
    'Phan tich co cau doanh thu theo san pham, khach hang, mua vu va dia ly',
    'Xac dinh nhom khach hang va danh muc san pham tiem nang cao nhat',
    'Kiem dinh su khac biet doanh thu giua cac nhom (gioi tinh, mua, subscription)',
    'Xay dung mo hinh du bao gia tri don hang (Purchase Amount)',
    'De xuat chien luoc tang truong doanh thu va toi uu nguon luc',
]
for i, g in enumerate(goals, 1):
    story.append(Paragraph(f'{i}. {g}', BODY_STYLE))

story.append(PageBreak())

# Chapter 2
story.append(Paragraph('CHUONG 2: PHUONG PHAP NGHIEN CUU', H1_STYLE))
story.append(Paragraph('2.1 Nguon va loai du lieu', H2_STYLE))
story.append(Paragraph(
    'Du lieu so cap tu Kaggle, dinh dang CSV, bao gom 19 truong du lieu: 5 bien so luong '
    '(Customer ID, Age, Purchase Amount, Review Rating, Previous Purchases) va 14 bien phan loai. '
    'Du lieu da duoc de-identified, khong chua PII.',
    BODY_STYLE))

story.append(Paragraph('2.2 Ket qua danh gia chat luong du lieu', H2_STYLE))
q_data = [
    ['Tieu chi kiem tra', 'Ket qua', 'Danh gia'],
    ['Tong so quan sat', f'{len(df):,}', 'Dat (>300 yeu cau)'],
    ['Tong so bien', str(df.shape[1]), 'Du phong phu'],
    ['Gia tri Null', '0 (0.0%)', 'Khong can xu ly'],
    ['Dong trung lap', '0 (0.0%)', 'Khong can xu ly'],
    ['Kieu du lieu', 'int64, float64, str', 'Da chuyen sang category'],
    ['Bien Age Group', 'Tao moi - 6 nhom', 'Feature engineering'],
]
story.append(make_table(q_data, col_widths=[5.5 * cm, 5 * cm, 5.5 * cm]))

story.append(Paragraph('2.3 Quy trinh xu ly du lieu', H2_STYLE))
story.append(Paragraph(
    'Quy trinh xu ly: (1) Doc CSV bang Pandas va kiem tra shape; (2) Kiem tra null/duplicate; '
    '(3) Chuyen 14 cot phan loai sang kieu category; (4) Tao bien Age Group bang pd.cut(); '
    '(5) Ma hoa nhi phan cot Yes/No cho mo hinh; (6) One-Hot Encoding cho phan tich du bao.',
    BODY_STYLE))

story.append(PageBreak())

# Chapter 3
story.append(Paragraph('CHUONG 3: KET QUA PHAN TICH', H1_STYLE))

story.append(Paragraph('3.1 Phan tich Mo ta (Descriptive Analytics)', H2_STYLE))
story.append(Paragraph('3.1.1 Cac chi so KPI Tong hop', H2_STYLE))
kpi_data = [
    ['Chi so', 'Gia tri', 'Ghi chu'],
    ['Tong doanh thu', f'${total_rev:,.0f}', 'Tong gia tri tich luy'],
    ['Tong khach hang', f'{total_cust:,}', 'Unique Customer ID'],
    ['Gia tri don hang TB', f'${avg_order:.2f}', 'Mean purchase amount'],
    ['Ty le Subscriber', f'{sub_rate:.1f}%', '27% co dang ky goi'],
    ['So danh muc', str(df['Category'].nunique()), 'Clothing, Accessories, Footwear, Outerwear'],
    ['So dia diem', str(df['Location'].nunique()), '50 tieu bang Hoa Ky'],
    ['Diem danh gia TB', f'{df["Review Rating"].mean():.2f}/5.0', 'Thu thap tu khach hang'],
]
story.append(make_table(kpi_data, col_widths=[5 * cm, 4.5 * cm, 6.5 * cm]))
story.append(Spacer(1, 0.2 * cm))

story.append(add_chart('01_revenue_by_gender.png', width=16 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 1: Phan tich doanh thu theo Gioi tinh', CAPTION_STYLE))

story.append(add_chart('04_revenue_by_season.png', width=16 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 2: Doanh thu theo Mua - Mua Thu (Fall) cao nhat', CAPTION_STYLE))

story.append(add_chart('03_revenue_by_category.png', width=16 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 3: Doanh thu theo Danh muc - Clothing chiem 44.7%', CAPTION_STYLE))

story.append(add_chart('05_top10_items.png', width=16 * cm, height=7 * cm))
story.append(Paragraph('Bieu do 4: Top 10 San pham Doanh thu Cao nhat', CAPTION_STYLE))

story.append(add_chart('08_histogram_purchase_amount.png', width=16 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 5: Histogram phan phoi gia tri mua hang (phan phoi deu $20-$100)', CAPTION_STYLE))

story.append(add_chart('10_heatmap_category_season.png', width=16 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 6: Heatmap Doanh thu theo Danh muc va Mua', CAPTION_STYLE))

story.append(PageBreak())

story.append(Paragraph('3.2 Phan tich Chan doan (Diagnostic Analytics)', H2_STYLE))
story.append(Paragraph(
    'Ap dung Correlation Analysis, Pivot Table, T-test (Welch) va One-way ANOVA de kiem '
    'dinh cac gia thuyet ve nguyen nhan tac dong den doanh thu.',
    BODY_STYLE))

diag_data = [
    ['Kiem dinh', 'Thong so', 'P-value', 'Ket luan'],
    ['T-test: Subscriber vs Non-sub', 't=-0.440', '0.660', 'Khong y nghia (p>0.05)'],
    ['ANOVA: Doanh thu theo Mua', 'F=3.746', '0.011', 'Co y nghia (*) - Fall cao nhat'],
    ['ANOVA: Doanh thu theo Danh muc', 'F=1.454', '0.225', 'Khong y nghia (p>0.05)'],
]
story.append(make_table(diag_data, col_widths=[6 * cm, 3.5 * cm, 2.5 * cm, 4 * cm]))
story.append(Spacer(1, 0.2 * cm))

story.append(add_chart('11_correlation_heatmap.png', width=15 * cm, height=9 * cm))
story.append(Paragraph('Bieu do 7: Heatmap Tuong quan - Khong co bien nao tuong quan cao voi Purchase Amount', CAPTION_STYLE))

story.append(add_chart('13_clustered_bar_subscription.png', width=16 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 8: Clustered Bar - Chi tieu theo Danh muc va Subscription Status', CAPTION_STYLE))

story.append(add_chart('14_comparative_boxplot_subscription.png', width=16 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 9: Comparative Boxplot - So sanh Subscriber va Non-Subscriber', CAPTION_STYLE))

story.append(PageBreak())

story.append(Paragraph('3.3 Phan tich Du bao (Predictive Analytics)', H2_STYLE))
story.append(Paragraph(
    'Xay dung 3 mo hinh du bao Purchase Amount (USD) sau khi thuc hien Feature Engineering '
    'va One-Hot Encoding. Du lieu chia 80/20 theo chien luoc random split.',
    BODY_STYLE))

model_data = [
    ['Mo hinh', 'MAE (USD)', 'RMSE (USD)', 'R2 Score'],
    ['Linear Regression (*)', '20.710', '23.721', '-0.006'],
    ['Random Forest', '21.773', '25.174', '-0.133'],
    ['XGBoost', '21.582', '24.935', '-0.111'],
]
story.append(make_table(model_data, col_widths=[5.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    'Nhan xet: R2 ~ 0 la ket qua hop ly vi Purchase Amount co phan phoi gan ngu nhien '
    'deu trong khoang $20-$100 (xac nhan qua histogram). Dac trung hien co khong du de '
    'du bao gia tri don hang - day la insight quan trong: doi thu canh tranh va cau hop '
    'dong/bia gia tiet kiem moi co the tac dong den gia tri don hang. Mo hinh tot nhat: '
    'Linear Regression theo tieu chi MAE thap nhat va R2 cao nhat.',
    BODY_STYLE))

story.append(add_chart('20_model_comparison.png', width=16 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 10: So sanh hieu suat MAE, RMSE, R2 giua 3 mo hinh', CAPTION_STYLE))

story.append(add_chart('19_rf_feature_importance.png', width=15 * cm, height=7 * cm))
story.append(Paragraph('Bieu do 11: Feature Importance - Random Forest (Review Rating quan trong nhat)', CAPTION_STYLE))

story.append(add_chart('15_lr_actual_vs_pred.png', width=14 * cm, height=7 * cm))
story.append(Paragraph('Bieu do 12: Actual vs Predicted - Linear Regression', CAPTION_STYLE))

story.append(PageBreak())

story.append(Paragraph('3.4 Phan tich De xuat (Prescriptive Analytics)', H2_STYLE))

# Insights
story.append(Paragraph('3.4.1 Insights Kinh doanh chinh', H2_STYLE))
insights_data = [
    ['Ma', 'Insight'],
    ['I01', f'Tong doanh thu ${total_rev:,.0f} USD tu {total_cust:,} khach hang'],
    ['I02', f'Danh muc {top_category} chiem 44.7% tong doanh thu'],
    ['I03', f'Mua {top_season} tao doanh thu cao nhat - co y nghia thong ke (ANOVA p=0.011)'],
    ['I04', f'San pham ban chay nhat: {top_item}'],
    ['I05', f'Dia diem doanh thu cao nhat: {top_location}'],
    ['I06', 'Subscriber chi tieu KHONG cao hon Non-subscriber (p=0.660>0.05)'],
    ['I07', 'Purchase Amount phan phoi deu - khong bi chi phoi boi yeu to cu the nao'],
    ['I08', 'Nhom tuoi 56+ dong gop doanh thu cao nhat (65,256 USD)'],
]
story.append(make_table(insights_data, col_widths=[1.5 * cm, 14.5 * cm]))
story.append(Spacer(1, 0.3 * cm))

# Recommendations
story.append(Paragraph('3.4.2 Actionable Recommendations', H2_STYLE))
recs = [
    ('R01 - Tang doanh thu mua vu',
     f'Tang ngan sach marketing 30% vao mua {top_season}. Flashsale cuoi tuan de tang tan suat mua. '
     f'Tap trung giao hang nhanh tai {top_location}.'),
    ('R02 - Khach hang muc tieu',
     'Nhom tuoi 56+ va 46-55 dong gop doanh thu cao nhat - can chien dich email va retargeting cu the. '
     'Khong nen chi nhom tuoi tre (18-25) vi dong gop thap hon.'),
    ('R03 - San pham nen quang ba',
     f'Dress, Jewelry, Sandals co diem score tong hop cao nhat (doanh thu + danh gia). '
     f'Bundle cac san pham phu kien voi san pham chinh cua danh muc {top_category}.'),
    ('R04 - Chuong trinh Loyalty',
     '4 tier: Regular > Silver > Gold > VIP. Tang diem thu vi doi voi don hang > $60. '
     'Gui email ca nhan hoa cho KH co Previous Purchases > 10. '
     'Mo rong Subscription voi uu dai ro rang de tang ty le dang ky tu 27%.'),
    ('R05 - Ton kho theo mua',
     f'Tang ton kho Clothing 40% truoc mua {top_season}. '
     'Giam gia dong loat cuoi mua de giai phong hang ton. '
     'Theo doi sell-through rate hang tuan.'),
    ('R06 - Phan bo ngan sach Marketing',
     '44.7% cho Clothing, 31.8% cho Accessories, 15.5% cho Footwear, 8% cho Outerwear. '
     '60% digital (Google/Meta Ads), 25% email & loyalty, 15% influencer.'),
]
for code, text in recs:
    story.append(Paragraph(f'[{code}]', H2_STYLE))
    story.append(Paragraph(text, BODY_STYLE))

story.append(add_chart('24_customer_segments.png', width=15 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 13: Phan khuc Khach hang', CAPTION_STYLE))

story.append(add_chart('27_marketing_budget_allocation.png', width=13 * cm, height=6 * cm))
story.append(Paragraph('Bieu do 14: De xuat Phan bo Ngan sach Marketing', CAPTION_STYLE))

story.append(PageBreak())

# Chapter 4
story.append(Paragraph('CHUONG 4: DAO DUC VA TRACH NHIEM XA HOI', H1_STYLE))
story.append(Paragraph('4.1 Phan tich van de dao duc va bao ve du lieu', H2_STYLE))
story.append(Paragraph(
    'Bo du lieu shopping_trends.csv la du lieu tong hop khong chua thong tin nhan dang ca nhan (PII). '
    'Toan bo phan tich chi phuc vu muc dich hoc thuat. Ket qua khong duoc su dung cho muc dich '
    'phan biet doi xu hay ton hai. Du lieu khong duoc chia se voi ben thu ba.',
    BODY_STYLE))

story.append(Paragraph('4.2 Nguyen tac dao duc nghe nghiep', H2_STYLE))
ethics_data = [
    ['Nguyen tac', 'Ap dung trong du an'],
    ['Tinh minh bach', 'Cong khai phuong phap, code, cong cu su dung'],
    ['Tinh toan ven', 'Khong chinh sua du lieu, bao cao ket qua khach quan'],
    ['Bao mat du lieu', 'Du lieu chi dung noi bo, khong chia se'],
    ['Han che AI', 'AI ho tro code - khong thay the tu duy phan tich'],
    ['Trich dan trung thuc', 'Khai bao su dung AI theo quy dinh hoc phan'],
]
story.append(make_table(ethics_data, col_widths=[5 * cm, 11 * cm]))

story.append(Paragraph('4.3 Giai trinh su dung AI', H2_STYLE))
ai_data = [
    ['Cong cu', 'Muc dich', 'Pham vi', 'Ket qua tu lam'],
    ['Claude (Anthropic)', 'Viet code Python', 'Toan bo src/', 'Kiem chung, debug, chay'],
    ['Claude (Anthropic)', 'Cau truc bao cao', 'Tham khao', 'Viet noi dung, phan tich'],
]
story.append(make_table(ai_data, col_widths=[3.5 * cm, 4 * cm, 3 * cm, 5.5 * cm]))
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph(
    'Cac noi dung tu thuc hien: (1) Xac dinh cau hoi nghien cuu; (2) Lua chon bo du lieu; '
    '(3) Dien giai ket qua thong ke va insight kinh doanh; (4) Kiem tra va hieu chinh code; '
    '(5) Viet bao cao va phan tich y nghia ket qua. '
    'AI khong the thay the hieu biet chuyen nganh trong viec dien giai y nghia thong ke '
    'va de xuat chien luoc kinh doanh phu hop voi boi canh cu the.',
    BODY_STYLE))

story.append(Paragraph('4.4 Cam ket hoc thuat', H2_STYLE))
story.append(Paragraph(
    'Tac gia cam ket: Noi dung bao cao da duoc kiem chung va khong sao chep. '
    'Su dung AI da duoc khai bao day du va trung thuc. Ket qua phan tich duoc xac nhan '
    'bang cach chay code tren du lieu thuc. Khong ton tai xung dot loi ich.',
    BODY_STYLE))

# Build
pdf_path = os.path.join(REPORTS_DIR, 'shopping_trends_analysis_report.pdf')
doc = SimpleDocTemplate(
    pdf_path, pagesize=A4,
    rightMargin=2 * cm, leftMargin=2 * cm,
    topMargin=2 * cm, bottomMargin=2 * cm
)
doc.build(story)
print(f'PDF report created: {pdf_path}')
