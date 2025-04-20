import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import os
from datetime import datetime

# ---------- 1. การโหลดข้อมูลและภาพรวม ----------
try:
    df = pd.read_csv("thai_road_accident_2019_2022.csv")
    print(f"โหลดชุดข้อมูลสำเร็จ มี {df.shape[0]} แถวและ {df.shape[1]} คอลัมน์")
except FileNotFoundError:
    # สำหรับการสาธิต สร้างข้อมูลตัวอย่างหากไม่พบไฟล์
    print("คำเตือน: ไม่พบไฟล์ชุดข้อมูล กำลังสร้างข้อมูลตัวอย่างสำหรับการสาธิต")
    # สร้างข้อมูลตัวอย่าง
    np.random.seed(42)
    provinces = ['กรุงเทพ', 'เชียงใหม่', 'ภูเก็ต', 'พัทยา', 'ขอนแก่น']
    vehicle_types = ['รถยนต์', 'รถจักรยานยนต์', 'รถบรรทุก', 'รถโดยสาร', 'รถตู้']
    causes = ['ขับเร็วเกินกำหนด', 'เมาแล้วขับ', 'สภาพอากาศ', 'รถเสีย', 'สภาพถนน']
    accident_types = ['การชน', 'รถพลิกคว่ำ', 'รถหลุดออกนอกถนน', 'ชนคนเดินเท้า']
    weather = ['แจ่มใส', 'ฝนตก', 'หมอก', 'พายุ']
    
    data = {
        'province_en': np.random.choice(provinces, 1000),
        'vehicle_type': np.random.choice(vehicle_types, 1000),
        'presumed_cause': np.random.choice(causes, 1000),
        'accident_type': np.random.choice(accident_types, 1000),
        'weather_condition': np.random.choice(weather, 1000),
        'number_of_injuries': np.random.randint(0, 10, 1000),
        'number_of_fatalities': np.random.randint(0, 5, 1000)
    }
    df = pd.DataFrame(data)

# ---------- 2. คำอธิบายชุดข้อมูลสำหรับแดชบอร์ด ----------
dataset_description = """
## ชุดข้อมูลอุบัติเหตุทางถนนในประเทศไทย (2562-2565)

**คำอธิบาย**: ชุดข้อมูลนี้มีข้อมูลเกี่ยวกับอุบัติเหตุทางถนนในประเทศไทยตั้งแต่ปี 2562 ถึง 2565 รวมถึงสถานที่เกิดเหตุ สาเหตุ ประเภทของยานพาหนะ และความรุนแรง

**คุณลักษณะ**:
- province_en: จังหวัดที่เกิดอุบัติเหตุ
- vehicle_type: ประเภทของยานพาหนะที่เกี่ยวข้อง (เช่น รถยนต์ รถจักรยานยนต์ รถบรรทุก)
- presumed_cause: สาเหตุที่สงสัยว่าเป็นต้นเหตุของอุบัติเหตุ (เช่น ขับเร็วเกินกำหนด เมาแล้วขับ)
- accident_type: ประเภทของอุบัติเหตุ (เช่น การชน รถพลิกคว่ำ)
- weather_condition: สภาพอากาศ ณ เวลาที่เกิดอุบัติเหตุ
- number_of_injuries: จำนวนผู้บาดเจ็บ
- number_of_fatalities: จำนวนผู้เสียชีวิต

**แหล่งที่มา**: กรมทางหลวง ประเทศไทย
"""

# ---------- ข้อมูลเมตาดาต้าของชุดข้อมูล (ใหม่) ----------
dataset_metadata = {
    'title': 'อุบัติเหตุทางถนนในประเทศไทย 2562-2565',
    'creator': 'ทวี วัชรพิชัย',
    'created_date': '25 มกราคม 2566',
    'last_updated': '25 มกราคม 2566',
    'source_url': 'https://www.kaggle.com/datasets/thaweewatboy/thailand-road-accident-2019-2022',
    'dataset_description': 'อุบัติเหตุทางถนนในประเทศไทยตั้งแต่ปี 2562 ถึง 2565 รวมถึงปัจจัยต่างๆ เช่น จังหวัด ประเภทยานพาหนะ สภาพอากาศ ฯลฯ',
    'license': 'CC0: สาธารณสมบัติ',
    'usability': '6.8',
    'file_size': '93.21 KB',
    'file_format': 'CSV',
    'records': 'บันทึกอุบัติเหตุทางถนนมากกว่า 6,000 รายการ'
}

# ---------- 3. Supervised Learning: Classification ----------
# a. Dataset selection is already done (Thai road accident dataset)

# b. Data Preparation for Classification
classification_preparation = """
### ขั้นตอนการเตรียมข้อมูลสำหรับการจำแนกประเภท:
1. **Feature Selection**: เลือก vehicle_type, presumed_cause, accident_type, และ weather_condition เป็นคุณลักษณะ
2. **Target Creation**: สร้างตัวแปรเป้าหมายแบบทวิภาค 'severity' โดยอิงจาก number_of_fatalities (รุนแรงถ้ามีผู้เสียชีวิต > 0)
3. **Handling Missing Values**: ลบแถวที่มีค่าขาดหายในคุณลักษณะที่เลือก
4. **Encoding Categorical Variables**: ใช้ LabelEncoder เพื่อแปลงตัวแปรเชิงหมวดหมู่เป็นรูปแบบตัวเลข
5. **Train-Test Split**: แบ่งข้อมูลเป็น 80% สำหรับการฝึกและ 20% สำหรับการทดสอบ โดยกำหนด random_state=42 เพื่อความสามารถในการทำซ้ำ
"""

df_cls = df.copy()
df_cls['severity'] = df_cls['number_of_fatalities'].apply(lambda x: 'severe' if x > 0 else 'non-severe')
features_cls = ['vehicle_type', 'presumed_cause', 'accident_type', 'weather_condition']
df_cls = df_cls[features_cls + ['severity']].dropna()

# แปลงคุณลักษณะเชิงหมวดหมู่ด้วย Label Encoder
le_dict = {}
for col in features_cls + ['severity']:
    le = LabelEncoder()
    df_cls[col] = le.fit_transform(df_cls[col])
    le_dict[col] = le

X = df_cls[features_cls]
y = df_cls['severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# c. วัตถุประสงค์ของการจำแนกประเภท
classification_objective = """
### วัตถุประสงค์ของแบบจำลองการจำแนกประเภท:
วัตถุประสงค์คือการจำแนกอุบัติเหตุทางถนนเป็น 'รุนแรง' (มีผู้เสียชีวิต) หรือ 'ไม่รุนแรง' (ไม่มีผู้เสียชีวิต) 
โดยอิงจากปัจจัยต่างๆ เช่น ประเภทของยานพาหนะ สาเหตุของอุบัติเหตุ ประเภทของอุบัติเหตุ และสภาพอากาศ การจำแนกนี้สามารถช่วย:

1. ระบุสถานการณ์อุบัติเหตุที่มีความเสี่ยงสูงซึ่งมีแนวโน้มที่จะส่งผลให้เกิดการเสียชีวิตมากขึ้น
2. พัฒนาการแทรกแซงด้านความปลอดภัยที่เฉพาะเจาะจงสำหรับการรวมกันของปัจจัยต่างๆ
3. จัดลำดับความสำคัญของทรัพยากรการตอบสนองฉุกเฉินสำหรับอุบัติเหตุที่มีการคาดการณ์ความรุนแรงสูงขึ้น
4. แนะนำการกำหนดนโยบายสำหรับข้อบังคับความปลอดภัยทางถนนตามปัจจัยที่เกี่ยวข้องกับอุบัติเหตุร้ายแรงมากที่สุด
"""

# d. สร้างแบบจำลองการจำแนกประเภท
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

model_results = {}
model_objects = {}
classification_reports = {}
confusion_matrices = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    model_objects[name] = model
    
    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    # Cross validation (5-fold)
    scores = cross_val_score(model, X, y, cv=5)
    cv_acc = scores.mean()
    
    # Store results
    model_results[name] = {
        'test_accuracy': round(test_acc, 3),
        'cv_accuracy': round(cv_acc, 3)
    }
    
    # Store classification report and confusion matrix
    classification_reports[name] = classification_report(y_test, y_pred, output_dict=True)
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

# e. Model Evaluation
model_names = list(model_results.keys())
test_accuracies = [model_results[model]['test_accuracy'] for model in model_names]
cv_accuracies = [model_results[model]['cv_accuracy'] for model in model_names]

fig_model_comparison = go.Figure(data=[
    go.Bar(name='Test Accuracy', x=model_names, y=test_accuracies),
    go.Bar(name='Cross-Validation Accuracy', x=model_names, y=cv_accuracies)
])
fig_model_comparison.update_layout(
    title='Model Accuracy Comparison (5-fold Cross-Validation)',
    xaxis_title='Classification Model',
    yaxis_title='Accuracy Score',
    barmode='group'
)

# ---------- 4. Unsupervised Learning: Clustering ----------
# a. Dataset selection is already done (Thai road accident dataset)

# b. Data Preparation for Clustering
clustering_preparation = """
### ขั้นตอนการเตรียมข้อมูลสำหรับการจัดกลุ่ม:
1. **Feature Selection**: เลือก province_en, vehicle_type, number_of_injuries, และ number_of_fatalities
2. **Handling Missing Values**: ลบแถวที่มีค่าขาดหายในคุณลักษณะที่เลือก
3. **Encoding Categorical Variables**: ใช้ LabelEncoder เพื่อแปลงจังหวัดและประเภทยานพาหนะเป็นรูปแบบตัวเลข
4. **Feature Scaling**: ใช้ StandardScaler เพื่อปรับมาตรฐานคุณลักษณะทั้งหมดให้มีขนาดคล้ายกัน
5. **Dimensionality**: เก็บคุณลักษณะทั้งหมดสำหรับการจัดกลุ่มเพื่อจับรูปแบบในทุกมิติ
"""

df_cluster = df[['province_en', 'vehicle_type', 'number_of_injuries', 'number_of_fatalities']].dropna()
df_cluster = df_cluster.copy()

# แปลงคุณลักษณะเชิงหมวดหมู่ด้วย Label Encoder
le_province = LabelEncoder()
le_vehicle = LabelEncoder()
df_cluster['province_encoded'] = le_province.fit_transform(df_cluster['province_en'])
df_cluster['vehicle_encoded'] = le_vehicle.fit_transform(df_cluster['vehicle_type'])

# ปรับขนาดคุณลักษณะตัวเลขเพื่อการจัดกลุ่มที่ดีขึ้น
scaler = StandardScaler()
features_for_clustering = ['province_encoded', 'vehicle_encoded', 'number_of_injuries', 'number_of_fatalities']
df_cluster_scaled = df_cluster.copy()
df_cluster_scaled[features_for_clustering] = scaler.fit_transform(df_cluster[features_for_clustering])

# หา K ที่เหมาะสมด้วยวิธี Elbow
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_cluster_scaled[features_for_clustering])
    wcss.append(kmeans.inertia_)

fig_elbow = px.line(x=list(K_range), y=wcss, markers=True)
fig_elbow.update_layout(
    title='วิธี Elbow สำหรับหา K ที่เหมาะสม',
    xaxis_title='จำนวนกลุ่ม (K)',
    yaxis_title='WCSS'
)

# c. สร้างอย่างน้อย 3 กลุ่ม
k = 3  # ขึ้นอยู่กับวิธี Elbow หรืออาจปรับได้
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(df_cluster_scaled[features_for_clustering])

# สร้างแผนภูมิการกระจาย 3D
fig_cluster = px.scatter_3d(
    df_cluster, 
    x='province_encoded', 
    y='number_of_injuries', 
    z='number_of_fatalities',
    color='cluster',
    hover_data=['province_en', 'vehicle_type'],
    labels={
        'province_encoded': 'จังหวัด',
        'number_of_injuries': 'จำนวนผู้บาดเจ็บ',
        'number_of_fatalities': 'จำนวนผู้เสียชีวิต'
    },
    title='การจัดกลุ่มของอุบัติเหตุทางถนน'
)

# d. การวิเคราะห์ลักษณะของกลุ่ม
cluster_analysis = df_cluster.groupby('cluster').agg({
    'province_en': lambda x: pd.Series.mode(x)[0],
    'vehicle_type': lambda x: pd.Series.mode(x)[0],
    'number_of_injuries': ['mean', 'median', 'max'],
    'number_of_fatalities': ['mean', 'median', 'max'],
}).reset_index()

# ทำให้ชื่อคอลัมน์อ่านง่ายขึ้น
cluster_analysis.columns = ['กลุ่ม', 'จังหวัดที่พบบ่อยที่สุด', 'ประเภทยานพาหนะที่พบบ่อยที่สุด', 
                           'ค่าเฉลี่ยผู้บาดเจ็บ', 'ค่ามัธยฐานผู้บาดเจ็บ', 'จำนวนผู้บาดเจ็บสูงสุด',
                           'ค่าเฉลี่ยผู้เสียชีวิต', 'ค่ามัธยฐานผู้เสียชีวิต', 'จำนวนผู้เสียชีวิตสูงสุด']

# ปัดเศษคอลัมน์ตัวเลข
for col in ['ค่าเฉลี่ยผู้บาดเจ็บ', 'ค่าเฉลี่ยผู้เสียชีวิต']:
    cluster_analysis[col] = cluster_analysis[col].round(2)

# เตรียมข้อความคำอธิบายกลุ่ม
cluster_descriptions = []
for _, row in cluster_analysis.iterrows():
    description = f"""
    **กลุ่ม {row['กลุ่ม']}**:
    - จังหวัดที่พบบ่อยที่สุด: {row['จังหวัดที่พบบ่อยที่สุด']}
    - ยานพาหนะที่พบบ่อยที่สุด: {row['ประเภทยานพาหนะที่พบบ่อยที่สุด']}
    - ค่าเฉลี่ยผู้บาดเจ็บ: {row['ค่าเฉลี่ยผู้บาดเจ็บ']}, สูงสุด: {row['จำนวนผู้บาดเจ็บสูงสุด']}
    - ค่าเฉลี่ยผู้เสียชีวิต: {row['ค่าเฉลี่ยผู้เสียชีวิต']}, สูงสุด: {row['จำนวนผู้เสียชีวิตสูงสุด']}
    """
    if row['ค่าเฉลี่ยผู้เสียชีวิต'] > cluster_analysis['ค่าเฉลี่ยผู้เสียชีวิต'].median():
        description += "- **กลุ่มที่มีการเสียชีวิตสูง**"
    elif row['ค่าเฉลี่ยผู้บาดเจ็บ'] > cluster_analysis['ค่าเฉลี่ยผู้บาดเจ็บ'].median():
        description += "- **กลุ่มที่มีการบาดเจ็บสูง แต่การเสียชีวิตต่ำ**"
    else:
        description += "- **กลุ่มที่มีความรุนแรงต่ำ**"
    
    cluster_descriptions.append(description)

# ---------- 5. Unsupervised Learning: Association Rule ----------
# a. Dataset selection is already done (Thai road accident dataset)

# b. การเตรียมข้อมูลสำหรับกฎความสัมพันธ์
association_preparation = """
### ขั้นตอนการเตรียมข้อมูลสำหรับกฎความสัมพันธ์:
1. **Feature Selection**: เลือก vehicle_type, presumed_cause, accident_type, และ weather_condition
2. **Handling Missing Values**: ลบแถวที่มีค่าขาดหายในคุณลักษณะที่เลือก
3. **One-Hot Encoding**: ใช้ pandas get_dummies เพื่อสร้างตัวบ่งชี้แบบไบนารีสำหรับตัวแปรเชิงหมวดหมู่ทั้งหมด
4. **Transaction Format**: แต่ละแถวแทนธุรกรรม (อุบัติเหตุ) ด้วยตัวบ่งชี้แบบไบนารีสำหรับแต่ละคุณลักษณะ
"""

df_assoc = df[['vehicle_type', 'presumed_cause', 'accident_type', 'weather_condition']].dropna()
df_assoc = pd.get_dummies(df_assoc)

# c. ใช้ Apriori ด้วยค่าสนับสนุนขั้นต่ำ 5% และความเชื่อมั่น 60%
# ใช้อัลกอริทึม Apriori
min_support = 0.05
min_confidence = 0.6

frequent_items = apriori(df_assoc, min_support=min_support, use_colnames=True)

# d. สร้างกฎความสัมพันธ์ที่แข็งแกร่ง 20 อันดับแรกโดยเรียงตามความเชื่อมั่น
if not frequent_items.empty:
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules.sort_values(by='confidence', ascending=False).head(20)
        
        # แปลง frozensets เป็นสตริงเพื่อการแสดงผล
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # จัดรูปแบบค่า
        rules_display = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
        rules_display.columns = ['เงื่อนไขก่อนหน้า', 'เงื่อนไขตามหลัง', 'ค่าสนับสนุน', 'ความเชื่อมั่น', 'Lift']
        rules_display['ค่าสนับสนุน'] = rules_display['ค่าสนับสนุน'].round(3)
        rules_display['ความเชื่อมั่น'] = rules_display['ความเชื่อมั่น'].round(3)
        rules_display['Lift'] = rules_display['Lift'].round(3)
    else:
        rules_display = pd.DataFrame(columns=['เงื่อนไขก่อนหน้า', 'เงื่อนไขตามหลัง', 'ค่าสนับสนุน', 'ความเชื่อมั่น', 'Lift'])
        print("ไม่พบกฎความสัมพันธ์ที่มีค่าขีดแบ่งตามที่กำหนด")
else:
    rules_display = pd.DataFrame(columns=['เงื่อนไขก่อนหน้า', 'เงื่อนไขตามหลัง', 'ค่าสนับสนุน', 'ความเชื่อมั่น', 'Lift'])
    print("ไม่พบชุดรายการที่เกิดบ่อยด้วยค่าสนับสนุนขั้นต่ำที่กำหนด")

# ---------- เค้าโครงของ Dash ----------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "การวิเคราะห์อุบัติเหตุทางถนนในประเทศไทย"

# แถบนำทาง
navbar = dbc.NavbarSimple(
    brand="แดชบอร์ดการวิเคราะห์อุบัติเหตุทางถนนในประเทศไทย",
    brand_style={"fontSize": "24px", "fontWeight": "bold"},
    color="primary",
    dark=True,
)

# ส่วนท้าย
footer = html.Footer(
    dbc.Container(
        [
            html.Hr(),
            html.P("การวิเคราะห์ข้อมูลอุบัติเหตุทางถนนในประเทศไทย (2562-2565)", className="text-center"),
            html.P("แดชบอร์ดสร้างด้วย Dash และ Plotly", className="text-center"),
        ]
    ),
    className="mt-5",
)

app.layout = html.Div([
    navbar,
    dbc.Container([
        html.Div([
            html.H1("การวิเคราะห์อุบัติเหตุทางถนนในประเทศไทย", className="text-center my-4"),
            html.P(
                "แดชบอร์ดนี้วิเคราะห์ข้อมูลอุบัติเหตุทางถนนในประเทศไทยโดยใช้เทคนิคการทำเหมืองข้อมูลหลากหลาย: "
                "การจำแนกประเภทเพื่อทำนายความรุนแรง, การจัดกลุ่มเพื่อค้นหารูปแบบ, และกฎความสัมพันธ์ "
                "เพื่อระบุความสัมพันธ์ระหว่างปัจจัยต่างๆ ของอุบัติเหตุ",
                className="lead text-center mb-4"
            ),
            # ข้อมูลชุดข้อมูล
            dbc.Card([
                dbc.CardHeader("ข้อมูลชุดข้อมูล"),
                dbc.CardBody(dcc.Markdown(dataset_description))
            ], className="mb-4"),
            
            dcc.Tabs(
                id="tabs", 
                value='classification', 
                className="mb-4",
                children=[
                    dcc.Tab(label='การจำแนกประเภท', value='classification'),
                    dcc.Tab(label='การจัดกลุ่ม', value='clustering'),
                    dcc.Tab(label='กฎความสัมพันธ์', value='association'),
                    dcc.Tab(label='เมตาดาต้าชุดข้อมูล', value='metadata')  # แท็บใหม่ที่เพิ่มเข้ามา
                ]
            ),
            html.Div(id='tabs-content')
        ], className="my-4"),
    ], fluid=True),
    footer
])

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'classification':
        return dbc.Container([
            # การเตรียมและวัตถุประสงค์ของการจำแนกประเภท
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("การเตรียมข้อมูล"),
                        dbc.CardBody(dcc.Markdown(classification_preparation))
                    ], className="mb-4")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("วัตถุประสงค์ของการจำแนกประเภท"),
                        dbc.CardBody(dcc.Markdown(classification_objective))
                    ], className="mb-4")
                ], md=6)
            ]),
            
            # ผลลัพธ์การจำแนกประเภท
            html.H3("ประสิทธิภาพของแบบจำลองการจำแนกประเภท", className="mb-4"),
            html.P(
                "เราจำแนกอุบัติเหตุทางถนนเป็น 'รุนแรง' (มีผู้เสียชีวิต) หรือ 'ไม่รุนแรง' (ไม่มีผู้เสียชีวิต) "
                "โดยอิงจากคุณลักษณะต่างๆ เช่น ประเภทของยานพาหนะ สาเหตุ ประเภทของอุบัติเหตุ และสภาพอากาศ",
                className="mb-3"
            ),
            dcc.Graph(figure=fig_model_comparison, className="mb-4"),
            
            html.H4("รายละเอียดประสิทธิภาพของแบบจำลอง (การตรวจสอบไขว้แบบ 5-fold)", className="mb-3"),
            dbc.Table.from_dataframe(
                pd.DataFrame({
                    'แบบจำลอง': model_names,
                    'ความแม่นยำของการทดสอบ': test_accuracies,
                    'ความแม่นยำของการตรวจสอบไขว้': cv_accuracies
                }),
                striped=True,
                bordered=True,
                hover=True,
                className="mb-4"
            ),
            
            html.H4("ความสำคัญของคุณลักษณะ (Random Forest)", className="mb-3"),
            dcc.Graph(
                figure=px.bar(
                    x=model_objects['Random Forest'].feature_importances_,
                    y=features_cls,
                    orientation='h',
                    labels={'x': 'ความสำคัญ', 'y': 'คุณลักษณะ'},
                    title='ความสำคัญของคุณลักษณะในการทำนายความรุนแรงของอุบัติเหตุ'
                )
            ),
            
            # เพิ่มรายงานการจำแนกสำหรับแบบจำลองที่ดีที่สุด
            html.H4("เมทริกซ์ประสิทธิภาพโดยละเอียด (แบบจำลองที่ดีที่สุด: Random Forest)", className="mb-4 mt-4"),
            html.Div([
                html.P(f"รายงานการจำแนก:", className="font-weight-bold"),
                html.Pre(
                    classification_report(
                        y_test, 
                        model_objects['Random Forest'].predict(X_test)
                    )
                )
            ], className="mb-4")
        ])

    elif tab == 'clustering':
        return dbc.Container([
            # การเตรียมข้อมูลสำหรับการจัดกลุ่ม
            dbc.Card([
                dbc.CardHeader("การเตรียมข้อมูลสำหรับการจัดกลุ่ม"),
                dbc.CardBody(dcc.Markdown(clustering_preparation))
            ], className="mb-4"),
            
            html.H3("การจัดกลุ่มอุบัติเหตุทางถนน", className="mb-4"),
            html.P(
                "ใช้ KMeans เพื่อระบุรูปแบบในข้อมูลอุบัติเหตุโดยอิงจากสถานที่ "
                "ประเภทยานพาหนะ ผู้บาดเจ็บ และผู้เสียชีวิต เราสร้าง 3 กลุ่มที่แสดงถึง "
                "รูปแบบความรุนแรงของอุบัติเหตุที่แตกต่างกัน",
                className="mb-3"
            ),
            dbc.Row([
                dbc.Col([
                    html.H4("การหาจำนวนกลุ่มที่เหมาะสม", className="mb-3"),
                    dcc.Graph(figure=fig_elbow)
                ], md=6),
                dbc.Col([
                    html.H4("ลักษณะของกลุ่ม", className="mb-3"),
                    dbc.Table.from_dataframe(
                        cluster_analysis[['กลุ่ม', 'จังหวัดที่พบบ่อยที่สุด', 'ประเภทยานพาหนะที่พบบ่อยที่สุด', 
                                          'ค่าเฉลี่ยผู้บาดเจ็บ', 'ค่าเฉลี่ยผู้เสียชีวิต']],
                        striped=True,
                        bordered=True,
                        hover=True
                    )
                ], md=6)
            ], className="mb-4"),
            
            # การวิเคราะห์กลุ่ม
            html.H4("การวิเคราะห์และการตีความกลุ่ม", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    dcc.Markdown("\n".join(cluster_descriptions))
                ])
            ], className="mb-4"),
            
            html.H4("การแสดงผลกลุ่มแบบ 3 มิติ", className="mb-3"),
            dcc.Graph(figure=fig_cluster)
        ])

    elif tab == 'association':
        if not rules_display.empty:
            return dbc.Container([
                # การเตรียมข้อมูลสำหรับกฎความสัมพันธ์
                dbc.Card([
                    dbc.CardHeader("การเตรียมข้อมูลสำหรับกฎความสัมพันธ์"),
                    dbc.CardBody(dcc.Markdown(association_preparation))
                ], className="mb-4"),
                
                html.H3("การวิเคราะห์กฎความสัมพันธ์", className="mb-4"),
                html.P(
                    "การค้นหาความสัมพันธ์ระหว่างปัจจัยของอุบัติเหตุโดยใช้อัลกอริทึม Apriori "
                    "กฎเหล่านี้แสดงให้เห็นว่าเงื่อนไขใดที่เกิดขึ้นร่วมกันบ่อยในอุบัติเหตุ",
                    className="mb-3"
                ),
                html.Div([
                    html.H4("กฎความสัมพันธ์ชั้นนำ", className="mb-3"),
                    html.P(
                        f"ค่าสนับสนุนขั้นต่ำ: {min_support*100}%, ความเชื่อมั่นขั้นต่ำ: {min_confidence*100}%, กฎที่พบ: {len(rules_display)}",
                        className="mb-3"
                    ),
                    dbc.Table.from_dataframe(
                        rules_display,
                        striped=True,
                        bordered=True,
                        hover=True
                    )
                ]),
                
                # การตีความกฎชั้นนำ
                html.H4("การตีความกฎชั้นนำ", className="mt-4 mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        html.P("กฎที่มีความเข้มแข็งที่สุดที่ค้นพบแสดงให้เห็น:"),
                        html.Ul([
                            html.Li(f"เมื่อเกิด {rules_display.iloc[0]['เงื่อนไขก่อนหน้า']} มีโอกาส {rules_display.iloc[0]['ความเชื่อมั่น']:.1%} ที่ {rules_display.iloc[0]['เงื่อนไขตามหลัง']} จะเกิดขึ้นด้วย"),
                            html.Li(f"การรวมกันของ {rules_display.iloc[1]['เงื่อนไขก่อนหน้า']} และ {rules_display.iloc[1]['เงื่อนไขตามหลัง']} ปรากฏใน {rules_display.iloc[1]['ค่าสนับสนุน']:.1%} ของอุบัติเหตุทั้งหมด"),
                            html.Li("กฎที่มีค่า Lift > 1 แสดงว่าการเกิดขึ้นของเงื่อนไขก่อนหน้าเพิ่มโอกาสของการเกิดเงื่อนไขตามหลัง")
                        ])
                    ])
                ], className="mb-4")
            ])
        else:
            return dbc.Container([
                html.H3("การวิเคราะห์กฎความสัมพันธ์", className="mb-4"),
                html.P(
                    f"ไม่พบกฎความสัมพันธ์ด้วยค่าขีดแบ่งปัจจุบัน (ค่าสนับสนุน={min_support*100}%, ความเชื่อมั่น={min_confidence*100}%) "
                    "ลองปรับค่าขีดแบ่งหรือตรวจสอบข้อมูลเพื่อหารูปแบบเพิ่มเติม",
                    className="alert alert-warning"
                )
            ])
    
    # เนื้อหาแท็บใหม่สำหรับเมตาดาต้าชุดข้อมูล
    elif tab == 'metadata':
        return dbc.Container([
            html.H3("เมตาดาต้าชุดข้อมูล", className="mb-4"),
            html.P(
                "ข้อมูลโดยละเอียดเกี่ยวกับแหล่งที่มาของชุดข้อมูลอุบัติเหตุทางถนนในประเทศไทย วันที่สร้าง และเมตาดาต้าอื่นๆ",
                className="mb-3"
            ),
            
            # การ์ดเมตาดาต้าชุดข้อมูล
            dbc.Card([
                dbc.CardHeader("ข้อมูลชุดข้อมูล"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("รายละเอียดชุดข้อมูล", className="mb-3"),
                            dbc.Table([
                                html.Tbody([
                                    html.Tr([
                                        html.Td("ชื่อ", className="font-weight-bold"),
                                        html.Td(dataset_metadata['title'])
                                    ]),
                                    html.Tr([
                                        html.Td("ผู้สร้าง", className="font-weight-bold"),
                                        html.Td(dataset_metadata['creator'])
                                    ]),
                                    html.Tr([
                                        html.Td("วันที่สร้าง", className="font-weight-bold"),
                                        html.Td(dataset_metadata['created_date'])
                                    ]),
                                    html.Tr([
                                        html.Td("ปรับปรุงล่าสุด", className="font-weight-bold"),
                                        html.Td(dataset_metadata['last_updated'])
                                    ]),
                                    html.Tr([
                                        html.Td("ขนาดไฟล์", className="font-weight-bold"),
                                        html.Td(dataset_metadata['file_size'])
                                    ]),
                                    html.Tr([
                                        html.Td("รูปแบบไฟล์", className="font-weight-bold"),
                                        html.Td(dataset_metadata['file_format'])
                                    ]),
                                    html.Tr([
                                        html.Td("จำนวนข้อมูล", className="font-weight-bold"),
                                        html.Td(dataset_metadata['records'])
                                    ]),
                                    html.Tr([
                                        html.Td("ลิขสิทธิ์", className="font-weight-bold"),
                                        html.Td(dataset_metadata['license'])
                                    ]),
                                    html.Tr([
                                        html.Td("คะแนนความใช้งานได้", className="font-weight-bold"),
                                        html.Td(dataset_metadata['usability'])
                                    ])
                                ])
                            ], bordered=True, hover=True)
                        ], md=6),
                        dbc.Col([
                            html.H5("คำอธิบายชุดข้อมูล", className="mb-3"),
                            html.P(dataset_metadata['dataset_description'], className="mb-4"),
                            
                            html.H5("ข้อมูลแหล่งที่มา", className="mb-3"),
                            html.P([
                                "ชุดข้อมูลนี้มีให้บริการสาธารณะบน Kaggle คุณสามารถเข้าถึงชุดข้อมูลต้นฉบับได้ที่: ",
                                html.A(
                                    "อุบัติเหตุทางถนนในประเทศไทย 2562-2565", 
                                    href=dataset_metadata['source_url'],
                                    target="_blank"
                                )
                            ])
                        ], md=6)
                    ])
                ])
            ], className="mb-4"),
            
            # สถิติชุดข้อมูล
            html.H4("สถิติชุดข้อมูล", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("สถิติสำคัญ", className="mb-3"),
                            html.Ul([
                                html.Li(f"จำนวนข้อมูลทั้งหมด: {df.shape[0]}"),
                                html.Li(f"คุณลักษณะ: {df.shape[1]}"),
                                html.Li(f"จังหวัดที่ไม่ซ้ำกัน: {df['province_en'].nunique() if 'province_en' in df.columns else 'ไม่มีข้อมูล'}"),
                                html.Li(f"ประเภทยานพาหนะที่ไม่ซ้ำกัน: {df['vehicle_type'].nunique() if 'vehicle_type' in df.columns else 'ไม่มีข้อมูล'}"),
                                html.Li(f"ผู้บาดเจ็บที่บันทึกทั้งหมด: {df['number_of_injuries'].sum() if 'number_of_injuries' in df.columns else 'ไม่มีข้อมูล'}"),
                                html.Li(f"ผู้เสียชีวิตที่บันทึกทั้งหมด: {df['number_of_fatalities'].sum() if 'number_of_fatalities' in df.columns else 'ไม่มีข้อมูล'}")
                            ])
                        ], md=6),
                        dbc.Col([
                            html.H5("ตัวอย่างชุดข้อมูล", className="mb-3"),
                            dbc.Table.from_dataframe(
                                df.head(5),
                                striped=True,
                                bordered=True,
                                hover=True,
                                responsive=True
                            )
                        ], md=6)
                    ])
                ])
            ])
        ])

if __name__ == '__main__':
    app.run(debug=True) 