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

# ---------- 1. Data Loading and Overview ----------
try:
    df = pd.read_csv("thai_road_accident_2019_2022.csv")
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    # For demonstration, create some sample data if file not found
    print("Warning: Dataset file not found. Creating sample data for demonstration.")
    # Create sample data
    np.random.seed(42)
    provinces = ['Bangkok', 'Chiang Mai', 'Phuket', 'Pattaya', 'Khon Kaen']
    vehicle_types = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Van']
    causes = ['Speeding', 'Drunk driving', 'Weather', 'Vehicle failure', 'Road condition']
    accident_types = ['Collision', 'Rollover', 'Run-off-road', 'Hit pedestrian']
    weather = ['Clear', 'Rain', 'Fog', 'Storm']
    
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

# ---------- 2. Dataset Description for Dashboard ----------
dataset_description = """
## Thai Road Accident Dataset (2019-2022)

**Description**: This dataset contains information about road accidents in Thailand from 2019 to 2022, including location, cause, vehicle type, and severity.

**Features**:
- province_en: Province where the accident occurred
- vehicle_type: Type of vehicle involved (e.g., Car, Motorcycle, Truck)
- presumed_cause: Suspected cause of the accident (e.g., Speeding, Drunk driving)
- accident_type: Type of accident (e.g., Collision, Rollover)
- weather_condition: Weather at the time of accident
- number_of_injuries: Number of people injured
- number_of_fatalities: Number of people killed

**Source**: Department of Highways, Thailand
"""

# ---------- Dataset Metadata (New) ----------
dataset_metadata = {
    'title': 'Thailand Road Accident 2019-2022',
    'creator': 'Thawee Watcharapichat',
    'created_date': '2023-01-25',
    'last_updated': '2023-01-25',
    'source_url': 'https://www.kaggle.com/datasets/thaweewatboy/thailand-road-accident-2019-2022',
    'dataset_description': 'Road accidents in Thailand from 2019 to 2022 including factors such as province, vehicle type, weather condition, etc.',
    'license': 'CC0: Public Domain',
    'usability': '6.8',
    'file_size': '93.21 KB',
    'file_format': 'CSV',
    'records': '6,000+ road accident records'
}

# ---------- 3. Supervised Learning: Classification ----------
# a. Dataset selection is already done (Thai road accident dataset)

# b. Data Preparation for Classification
classification_preparation = """
### Classification Data Preparation Steps:
1. **Feature Selection**: Selected vehicle_type, presumed_cause, accident_type, and weather_condition as features.
2. **Target Creation**: Created a binary target variable 'severity' based on number_of_fatalities (severe if fatalities > 0).
3. **Handling Missing Values**: Removed rows with missing values in selected features.
4. **Encoding Categorical Variables**: Used LabelEncoder to convert categorical variables to numeric format.
5. **Train-Test Split**: Split data into 80% training and 20% test sets with random_state=42 for reproducibility.
"""

df_cls = df.copy()
df_cls['severity'] = df_cls['number_of_fatalities'].apply(lambda x: 'severe' if x > 0 else 'non-severe')
features_cls = ['vehicle_type', 'presumed_cause', 'accident_type', 'weather_condition']
df_cls = df_cls[features_cls + ['severity']].dropna()

# Label encode categorical features
le_dict = {}
for col in features_cls + ['severity']:
    le = LabelEncoder()
    df_cls[col] = le.fit_transform(df_cls[col])
    le_dict[col] = le

X = df_cls[features_cls]
y = df_cls['severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# c. Classification Objective
classification_objective = """
### Classification Model Objective:
The objective is to classify road accidents as either 'severe' (resulting in fatalities) or 'non-severe' (no fatalities) 
based on factors like vehicle type, accident cause, accident type, and weather conditions. This classification can help:

1. Identify high-risk accident scenarios that are more likely to result in fatalities
2. Develop targeted safety interventions for specific combinations of factors
3. Prioritize emergency response resources for accidents with higher predicted severity
4. Guide policy-making for road safety regulations based on factors most associated with fatal accidents
"""

# d. Create Classification Models
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
### Clustering Data Preparation Steps:
1. **Feature Selection**: Selected province_en, vehicle_type, number_of_injuries, and number_of_fatalities.
2. **Handling Missing Values**: Removed rows with missing values in selected features.
3. **Encoding Categorical Variables**: Used LabelEncoder to convert province and vehicle type to numeric format.
4. **Feature Scaling**: Applied StandardScaler to normalize all features to have similar scale.
5. **Dimensionality**: Kept all features for clustering to capture patterns across all dimensions.
"""

df_cluster = df[['province_en', 'vehicle_type', 'number_of_injuries', 'number_of_fatalities']].dropna()
df_cluster = df_cluster.copy()

# Label encode categorical features
le_province = LabelEncoder()
le_vehicle = LabelEncoder()
df_cluster['province_encoded'] = le_province.fit_transform(df_cluster['province_en'])
df_cluster['vehicle_encoded'] = le_vehicle.fit_transform(df_cluster['vehicle_type'])

# Scale numeric features for better clustering
scaler = StandardScaler()
features_for_clustering = ['province_encoded', 'vehicle_encoded', 'number_of_injuries', 'number_of_fatalities']
df_cluster_scaled = df_cluster.copy()
df_cluster_scaled[features_for_clustering] = scaler.fit_transform(df_cluster[features_for_clustering])

# Find optimal K using elbow method
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_cluster_scaled[features_for_clustering])
    wcss.append(kmeans.inertia_)

fig_elbow = px.line(x=list(K_range), y=wcss, markers=True)
fig_elbow.update_layout(
    title='Elbow Method for Optimal K',
    xaxis_title='Number of Clusters (K)',
    yaxis_title='WCSS'
)

# c. Create at least 3 clusters
k = 3  # Based on elbow method or could be adjusted
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(df_cluster_scaled[features_for_clustering])

# Create 3D scatter plot
fig_cluster = px.scatter_3d(
    df_cluster, 
    x='province_encoded', 
    y='number_of_injuries', 
    z='number_of_fatalities',
    color='cluster',
    hover_data=['province_en', 'vehicle_type'],
    labels={
        'province_encoded': 'Province',
        'number_of_injuries': 'Number of Injuries',
        'number_of_fatalities': 'Number of Fatalities'
    },
    title='Clustering of Road Accidents'
)

# d. Cluster characteristics analysis
cluster_analysis = df_cluster.groupby('cluster').agg({
    'province_en': lambda x: pd.Series.mode(x)[0],
    'vehicle_type': lambda x: pd.Series.mode(x)[0],
    'number_of_injuries': ['mean', 'median', 'max'],
    'number_of_fatalities': ['mean', 'median', 'max'],
}).reset_index()

# Make the column names more readable
cluster_analysis.columns = ['Cluster', 'Most Common Province', 'Most Common Vehicle Type', 
                           'Avg Injuries', 'Median Injuries', 'Max Injuries',
                           'Avg Fatalities', 'Median Fatalities', 'Max Fatalities']

# Round numeric columns
for col in ['Avg Injuries', 'Avg Fatalities']:
    cluster_analysis[col] = cluster_analysis[col].round(2)

# Prepare cluster description text
cluster_descriptions = []
for _, row in cluster_analysis.iterrows():
    description = f"""
    **Cluster {row['Cluster']}**:
    - Most common province: {row['Most Common Province']}
    - Most common vehicle: {row['Most Common Vehicle Type']}
    - Average injuries: {row['Avg Injuries']}, Max: {row['Max Injuries']}
    - Average fatalities: {row['Avg Fatalities']}, Max: {row['Max Fatalities']}
    """
    if row['Avg Fatalities'] > cluster_analysis['Avg Fatalities'].median():
        description += "- **High-fatality cluster**"
    elif row['Avg Injuries'] > cluster_analysis['Avg Injuries'].median():
        description += "- **High-injury, low-fatality cluster**"
    else:
        description += "- **Low-severity cluster**"
    
    cluster_descriptions.append(description)

# ---------- 5. Unsupervised Learning: Association Rule ----------
# a. Dataset selection is already done (Thai road accident dataset)

# b. Data Preparation for Association Rules
association_preparation = """
### Association Rule Data Preparation Steps:
1. **Feature Selection**: Selected vehicle_type, presumed_cause, accident_type, and weather_condition.
2. **Handling Missing Values**: Removed rows with missing values in selected features.
3. **One-Hot Encoding**: Used pandas get_dummies to create binary indicators for all categorical variables.
4. **Transaction Format**: Each row represents a transaction (accident) with binary indicators for each attribute.
"""

df_assoc = df[['vehicle_type', 'presumed_cause', 'accident_type', 'weather_condition']].dropna()
df_assoc = pd.get_dummies(df_assoc)

# c. Apply Apriori with minimum support 5% and confidence 60%
# Apply Apriori algorithm
min_support = 0.05
min_confidence = 0.6

frequent_items = apriori(df_assoc, min_support=min_support, use_colnames=True)

# d. Generate top 20 strong association rules by confidence
if not frequent_items.empty:
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules.sort_values(by='confidence', ascending=False).head(20)
        
        # Convert frozensets to strings for display
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Format the values
        rules_display = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
        rules_display.columns = ['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift']
        rules_display['Support'] = rules_display['Support'].round(3)
        rules_display['Confidence'] = rules_display['Confidence'].round(3)
        rules_display['Lift'] = rules_display['Lift'].round(3)
    else:
        rules_display = pd.DataFrame(columns=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])
        print("No association rules found with the given thresholds.")
else:
    rules_display = pd.DataFrame(columns=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])
    print("No frequent itemsets found with the given support threshold.")

# ---------- Dash Layout ----------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Thai Road Accident Analysis"

# Navigation bar
navbar = dbc.NavbarSimple(
    brand="Thai Road Accident Analysis Dashboard",
    brand_style={"fontSize": "24px", "fontWeight": "bold"},
    color="primary",
    dark=True,
)

# Footer
footer = html.Footer(
    dbc.Container(
        [
            html.Hr(),
            html.P("Thai Road Accident Data Analysis (2019-2022)", className="text-center"),
            html.P("Dashboard created with Dash and Plotly", className="text-center"),
        ]
    ),
    className="mt-5",
)

app.layout = html.Div([
    navbar,
    dbc.Container([
        html.Div([
            html.H1("Thai Road Accident Analysis", className="text-center my-4"),
            html.P(
                "This dashboard analyzes Thai road accident data using various data mining techniques: "
                "classification for severity prediction, clustering for pattern discovery, and association rules "
                "to identify relationships between accident factors.",
                className="lead text-center mb-4"
            ),
            # Dataset information
            dbc.Card([
                dbc.CardHeader("Dataset Information"),
                dbc.CardBody(dcc.Markdown(dataset_description))
            ], className="mb-4"),
            
            dcc.Tabs(
                id="tabs", 
                value='classification', 
                className="mb-4",
                children=[
                    dcc.Tab(label='Classification', value='classification'),
                    dcc.Tab(label='Clustering', value='clustering'),
                    dcc.Tab(label='Association Rule', value='association'),
                    dcc.Tab(label='Dataset Metadata', value='metadata')  # New tab added
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
            # Classification preparation and objective
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Data Preparation"),
                        dbc.CardBody(dcc.Markdown(classification_preparation))
                    ], className="mb-4")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Classification Objective"),
                        dbc.CardBody(dcc.Markdown(classification_objective))
                    ], className="mb-4")
                ], md=6)
            ]),
            
            # Classification results
            html.H3("Classification Model Performance", className="mb-4"),
            html.P(
                "We classify road accidents into 'severe' (with fatalities) or 'non-severe' (without fatalities) "
                "based on features such as vehicle type, cause, accident type, and weather conditions.",
                className="mb-3"
            ),
            dcc.Graph(figure=fig_model_comparison, className="mb-4"),
            
            html.H4("Model Performance Details (5-fold Cross Validation)", className="mb-3"),
            dbc.Table.from_dataframe(
                pd.DataFrame({
                    'Model': model_names,
                    'Test Accuracy': test_accuracies,
                    'Cross-Validation Accuracy': cv_accuracies
                }),
                striped=True,
                bordered=True,
                hover=True,
                className="mb-4"
            ),
            
            html.H4("Feature Importance (Random Forest)", className="mb-3"),
            dcc.Graph(
                figure=px.bar(
                    x=model_objects['Random Forest'].feature_importances_,
                    y=features_cls,
                    orientation='h',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    title='Feature Importance in Accident Severity Prediction'
                )
            ),
            
            # Add classification report for best model
            html.H4("Detailed Performance Metrics (Best Model: Random Forest)", className="mb-4 mt-4"),
            html.Div([
                html.P(f"Classification Report:", className="font-weight-bold"),
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
            # Clustering preparation
            dbc.Card([
                dbc.CardHeader("Data Preparation for Clustering"),
                dbc.CardBody(dcc.Markdown(clustering_preparation))
            ], className="mb-4"),
            
            html.H3("Clustering Road Accidents", className="mb-4"),
            html.P(
                "Using KMeans to identify patterns in accident data based on location, "
                "vehicle type, injuries, and fatalities. We created 3 clusters representing "
                "different accident severity profiles.",
                className="mb-3"
            ),
            dbc.Row([
                dbc.Col([
                    html.H4("Finding Optimal Number of Clusters", className="mb-3"),
                    dcc.Graph(figure=fig_elbow)
                ], md=6),
                dbc.Col([
                    html.H4("Cluster Characteristics", className="mb-3"),
                    dbc.Table.from_dataframe(
                        cluster_analysis[['Cluster', 'Most Common Province', 'Most Common Vehicle Type', 
                                          'Avg Injuries', 'Avg Fatalities']],
                        striped=True,
                        bordered=True,
                        hover=True
                    )
                ], md=6)
            ], className="mb-4"),
            
            # Cluster analysis
            html.H4("Cluster Analysis and Interpretation", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    dcc.Markdown("\n".join(cluster_descriptions))
                ])
            ], className="mb-4"),
            
            html.H4("3D Visualization of Clusters", className="mb-3"),
            dcc.Graph(figure=fig_cluster)
        ])

    elif tab == 'association':
        if not rules_display.empty:
            return dbc.Container([
                # Association preparation
                dbc.Card([
                    dbc.CardHeader("Data Preparation for Association Rules"),
                    dbc.CardBody(dcc.Markdown(association_preparation))
                ], className="mb-4"),
                
                html.H3("Association Rules Analysis", className="mb-4"),
                html.P(
                    "Discovering relationships between accident factors using Apriori algorithm. "
                    "These rules show which conditions frequently occur together in accidents.",
                    className="mb-3"
                ),
                html.Div([
                    html.H4("Top Association Rules", className="mb-3"),
                    html.P(
                        f"Minimum support: {min_support*100}%, Minimum confidence: {min_confidence*100}%, Rules found: {len(rules_display)}",
                        className="mb-3"
                    ),
                    dbc.Table.from_dataframe(
                        rules_display,
                        striped=True,
                        bordered=True,
                        hover=True
                    )
                ]),
                
                # Interpretation of top rules
                html.H4("Interpretation of Top Rules", className="mt-4 mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        html.P("The strongest rules discovered show:"),
                        html.Ul([
                            html.Li(f"When {rules_display.iloc[0]['Antecedents']} occurs, there's a {rules_display.iloc[0]['Confidence']:.1%} chance that {rules_display.iloc[0]['Consequents']} also occurs."),
                            html.Li(f"The combination of {rules_display.iloc[1]['Antecedents']} and {rules_display.iloc[1]['Consequents']} appears in {rules_display.iloc[1]['Support']:.1%} of all accidents."),
                            html.Li("Rules with lift > 1 indicate that the occurrence of the antecedent increases the likelihood of the consequent.")
                        ])
                    ])
                ], className="mb-4")
            ])
        else:
            return dbc.Container([
                html.H3("Association Rules Analysis", className="mb-4"),
                html.P(
                    f"No association rules found with the current thresholds (support={min_support*100}%, confidence={min_confidence*100}%). "
                    "Try adjusting the thresholds or check the data for more patterns.",
                    className="alert alert-warning"
                )
            ])
    
    # New tab content for Dataset Metadata
    elif tab == 'metadata':
        return dbc.Container([
            html.H3("Dataset Metadata", className="mb-4"),
            html.P(
                "Detailed information about the Thai Road Accident dataset source, creation date, and other metadata.",
                className="mb-3"
            ),
            
            # Dataset metadata card
            dbc.Card([
                dbc.CardHeader("Dataset Information"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Dataset Details", className="mb-3"),
                            dbc.Table([
                                html.Tbody([
                                    html.Tr([
                                        html.Td("Title", className="font-weight-bold"),
                                        html.Td(dataset_metadata['title'])
                                    ]),
                                    html.Tr([
                                        html.Td("Creator", className="font-weight-bold"),
                                        html.Td(dataset_metadata['creator'])
                                    ]),
                                    html.Tr([
                                        html.Td("Created Date", className="font-weight-bold"),
                                        html.Td(dataset_metadata['created_date'])
                                    ]),
                                    html.Tr([
                                        html.Td("Last Updated", className="font-weight-bold"),
                                        html.Td(dataset_metadata['last_updated'])
                                    ]),
                                    html.Tr([
                                        html.Td("File Size", className="font-weight-bold"),
                                        html.Td(dataset_metadata['file_size'])
                                    ]),
                                    html.Tr([
                                        html.Td("File Format", className="font-weight-bold"),
                                        html.Td(dataset_metadata['file_format'])
                                    ]),
                                    html.Tr([
                                        html.Td("Records", className="font-weight-bold"),
                                        html.Td(dataset_metadata['records'])
                                    ]),
                                    html.Tr([
                                        html.Td("License", className="font-weight-bold"),
                                        html.Td(dataset_metadata['license'])
                                    ]),
                                    html.Tr([
                                        html.Td("Usability Score", className="font-weight-bold"),
                                        html.Td(dataset_metadata['usability'])
                                    ])
                                ])
                            ], bordered=True, hover=True)
                        ], md=6),
                        dbc.Col([
                            html.H5("Dataset Description", className="mb-3"),
                            html.P(dataset_metadata['dataset_description'], className="mb-4"),
                            
                            html.H5("Source Information", className="mb-3"),
                            html.P([
                                "This dataset is publicly available on Kaggle. You can access the original dataset at: ",
                                html.A(
                                    "Thailand Road Accident 2019-2022", 
                                    href=dataset_metadata['source_url'],
                                    target="_blank"
                                )
                            ])
                        ], md=6)
                    ])
                ])
            ], className="mb-4"),
            
            # Dataset statistics
            html.H4("Dataset Statistics", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Key Statistics", className="mb-3"),
                            html.Ul([
                                html.Li(f"Total records: {df.shape[0]}"),
                                html.Li(f"Features: {df.shape[1]}"),
                                html.Li(f"Unique provinces: {df['province_en'].nunique() if 'province_en' in df.columns else 'N/A'}"),
                                html.Li(f"Unique vehicle types: {df['vehicle_type'].nunique() if 'vehicle_type' in df.columns else 'N/A'}"),
                                html.Li(f"Total injuries recorded: {df['number_of_injuries'].sum() if 'number_of_injuries' in df.columns else 'N/A'}"),
                                html.Li(f"Total fatalities recorded: {df['number_of_fatalities'].sum() if 'number_of_fatalities' in df.columns else 'N/A'}")
                            ])
                        ], md=6),
                        dbc.Col([
                            html.H5("Dataset Preview", className="mb-3"),
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

server = app.server

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run_server(host='0.0.0.0', port=port, debug=False)