import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Setup output dir
os.makedirs('outputs', exist_ok=True)

# 1. Load data
orders = pd.read_csv('data/clean/orders.csv')
customers = pd.read_csv('data/clean/customers.csv')
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
completed_orders = orders[orders['status'] == 'completed'].copy()

# 2. RFM Table
snapshot = pd.to_datetime('2025-10-01 00:00:00')
rfm = completed_orders.groupby('customer_id').agg(
    recency=('order_ts', lambda x: (snapshot - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum')
).reset_index()

# 3. K-Means Clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

print("K-Means Tuning:")
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(rfm_scaled)
    print(f"K={k}: Inertia={kmeans.inertia_:.1f}, Silhouette={silhouette_score(rfm_scaled, kmeans.labels_):.3f}")

# Choosing K=4 
final_k = 4
kmeans = KMeans(n_clusters=final_k, n_init=10, random_state=42)
rfm['segment_id'] = kmeans.fit_predict(rfm_scaled)

segment_stats = rfm.groupby('segment_id')[['recency', 'frequency', 'monetary']].mean()
premium_segment_id = segment_stats['monetary'].idxmax()
rfm['is_premium'] = rfm['segment_id'] == premium_segment_id

# Assign business names
names = {premium_segment_id: 'Premium'}
other_names = ['Frequent', 'Rare', 'At-Risk']
idx = 0
for sid in rfm['segment_id'].unique():
    if sid != premium_segment_id:
        names[sid] = other_names[idx]
        idx += 1
rfm['segment_name'] = rfm['segment_id'].map(names)

pca = PCA(n_components=2)
pca.fit(rfm_scaled)
print(f"PCA Explained Variance: {pca.explained_variance_ratio_}")
rfm.to_csv('outputs/customer_segments.csv', index=False)

# 4. Lapse Prediction Setup
cutoff = pd.to_datetime('2025-09-01')
before_cutoff = orders[orders['order_ts'] < cutoff].copy()
cohort_customers = before_cutoff[before_cutoff['status'] == 'completed']['customer_id'].unique()

sep_orders = orders[(orders['order_ts'] >= cutoff) & (orders['order_ts'] < snapshot) & (orders['status'] == 'completed')]
sep_purchasers = sep_orders['customer_id'].unique()

# Feature Engineering
features_list = []
for cid in cohort_customers:
    cust_orders = before_cutoff[before_cutoff['customer_id'] == cid]
    cust_comp = cust_orders[cust_orders['status'] == 'completed']
    
    lapsed = 1 if cid not in sep_purchasers else 0
    recency = (cutoff - cust_comp['order_ts'].max()).days if not cust_comp.empty else pd.NA
    freq = cust_comp['order_id'].nunique()
    monetary = cust_comp['order_amount'].sum()
    avg_disc = cust_comp['discount_pct'].mean() if not cust_comp.empty else 0
    
    final_orders = cust_orders[cust_orders['status'].isin(['completed', 'cancelled'])]
    cancel_rate = (final_orders['status'] == 'cancelled').mean() if not final_orders.empty else 0
    avg_del = cust_comp['delivery_minutes'].mean() if not cust_comp.empty else 0
    
    cust_info = customers[customers['customer_id'] == cid].iloc[0]
    features_list.append({
        'customer_id': cid, 'recency': recency, 'frequency': freq, 'monetary': monetary,
        'avg_discount': avg_disc, 'cancel_rate': cancel_rate, 'avg_delivery': avg_del,
        'city': cust_info['city'], 'age': cust_info['age'], 'gender': cust_info['gender'],
        'tenure': (cutoff - cust_info['signup_date']).days, 'lapsed': lapsed
    })

df_features = pd.DataFrame(features_list).fillna(0)
X = df_features.drop(columns=['customer_id', 'lapsed'])
y = df_features['lapsed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing Pipeline
num_cols = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancel_rate', 'avg_delivery', 'age', 'tenure']
cat_cols = ['city', 'gender']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# Random Forest Tuning
rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestClassifier(random_state=42))
])
param_grid = {
    'rf__max_depth': [3, 5, 8],
    'rf__min_samples_leaf': [5, 10, 20],
    'rf__n_estimators': [100]
}
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

# Persist Best Model
best_model = grid.best_estimator_
joblib.dump(best_model, 'outputs/lapse_model.joblib')

# Retention List (Premium test customers)
test_cids = df_features.iloc[X_test.index]['customer_id']
premium_cids = rfm[rfm['is_premium']]['customer_id']
test_premium_mask = test_cids.isin(premium_cids)

premium_test_X = X_test[test_premium_mask]
if not premium_test_X.empty:
    probs = best_model.predict_proba(premium_test_X)[:, 1]
    retention_df = pd.DataFrame({
        'customer_id': test_cids[test_premium_mask],
        'lapse_probability': probs,
        'actually_lapsed': y_test[test_premium_mask]
    }).sort_values('lapse_probability', ascending=False)
    retention_df.head(10).to_csv('outputs/retention_list.csv', index=False)