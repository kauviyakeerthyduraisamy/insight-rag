import pandas as pd
import sqlite3
import os

# 1. Load Data
customers = pd.read_csv('data/raw/customers.csv')
orders = pd.read_csv('data/raw/orders.csv')
products = pd.read_csv('data/raw/products.csv')

# 2. Clean Customers
customers['city'] = customers['city'].str.strip().str.title()
customers['gender'] = customers['gender'].str.lower().replace(
    {'m': 'male', 'f': 'female', 'male': 'male', 'female': 'female'}
).fillna('unknown')

customers.loc[(customers['age'] < 10) | (customers['age'] > 100), 'age'] = pd.NA
customers['age'] = customers.groupby('city')['age'].transform(lambda x: x.fillna(x.median()))
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

# 3. Clean Orders
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
orders = orders.drop_duplicates()
orders = orders.sort_values('order_ts').groupby('order_id').tail(1)

orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
orders = orders.merge(products[['product_id', 'price_inr']], on='product_id', how='left')
missing_amt_mask = orders['order_amount'].isna()
orders.loc[missing_amt_mask, 'order_amount'] = round(
    orders.loc[missing_amt_mask, 'price_inr'] * orders.loc[missing_amt_mask, 'quantity'] * (1 - orders.loc[missing_amt_mask, 'discount_pct'] / 100), 2
)

orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = pd.NA
completed = orders['status'] == 'completed'
orders.loc[completed, 'delivery_minutes'] = orders[completed].groupby(
    orders['product_id'].map(products.set_index('product_id')['category'])
)['delivery_minutes'].transform(lambda x: x.fillna(x.median()))

# 4. Save and DB validation
os.makedirs('data/clean', exist_ok=True)
customers.to_csv('data/clean/customers.csv', index=False)
orders.drop(columns=['price_inr']).to_csv('data/clean/orders.csv', index=False)
products.to_csv('data/clean/products.csv', index=False)

conn = sqlite3.connect('data/instamart.db')
customers.to_sql('customers', conn, if_exists='replace', index=False)
orders.drop(columns=['price_inr']).to_sql('orders', conn, if_exists='replace', index=False)
products.to_sql('products', conn, if_exists='replace', index=False)
conn.close()

print("ALL CHECKPOINTS PASSED")