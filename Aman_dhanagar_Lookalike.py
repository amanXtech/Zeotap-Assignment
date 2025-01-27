import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# 1. Load the data
customers_df = pd.read_csv('customers.csv')
transactions_df = pd.read_csv('transactions.csv')

# 2. Create customer profile features
def create_customer_features(customers_df, transactions_df):
    # Convert SignupDate to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    
    # Calculate transaction-based features
    transaction_features = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',  # Number of transactions
        'Quantity': ['sum', 'mean', 'std'],  # Purchase quantity metrics
        'TotalValue': ['sum', 'mean', 'std']  # Spending metrics
    }).fillna(0)
    
    # Flatten column names
    transaction_features.columns = [
        'total_transactions',
        'total_quantity',
        'avg_quantity',
        'std_quantity',
        'total_spend',
        'avg_spend',
        'std_spend'
    ]
    
    # Create region dummies
    region_dummies = pd.get_dummies(customers_df['Region'], prefix='region')
    
    # Calculate account age in days
    reference_date = pd.to_datetime('2024-12-31')
    customers_df['account_age'] = (reference_date - customers_df['SignupDate']).dt.days
    
    # Combine all features
    customer_features = pd.concat([
        customers_df[['CustomerID', 'account_age']],
        region_dummies
    ], axis=1)
    
    # Merge with transaction features
    customer_features = customer_features.merge(
        transaction_features, 
        left_on='CustomerID', 
        right_index=True,
        how='left'
    ).fillna(0)
    
    return customer_features

# 3. Create feature matrix
customer_features = create_customer_features(customers_df, transactions_df)

# 4. Normalize features
feature_columns = [col for col in customer_features.columns if col != 'CustomerID']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features[feature_columns])

# 5. Compute similarity matrix
similarity_matrix = cosine_similarity(scaled_features)

# 6. Create lookalike recommendations
def get_top_similar_customers(customer_idx, similarity_matrix, customer_ids, n=3):
    customer_similarities = similarity_matrix[customer_idx]
    # Get indices of top similar customers (excluding self)
    similar_indices = customer_similarities.argsort()[::-1][1:n+1]
    similar_scores = customer_similarities[similar_indices]
    similar_customers = [customer_ids[idx] for idx in similar_indices]
    return list(zip(similar_customers, similar_scores))

# 7. Generate recommendations for first 20 customers
lookalike_data = []
customer_ids = customer_features['CustomerID'].values

for i in range(20):  # First 20 customers
    customer_id = customer_ids[i]
    similar_customers = get_top_similar_customers(i, similarity_matrix, customer_ids)
    
    # Format recommendations
    recommendations = [
        {
            'similar_customer_id': sim_cust,
            'similarity_score': round(score, 4)
        }
        for sim_cust, score in similar_customers
    ]
    
    lookalike_data.append({
        'customer_id': customer_id,
        'recommendations': recommendations
    })

# 8. Create and save Lookalike.csv
lookalike_rows = []
for entry in lookalike_data:
    customer_id = entry['customer_id']
    for rank, rec in enumerate(entry['recommendations'], 1):
        lookalike_rows.append({
            'source_customer_id': customer_id,
            'rank': rank,
            'similar_customer_id': rec['similar_customer_id'],
            'similarity_score': rec['similarity_score']
        })

lookalike_df = pd.DataFrame(lookalike_rows)
lookalike_df.to_csv('Lookalike.csv', index=False)

# Print sample results
print("\nSample Lookalike Recommendations:")
for entry in lookalike_data[:5]:  # Show first 5 customers
    print(f"\nCustomer {entry['customer_id']}:")
    for rec in entry['recommendations']:
        print(f"Similar customer: {rec['similar_customer_id']}, "
              f"Similarity score: {rec['similarity_score']:.4f}")

# Additional analysis: Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Std_Dev': np.std(scaled_features, axis=0)
})
print("\nFeature Importance (based on standard deviation):")
print(feature_importance.sort_values('Std_Dev', ascending=False).head())