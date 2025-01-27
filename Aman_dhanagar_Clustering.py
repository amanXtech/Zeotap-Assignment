import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# 1. Load and Prepare Data
def load_and_prepare_data():
    # Load data
    customers_df = pd.read_csv('customers.csv')
    transactions_df = pd.read_csv('transactions.csv')
    
    # Convert dates
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    # Create customer features
    # Transaction-based features
    transaction_features = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'Quantity': ['sum', 'mean', 'std'],
        'TotalValue': ['sum', 'mean', 'std']
    }).fillna(0)
    
    # Flatten column names
    transaction_features.columns = [
        'transaction_count',
        'total_quantity',
        'avg_quantity',
        'std_quantity',
        'total_spend',
        'avg_spend',
        'std_spend'
    ]
    
    # Region encoding
    region_dummies = pd.get_dummies(customers_df['Region'], prefix='region')
    
    # Account age
    max_date = pd.to_datetime('2024-12-31')
    customers_df['account_age_days'] = (max_date - customers_df['SignupDate']).dt.days
    
    # Combine features
    features_df = pd.concat([
        transaction_features,
        region_dummies.set_index(customers_df['CustomerID']),
        customers_df.set_index('CustomerID')[['account_age_days']]
    ], axis=1)
    
    return features_df

# 2. Calculate Davies-Bouldin Index
def davies_bouldin_index(X, labels, centroids):
    n_clusters = len(np.unique(labels))
    
    # Calculate cluster dispersions (average distance to centroid)
    cluster_dispersions = []
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        centroid = centroids[i]
        dispersion = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
        cluster_dispersions.append(dispersion)
    
    # Calculate Davies-Bouldin Index
    db_index = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                # Calculate centroid distance
                centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
                ratio = (cluster_dispersions[i] + cluster_dispersions[j]) / centroid_distance
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    
    return db_index / n_clusters

# 3. Perform Clustering Analysis
def perform_clustering(features_df):
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)
    
    # Store metrics for different k values
    metrics = []
    k_range = range(2, 11)
    
    for k in k_range:
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        db_idx = davies_bouldin_index(X, labels, kmeans.cluster_centers_)
        
        metrics.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': db_idx
        })
    
    return pd.DataFrame(metrics), X

# 4. Visualize Results
def visualize_clusters(X, best_k):
    # Perform PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Fit best model
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Cluster visualization
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f'Customer Clusters (k={best_k})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    
    # 2. Metrics comparison
    metrics_df = pd.DataFrame(metrics)
    plt.subplot(2, 2, 2)
    plt.plot(metrics_df['k'], metrics_df['silhouette'], marker='o', label='Silhouette Score')
    plt.plot(metrics_df['k'], metrics_df['davies_bouldin'], marker='o', label='Davies-Bouldin Index')
    plt.title('Clustering Metrics by K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.legend()
    
    # 3. Cluster sizes
    plt.subplot(2, 2, 3)
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    cluster_sizes.plot(kind='bar')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    
    # 4. Feature importance
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=features_df.columns
    )
    plt.subplot(2, 2, 4)
    sns.heatmap(pca_components, cmap='coolwarm', center=0)
    plt.title('Feature Importance in Principal Components')
    
    plt.tight_layout()
    plt.show()
    
    return labels

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    print("Loading and preparing data...")
    features_df = load_and_prepare_data()
    
    # Perform clustering analysis
    print("Performing clustering analysis...")
    metrics, X = perform_clustering(features_df)
    
    # Find best k based on Davies-Bouldin Index
    best_k = metrics.loc[metrics['davies_bouldin'].idxmin(), 'k']
    
    # Visualize results
    print("Creating visualizations...")
    final_labels = visualize_clusters(X, best_k)
    
    # Print final report
    print("\nClustering Results Report:")
    print(f"Number of clusters: {best_k}")
    print("\nClustering Metrics:")
    best_metrics = metrics[metrics['k'] == best_k].iloc[0]
    print(f"Davies-Bouldin Index: {best_metrics['davies_bouldin']:.4f}")
    print(f"Silhouette Score: {best_metrics['silhouette']:.4f}")
    print(f"Calinski-Harabasz Score: {best_metrics['calinski_harabasz']:.4f}")
    
    # Save cluster assignments
    cluster_assignments = pd.DataFrame({
        'CustomerID': features_df.index,
        'Cluster': final_labels
    })
    cluster_assignments.to_csv('cluster_assignments.csv', index=False)
    
    # Save metrics
    metrics.to_csv('clustering_metrics.csv', index=False)