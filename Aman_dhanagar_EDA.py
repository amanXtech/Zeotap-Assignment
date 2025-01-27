# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Read the CSV files
customers_df = pd.read_csv('customers.csv')
products_df = pd.read_csv('products.csv')
transactions_df = pd.read_csv('transactions.csv')

# Basic data exploration
print("\nCustomer Data Overview:")
print(customers_df.info())
print("\nProduct Data Overview:")
print(products_df.info())
print("\nTransaction Data Overview:")
print(transactions_df.info())

# 1. Regional Analysis
plt.figure(figsize=(10, 6))
region_counts = customers_df['Region'].value_counts()
plt.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%')
plt.title('Customer Distribution by Region')
plt.show()

# 2. Product Category Analysis
plt.figure(figsize=(10, 6))
category_counts = products_df['Category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Product Distribution by Category')
plt.xticks(rotation=45)
plt.show()

# 3. Price Distribution by Category
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='Price', data=products_df)
plt.title('Price Distribution by Category')
plt.xticks(rotation=45)
plt.show()

# 4. Transaction Analysis
# Convert TransactionDate to datetime
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# Monthly transaction volume
plt.figure(figsize=(12, 6))
monthly_transactions = transactions_df.groupby(transactions_df['TransactionDate'].dt.month)['TransactionID'].count()
plt.plot(monthly_transactions.index, monthly_transactions.values, marker='o')
plt.title('Monthly Transaction Volume')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.show()

# 5. Customer Purchase Frequency
customer_purchase_freq = transactions_df['CustomerID'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(customer_purchase_freq.values, bins=20)
plt.title('Customer Purchase Frequency Distribution')
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Customers')
plt.show()

# 6. Advanced Analysis - Merge data for deeper insights
# Merge transactions with products
merged_df = transactions_df.merge(products_df, on='ProductID')
merged_df = merged_df.merge(customers_df, on='CustomerID')

# Average spending by region
plt.figure(figsize=(10, 6))
avg_spending = merged_df.groupby('Region')['TotalValue'].mean()
sns.barplot(x=avg_spending.index, y=avg_spending.values)
plt.title('Average Transaction Value by Region')
plt.xticks(rotation=45)
plt.show()

# 7. Category Revenue Analysis
category_revenue = merged_df.groupby('Category')['TotalValue'].sum()
plt.figure(figsize=(10, 6))
plt.pie(category_revenue.values, labels=category_revenue.index, autopct='%1.1f%%')
plt.title('Revenue Distribution by Category')
plt.show()

# 8. Time Series Analysis
daily_revenue = merged_df.groupby('TransactionDate')['TotalValue'].sum().reset_index()
plt.figure(figsize=(15, 6))
plt.plot(daily_revenue['TransactionDate'], daily_revenue['TotalValue'])
plt.title('Daily Revenue Trend')
plt.xticks(rotation=45)
plt.show()

# 9. Customer Segmentation by Purchase Value
customer_segments = merged_df.groupby('CustomerID')['TotalValue'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.histplot(customer_segments['TotalValue'], bins=30)
plt.title('Customer Segmentation by Total Purchase Value')
plt.xlabel('Total Purchase Value')
plt.ylabel('Number of Customers')
plt.show()

# 10. Correlation Analysis for Numerical Variables
numerical_cols = ['Quantity', 'Price', 'TotalValue']
correlation_matrix = merged_df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Generate Summary Statistics
print("\nSummary Statistics:")
print("\nProduct Price Statistics:")
print(products_df['Price'].describe())

print("\nTransaction Value Statistics:")
print(transactions_df['TotalValue'].describe())

print("\nTop 5 Best-Selling Products:")
top_products = merged_df.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head()
print(top_products)

print("\nTop 5 Revenue-Generating Categories:")
top_categories = merged_df.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
print(top_categories)

# Save the figures
plt.close('all')