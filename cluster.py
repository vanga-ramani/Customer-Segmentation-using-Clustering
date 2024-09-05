import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/mall_customers.csv'
df = pd.read_csv(url)
df.head()
df = df[['CustomerID', 'Age', 'AnnualIncome', 'SpendingScore']]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'AnnualIncome', 'SpendingScore']])

df_scaled = pd.DataFrame(df_scaled, columns=['Age', 'AnnualIncome', 'SpendingScore'])
df = df[['CustomerID', 'Age', 'AnnualIncome', 'SpendingScore']]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'AnnualIncome', 'SpendingScore']])
df_scaled = pd.DataFrame(df_scaled, columns=['Age', 'AnnualIncome', 'SpendingScore'])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(df_scaled)


df['Cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Cluster', data=df, palette='Set1', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)
sil_score = silhouette_score(df_scaled, clusters)
print(f'Silhouette Score: {sil_score}')
