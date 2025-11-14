#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
product_clusterer.py

This module contains functions to cluster products into metacategories
based on text data, evaluate cluster quality, and analyze average
positivity per product/cluster.

Dependencies:
    pandas, numpy, os, sys
    sklearn (TfidfVectorizer, PCA, KMeans, metrics)
    matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, List

# --- Plotting and Metrics Imports ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
)
import warnings
warnings.filterwarnings('ignore') # Ignore warnings, especially from KMeans initialization

# --- 1. NEW CLUSTERING METRICS FUNCTION ---

def calculate_clustering_metrics(features: np.ndarray, labels: np.ndarray) -> str:
    """
    Calculates and returns the main cluster evaluation metrics as a formatted string.
    
    - Silhouette Score: (Closer to 1 is better)
    - Davies-Bouldin Index: (Closer to 0 is better)
    - Calinski-Harabasz Index: (Higher is better)
    """
    output = ["\n--- Evaluating Cluster Quality Metrics ---"]
    
    if len(np.unique(labels)) < 2:
        output.append("Error: Need at least 2 unique clusters to calculate metrics.")
        return "\n".join(output)

    try:
        # 1. Silhouette Score
        sil_score = silhouette_score(features, labels)
        output.append(f"  - Silhouette Score: {sil_score:.4f}")
        output.append("    (Interpretation: Close to +1 is great, 0 is overlapping, negative is bad)")

        # 2. Davies-Bouldin Index
        db_score = davies_bouldin_score(features, labels)
        output.append(f"  - Davies-Bouldin Index: {db_score:.4f}")
        output.append("    (Interpretation: Closer to 0 is better. Measures separation)")
        
        # 3. Calinski-Harabasz Index
        ch_score = calinski_harabasz_score(features, labels)
        output.append(f"  - Calinski-Harabasz Index: {ch_score:.4f}")
        output.append("    (Interpretation: Higher is better. Measures density vs. separation)")
        
    except Exception as e:
        output.append(f"Error calculating cluster metrics: {e}")
        
    # Print the output and return the string
    metrics_string = "\n".join(output)
    print(metrics_string)
    return metrics_string

# --- 2. CLUSTER PLOTTING FUNCTION ---
# (Logic remains unchanged)

def plot_cluster_scatter(
    embeddings_reduced: np.ndarray, 
    cluster_labels: np.ndarray, 
    category_map: Dict[int, str], 
    output_dir: str = 'results'
):
    """
    Generates and saves a 2D scatter plot of the clusters using PCA for visualization.
    """
    print("\nGenerating cluster scatter plot...")
    
    if embeddings_reduced.shape[1] > 2:
        pca_vis = PCA(n_components=2, random_state=42)
        embeddings_2d = pca_vis.fit_transform(embeddings_reduced)
    else:
        embeddings_2d = embeddings_reduced
    
    df_plot = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'cluster_id': cluster_labels
    })
    df_plot['metacategory'] = df_plot['cluster_id'].map(category_map)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='metacategory',
        palette='bright',
        s=50,
        alpha=0.6,
        legend='full'
    )
    
    plt.title('Product Cluster Visualization (PCA 2D)')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.legend(title='Metacategory', loc='best', markerscale=1.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_path = os.path.join(output_dir, 'cluster_visualization.png')
    try:
        plt.savefig(plot_path)
        print(f"Cluster plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving cluster plot: {e}")
    
    plt.close()

# --- 3. CLUSTERING FUNCTION (MODIFIED AND CORRECTED) ---

def cluster_categories(df: pd.DataFrame, n_clusters: int = 4, results_dir: str = './results') -> pd.DataFrame:
    """
    Applies K-Means clustering to **UNIQUE products** based on the 'categories' text.
    This corrects the over-clustering issue and improves performance.
    """
    print(f"\n--- Starting Product Clustering with k={n_clusters} ---")
    
    # 1. Create a DataFrame of UNIQUE products for clustering.
    df_products = df.groupby('id').agg(
        categories=('categories', 'first'),
        product_name=('product_name', 'first')
    ).reset_index()
    
    print(f"Clustering based on {df_products.shape[0]} unique products (vs. {df.shape[0]} reviews).")
    
    # 1. Preprocess text
    text_column = 'categories'
    print(f"Using the column: {text_column}")
    # 🌟 CORRECTION: Restoring full text preprocessing for TF-IDF accuracy
    df_products['processed_text'] = (
        df_products[text_column]
        .astype(str)
        .fillna('unknown product')
        .str.replace(r'[\[\]\'"]', '', regex=True)
        .str.lower()
    )

    # 2. Feature Generation (TF-IDF)
    # The print statement is verbose but kept to maintain structure
    print('PROCESSED TEXT >>>>> [First 5 samples]') 
    print(df_products['processed_text'].head()) 
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english', ngram_range=(1,2))
    embeddings = vectorizer.fit_transform(df_products['processed_text']).toarray()
    print(f"Feature Matrix Shape: {embeddings.shape}")

    # 3. Dimensionality Reduction (PCA)
    n_components = min(50, embeddings.shape[1]) 
    print(f"Reducing dimensionality with PCA to {n_components} components...")
    reducer = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = reducer.fit_transform(embeddings)

    # 4. Clustering (K-Means)
    print("Applying K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50, verbose=0)
    cluster_labels = kmeans.fit_predict(embeddings_reduced)
    df_products['cluster_id'] = cluster_labels
    
    print("Cluster Distribution:")
    print(df_products['cluster_id'].value_counts().sort_index())

    # --- 5. CALL FOR METRICS (Silhouette Score, etc. is run here) ---
    metrics_output_string = calculate_clustering_metrics(embeddings_reduced, cluster_labels)
    
    # 6. Mapping to Metacategory 
    metacategory_mapping = {
        0: "Smart Speakers & Home Control", 
        1: "Tablets & E-Readers", 
        2: "Video & Streaming Devices", 
        3: "Digital Media & Apps"
    }
    
    # 🌟 CORRECTION: Ensure the string names are mapped for the report/plot legend
    df_products['metacategory'] = df_products['cluster_id'].map(metacategory_mapping)
    print("Metacategory mapping complete.")
    
    # 7. Cleanup
    df_products = df_products.drop(columns=['processed_text'])
    
    # --- 8. CALL FOR PLOTTING (Visualization is generated here) ---
    output_directory = results_dir    
    
    plot_cluster_scatter(
        embeddings_reduced,      
        cluster_labels,          
        metacategory_mapping,    
        output_dir=output_directory
    )
    
    # 9. Merge the results back to the original review DataFrame (df).
    df = df.merge(
        df_products[['id', 'metacategory', 'cluster_id']],
        on='id',
        how='left',
        suffixes=('_old', '')
    )
    
    if 'metacategory_old' in df.columns:
        df = df.drop(columns=['metacategory_old'])
    
    df['metacategory'] = df['metacategory'].fillna('Unknown Product Category')
    
    print(f"Clustering results successfully mapped back to {df.shape[0]} reviews.")
    
    return df, metrics_output_string

# --- 4. REPORT AND ANALYSIS FUNCTION ---
# (Logic remains unchanged)

def get_top_products_by_category(df_clustered: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Finds the N products with the highest average 'positive_proba' for each 'metacategory'.
    """
    print(f"\n--- Generating Report: Top {top_n} Products per Metacategory ---")

    df_agg = (
        df_clustered.groupby(['metacategory', 'id'], as_index=False)
          .agg(
              positive_proba_mean=('positive_proba', 'mean'), 
              review_count=('id', 'size'),
              product_name=('product_name', 'first') 
          )
    )
    
    df_sorted = df_agg.sort_values(by='positive_proba_mean', ascending=False)
    
    top_products_report = (
        df_sorted.groupby('metacategory')
                 .head(top_n)
                 .reset_index(drop=True)
    )
    
    print("Final analysis complete.")
    return top_products_report[['metacategory', 'product_name', 'positive_proba_mean', 'review_count']]

# --- 5. SAVING FUNCTION ---
# (Logic remains unchanged)

def save_report(
    df_report: pd.DataFrame, 
    output_dir: str, 
    filename: str
):
    """
    Saves the report DataFrame to a CSV file in the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
        
    file_path = os.path.join(output_dir, filename)
    
    try:
        df_report.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"Report successfully saved to: {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def save_text_file(text_content: str, output_dir: str, filename: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_path = os.path.join(output_dir, filename)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"Metrics successfully saved to: {file_path}")
    except Exception as e:
        print(f"Error saving text file: {e}")
