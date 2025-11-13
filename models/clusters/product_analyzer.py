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

# --- Helper Functions (Assumed to be defined in imported modules) ---
# NOTE: The actual implementation of load_and_clean_data and run_complete_pipeline
#       is assumed to exist in utils.dataframe and models.classifiers.classifier_model

# --- 1. NEW CLUSTERING METRICS FUNCTION ---

def calculate_clustering_metrics(features: np.ndarray, labels: np.ndarray):
    """
    Calculates and prints the main cluster evaluation metrics.
    
    - Silhouette Score: (Closer to 1 is better)
    - Davies-Bouldin Index: (Closer to 0 is better)
    - Calinski-Harabasz Index: (Higher is better)
    """
    print("\n--- Evaluating Cluster Quality Metrics ---")
    
    # Check for sufficient unique labels
    if len(np.unique(labels)) < 2:
        print("Error: Need at least 2 unique clusters to calculate metrics.")
        return

    try:
        # 1. Silhouette Score
        sil_score = silhouette_score(features, labels)
        print(f"  - Silhouette Score: {sil_score:.4f}")
        print("    (Interpretation: Close to +1 is great, 0 is overlapping, negative is bad)")

        # 2. Davies-Bouldin Index
        db_score = davies_bouldin_score(features, labels)
        print(f"  - Davies-Bouldin Index: {db_score:.4f}")
        print("    (Interpretation: Closer to 0 is better. Measures separation)")
        
        # 3. Calinski-Harabasz Index
        ch_score = calinski_harabasz_score(features, labels)
        print(f"  - Calinski-Harabasz Index: {ch_score:.4f}")
        print("    (Interpretation: Higher is better. Measures density vs. separation)")
        
    except Exception as e:
        print(f"Error calculating cluster metrics: {e}")

# --- 2. CLUSTER PLOTTING FUNCTION ---

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
    
    # 1. Reduce to 2D for visualization
    if embeddings_reduced.shape[1] > 2:
        pca_vis = PCA(n_components=2, random_state=42)
        embeddings_2d = pca_vis.fit_transform(embeddings_reduced)
    else:
        embeddings_2d = embeddings_reduced
    
    # 2. Create a DataFrame for the plot
    df_plot = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'cluster_id': cluster_labels
    })
    df_plot['metacategory'] = df_plot['cluster_id'].map(category_map)

    # 3. Plot
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
    
    # 4. Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_path = os.path.join(output_dir, 'cluster_visualization.png')
    try:
        plt.savefig(plot_path)
        print(f"Cluster plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving cluster plot: {e}")
    
    plt.close()

# --- 3. CLUSTERING FUNCTION (Updated) ---

def cluster_categories(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Applies K-Means clustering to the 'categories' text, calculates cluster 
    metrics, and plots the visualization.
    """
    print(f"\n--- Starting Product Clustering with k={n_clusters} ---")
    
    # 1. Preprocess text
    text_column = 'categories'
    print(f"Using the column: {text_column}")
    df['processed_text'] = (
        df[text_column]
        .astype(str)
        .fillna('unknown product')
        .str.replace(r'[\[\]\'"]', '', regex=True)
        .str.lower() # Convert to lowercase for better feature generation
    )

    # 2. Feature Generation (TF-IDF)
    print("Generating features with TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english', ngram_range=(1,2))
    embeddings = vectorizer.fit_transform(df['processed_text']).toarray()
    print(f"Feature Matrix Shape: {embeddings.shape}")

    # 3. Dimensionality Reduction (PCA)
    n_components = 50
    print(f"Reducing dimensionality with PCA to {n_components} components...")
    reducer = PCA(n_components=n_components, random_state=42)
    # embeddings_reduced contains the data K-Means will use
    embeddings_reduced = reducer.fit_transform(embeddings)

    # 4. Clustering (K-Means)
    print("Applying K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, verbose=0)
    cluster_labels = kmeans.fit_predict(embeddings_reduced)
    df['cluster_id'] = cluster_labels
    
    print("Cluster Distribution:")
    print(df['cluster_id'].value_counts().sort_index())

    # --- 5. CALL FOR METRICS ---
    calculate_clustering_metrics(embeddings_reduced, cluster_labels)

    # 6. Mapping to Metacategory (Based on assumed K=4 product categories)
    # These names are assigned AFTER clustering by analyzing the common words in each cluster.
    metacategory_mapping = {
        0: "Smart Speakers & Home Control", 
        1: "Tablets & E-Readers", 
        2: "Video & Streaming Devices", 
        3: "Digital Media & Apps"
    }
    
    df['metacategory'] = df['cluster_id'].map(metacategory_mapping)
    print("Metacategory mapping complete.")
    
    # --- 7. CALL FOR PLOTTING ---
    # The output directory assumes the current script is deep inside the project structure.
    output_directory = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results')))
    
    plot_cluster_scatter(
        embeddings_reduced,      # The 50-dim data
        cluster_labels,          # The 0-to-3 labels
        metacategory_mapping,    # The dictionary for the legend
        output_dir=output_directory
    )
    
    df = df.drop(columns=['processed_text'])
    
    return df

# --- 4. REPORT AND ANALYSIS FUNCTION ---

def get_top_products_by_category(df_clustered: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Finds the N products with the highest average 'positive_proba' for each 'metacategory'.
    """
    print(f"\n--- Generating Report: Top {top_n} Products per Metacategory ---")

    # Group by metacategory and product ID
    df_agg = (
        df_clustered.groupby(['metacategory', 'id'], as_index=False)
          .agg(
              # Calculate the mean of the positive prediction probability
              positive_proba_mean=('positive_proba', 'mean'), 
              # Calculate the total number of reviews for the product
              review_count=('id', 'size'),
              # Get the product name (taking the first non-null instance)
              product_name=('product_name', 'first') 
          )
    )
    
    # Sort and get top N per metacategory
    df_sorted = df_agg.sort_values(by='positive_proba_mean', ascending=False)
    
    top_products_report = (
        df_sorted.groupby('metacategory')
                 .head(top_n)
                 .reset_index(drop=True)
    )
    
    print("Final analysis complete.")
    return top_products_report[['metacategory', 'product_name', 'positive_proba_mean', 'review_count']]

# --- 5. SAVING FUNCTION ---

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

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    """
    This block runs the script directly (e.g., python product_clusterer.py)
    """
    
    # --- STEP 0: Set Up Path and Imports ---
    try:
        # Assumes this script is in: /models/classifiers/ or similar
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if project_root not in sys.path:
            sys.path.append(project_root)
            print(f"Adding project root to path: {project_root}")
            
        # Import necessary components from other modules
        from utils.dataframe import load_and_clean_data
        from models.classifiers.classifier_model import run_complete_pipeline
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please check your project structure and ensure necessary libraries (matplotlib, seaborn) are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error setting up path: {e}")
        sys.exit(1)

    print("==================================================")
    print("Starting Product Clustering and Analysis Pipeline...")
    print("==================================================")
    
    # --- STEP 1: Data Loading and Sentiment Analysis ---
    print("\n[STEP 1/4] Loading and cleaning data...")
    df = load_and_clean_data()
    
    print("\n[STEP 2/4] Running sentiment analysis pipeline...")
    # This function should add 'predicted_sentiment', 'confidence_score', and 'positive_proba'
    df_with_sentiment = run_complete_pipeline(df)
    
    if 'positive_proba' not in df_with_sentiment.columns:
        print("Error: The 'positive_proba' column was not found after sentiment analysis.")
        sys.exit(1)
        
    # --- STEP 2: Clustering and Cluster Metrics ---
    print("\n[STEP 3/4] Executing product clustering...")
    # This function prints metrics and saves the cluster visualization plot.
    df_clustered = cluster_categories(df_with_sentiment, n_clusters=4)

    # --- STEP 3: Final Report Generation ---
    print("\n[STEP 4/4] Generating and saving final report...")
    final_report = get_top_products_by_category(df_clustered, top_n=3)
    
    # Define output directory relative to the project structure
    # Assuming 'results' is at the project root level: /robo-review/results
    results_dir = os.path.join(project_root, 'results')
    
    # Save the report of top products
    save_report(final_report, output_dir=results_dir, filename='top_products_report.csv')

    # --- STEP 4: Summarization Data Export (Requested by user) ---
    # Create the DataFrame for summarization process
    df_summarization = df_clustered[[
        'review_text', 'review_title', 'product_name', 
        'metacategory', 'predicted_sentiment', 'id'
    ]].copy()
    
    save_report(df_summarization, output_dir=results_dir, filename='summarization_data_clustered.csv')


    print("\n==================================================")
    print("Pipeline Complete. Final Report:")
    print("==================================================")
    print(final_report)