#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
product_clusterer.py

Este módulo contém funções para agrupar produtos em metacategorias,
avaliar a qualidade dos clusters e analisar a positividade média por produto.

Novas dependências (para plotagem e métricas):
    matplotlib
    seaborn
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, List

# --- Importações para plotagem e métricas ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score
)


# --- 1. NOVA FUNÇÃO DE MÉTRICAS DE CLUSTER ---

def calculate_clustering_metrics(features: np.ndarray, labels: np.ndarray):
    """
    Calcula e imprime as principais métricas de avaliação de cluster.
    
    - Silhouette Score: (Quanto mais perto de 1, melhor)
    - Davies-Bouldin Index: (Quanto mais perto de 0, melhor)
    - Calinski-Harabasz Index: (Quanto mais alto, melhor)
    """
    print("\n--- Avaliando Métricas de Qualidade do Cluster ---")
    
    try:
        # 1. Silhouette Score
        sil_score = silhouette_score(features, labels)
        print(f"  - Silhouette Score: {sil_score:.4f}")
        print("    (Interpretação: Próximo de +1 é ótimo, 0 é sobreposto, negativo é ruim)")

        # 2. Davies-Bouldin Index
        db_score = davies_bouldin_score(features, labels)
        print(f"  - Davies-Bouldin Index: {db_score:.4f}")
        print("    (Interpretação: Quanto mais próximo de 0, melhor. Mede a separação)")
        
        # 3. Calinski-Harabasz Index
        ch_score = calinski_harabasz_score(features, labels)
        print(f"  - Calinski-Harabasz Index: {ch_score:.4f}")
        print("    (Interpretação: Quanto mais alto, melhor. Mede a densidade vs. separação)")
        
    except Exception as e:
        print(f"Erro ao calcular métricas de cluster: {e}")

# --- 2. FUNÇÃO DE PLOTAGEM DE CLUSTER ---

def plot_cluster_scatter(
    embeddings_reduced: np.ndarray, 
    cluster_labels: np.ndarray, 
    category_map: Dict[int, str], 
    output_dir: str = 'result'
):
    """
    Gera e salva um gráfico de dispersão 2D dos clusters.
    """
    print("\nGerando gráfico de dispersão dos clusters...")
    
    # 1. Reduzir para 2D para visualização
    if embeddings_reduced.shape[1] > 2:
        pca_vis = PCA(n_components=2, random_state=42)
        embeddings_2d = pca_vis.fit_transform(embeddings_reduced)
    else:
        embeddings_2d = embeddings_reduced
    
    # 2. Criar um DataFrame para o plot
    df_plot = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'cluster_id': cluster_labels
    })
    df_plot['metacategory'] = df_plot['cluster_id'].map(category_map)

    # 3. Plotar
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
    
    plt.title('Visualização dos Clusters de Produtos (PCA 2D)')
    plt.xlabel('Componente Principal 1 (PC1)')
    plt.ylabel('Componente Principal 2 (PC2)')
    plt.legend(title='Metacategory', loc='best', markerscale=1.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Salvar
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_path = os.path.join(output_dir, 'cluster_visualization.png')
    try:
        plt.savefig(plot_path)
        print(f"Gráfico de clusters salvo em: {plot_path}")
    except Exception as e:
        print(f"Erro ao salvar gráfico de clusters: {e}")
    
    plt.close()

# --- 3. FUNÇÃO DE CLUSTERING (Atualizada) ---

def cluster_categories(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Aplica clustering, calcula métricas de cluster e plota a visualização.
    """
    print(f"\n--- Iniciando Clustering com k={n_clusters} ---")
    
    # 1. Pré-processar texto
    text_column = 'categories'
    print(f"Usando a coluna: {text_column}")
    df['processed_text'] = (
        df[text_column]
        .astype(str)
        .fillna('unknown product')
        .str.replace(r'[\[\]\'\"]', '', regex=True)
    )

    # 2. Feature Generation (TF-IDF)
    print("Gerando features com TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
    embeddings = vectorizer.fit_transform(df['processed_text']).toarray()
    print(f"Shape da matriz de features: {embeddings.shape}")

    # 3. Dimensionality Reduction (PCA)
    n_components = 50
    print(f"Reduzindo dimensionalidade com PCA para {n_components} componentes...")
    reducer = PCA(n_components=n_components, random_state=42)
    # Usamos os embeddings de 50 dimensões para as métricas,
    # pois é o dado real que o K-Means usou.
    embeddings_reduced = reducer.fit_transform(embeddings)

    # 4. Clustering (K-Means)
    print("Aplicando K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
    cluster_labels = kmeans.fit_predict(embeddings_reduced)
    df['cluster_id'] = cluster_labels
    
    print("Distribuição dos Clusters:")
    print(df['cluster_id'].value_counts().sort_index())

    # --- 5. CHAMADA PARA MÉTRICAS (Nova Adição) ---
    # Passamos os dados que o K-Means usou (embeddings_reduced) e os resultados (cluster_labels)
    calculate_clustering_metrics(embeddings_reduced, cluster_labels)

    # 6. Mapeamento para Metacategory
    metacategory_mapping = {
        0: "Tablets & E-Readers", 
        1: "Batteries & Power Accessories", 
        2: "Smart Speakers & Connected Home",    
        3: "Streaming Devices & Home Entertainment"
    }
    
    df['metacategory'] = df['cluster_id'].map(metacategory_mapping)
    print("Mapeamento de Metacategorias concluído.")
    
    # --- 7. CHAMADA PARA PLOTAGEM ---
    plot_cluster_scatter(
        embeddings_reduced,  # Os dados de 50 dimensões
        cluster_labels,      # As labels de 0 a 3
        metacategory_mapping, # O dicionário para a legenda
        output_dir='../../../results'
    )
    
    df = df.drop(columns=['processed_text'])
    
    return df

# --- 4. FUNÇÃO DE ANÁLISE E RELATÓRIO ---

def get_top_products_by_category(df_clustered: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Encontra os N produtos com maior 'positive_proba' média para cada 'metacategory'.
    """
    print(f"\n--- Gerando Relatório: Top {top_n} Produtos por Metacategory ---")

    df_agg = (
        df_clustered.groupby(['metacategory', 'id'], as_index=False)
          .agg(
              positive_proba_mean=('positive_proba', 'mean'),
              review_count=('id', 'size')
          )
    )
    
    df_sorted = df_agg.sort_values(by='positive_proba_mean', ascending=False)
    
    top_products_report = (
        df_sorted.groupby('metacategory')
                 .head(top_n)
                 .reset_index(drop=True)
    )
    
    print("Análise final concluída.")
    return top_products_report

# --- 5. FUNÇÃO PARA SALVAR ---

def save_report(
    df_report: pd.DataFrame, 
    output_dir: str = 'result', 
    filename: str = 'top_products_report.csv'
):
    """
    Salva o DataFrame do relatório em um arquivo CSV no diretório especificado.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório '{output_dir}' criado.")
        
    file_path = os.path.join(output_dir, filename)
    
    try:
        df_report.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"Relatório salvo com sucesso em: {file_path}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo: {e}")

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    """
    Este bloco será executado se você rodar o script diretamente
    (ex: python product_clusterer.py)
    """
    
    # --- ETAPA 0: Configurar o Path e Importar ---
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if project_root not in sys.path:
            sys.path.append(project_root)
            print(f"Adicionando ao path: {project_root}")
            
        from utils.dataframe import load_and_clean_data
        from models.classifiers.classifier_model import run_complete_pipeline
        
        import matplotlib
        import seaborn
        
    except ImportError as e:
        print(f"Erro de Importação: {e}")
        print("Verifique se 'product_clusterer.py' está na pasta correta.")
        print("Certifique-se de ter instalado: pip install matplotlib seaborn scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"Erro inesperado ao configurar o path: {e}")
        sys.exit(1)

    print("==================================================")
    print("Iniciando Pipeline de Análise e Clustering...")
    print("==================================================")
    
    # --- ETAPA 1: Limpeza e Sentimento ---
    print("\n[ETAPA 1/4] Carregando e limpando dados...")
    df = load_and_clean_data()
    
    print("\n[ETAPA 2/4] Executando pipeline de análise de sentimentos...")
    # Esta função deve imprimir o "Classification Report" e mostrar o
    # gráfico da Matriz de Confusão do modelo de *sentimento*.
    df = run_complete_pipeline(df) 
    
    if 'positive_proba' not in df.columns:
        print("Erro: A coluna 'positive_proba' não foi encontrada.")
        sys.exit(1)
        
    # --- ETAPA 2: Clustering e Métricas de Cluster ---
    print("\n[ETAPA 3/4] Executando clustering de produtos...")
    # Esta função agora imprime as métricas de cluster (Silhouette, etc.)
    # e salva o gráfico de dispersão (pontinhos).
    df_clustered = cluster_categories(df, n_clusters=4)

    # --- ETAPA 3: Relatório Final ---
    print("\n[ETAPA 4/4] Gerando e salvando relatório final...")
    final_report = get_top_products_by_category(df_clustered, top_n=3)
    
    save_report(final_report, output_dir='../../result', filename='top_products_report.csv')

    print("\n==================================================")
    print("Pipeline Concluído. Relatório Final:")
    print("==================================================")
    print(final_report)