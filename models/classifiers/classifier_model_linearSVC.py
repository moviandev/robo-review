import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from utils.dataframe import load_and_clean_data

def get_sentiment(rating: float) -> str:
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    elif rating <= 2:
        return 'Negative'
    return pd.NA

def run_sentiment_pipeline():
    df = load_and_clean_data()
    if df.empty:
        print("\nCould not find data files. Please ensure the CSVs are in the correct directory.")
        return

    df['sentiment'] = df['review_rating'].apply(get_sentiment)
    df.dropna(subset=['review_text', 'sentiment'], inplace=True)
    df = df[df['sentiment'].isin(['Positive', 'Neutral', 'Negative'])].copy()

    # Define features (X) and target (y)
    X = df['review_text']
    y = df['sentiment']

    # Split data (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Feature Engineering: TF-IDF Vectorization with bigrams
    tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.8, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 3. Model Training: LinearSVC
    model = LinearSVC(random_state=42, C=0.5, max_iter=2000) 
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    # 4. Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "="*40)
    print("      Sentiment Model Performance")
    print("="*40)
    print(f"Accuracy Score: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
    print("\nClassification Report:\n", report)

    # 5. Visualization: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Negative', 'Neutral', 'Positive'])
    cm_df = pd.DataFrame(cm, index=['True Negative', 'True Neutral', 'True Positive'], 
                         columns=['Predicted Negative', 'Predicted Neutral', 'Predicted Positive'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=.5, linecolor='black')
    plt.title('Confusion Matrix for Sentiment Analysis (Raw Counts)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.yticks(rotation=0)
    plt.savefig('confusion_matrix_model_linearSVC.png')
    plt.close()
    print("Saved Confusion Matrix: 'confusion_matrix.png'")
    
    # 6. Final Prediction and Export
    X_full_tfidf = tfidf.transform(df['review_text'])
    df['predicted_sentiment'] = model.predict(X_full_tfidf)

    output_dir = '../../../results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sentiment_analysis_results.csv')

    final_df_sentiment_analysis = df[['id', 'product_name', 'review_rating', 'review_text', 'review_title', 
                   'sentiment', 'predicted_sentiment', 'data_source']]
    final_df_sentiment_analysis.to_csv(output_path, index=False)

    print(f"\nFinal results exported to: {output_path}")
    print("\nFirst 5 rows of the exported DataFrame:")
    print(final_df_sentiment_analysis.head().to_markdown(index=False, numalign="left", stralign="left"))
    return final_df_sentiment_analysis