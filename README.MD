# 🤖 Robo Reviews: AI-Powered E-commerce Review Analysis

## 🎯 Project Goal
The Robo Reviews project automates the process of analyzing a large dataset of e-commerce product reviews. It employs a Machine Learning pipeline to classify review sentiment, cluster products into logical "metacategories" based on their characteristics, and then uses the Google Gemini API to synthesize the entire analysis into a professional, human-readable product recommendation article.

The output is a data-driven report that identifies the top-performing products across various metacategories based on sentiment, rating, and review count.

## 🚀 Key Features
Data Ingestion & Cleaning: Loads and merges three raw CSV datasets, handling missing values, duplicates, and column standardization.

Sentiment Analysis (Baseline Model): Trains a Logistic Regression model using TF-IDF features to classify reviews into 'positive', 'neutral', or 'negative' sentiment.

Product Clustering: Uses K-Means clustering on product categories (features reduced via PCA) to group similar products into broad Metacategories (e.g., "Smart Speakers", "Tablets").

Product Ranking: Implements a multi-factor ranking system using a weighted formula that considers:

- Percentage of positive reviews.
- Bayesian average rating.
- Average predicted positive probability.
- Review count.

AI Synthesis (Gemini API): Takes the final ranked products and samples of their corresponding positive and negative reviews to generate a cohesive, market-analyst-style recommendation article using the gemini-2.5-flash model.

## 🛠️ Setup and Installation
1. Prerequisites
You need a Python environment (e.g., Conda, virtualenv) and a Google Gemini API Key.

2. Python Dependencies
The core ML and AI components require several standard data science and Google-specific libraries.

```
pip install pandas numpy scikit-learn matplotlib seaborn datasets google-genai python-dotenv
```
(Note: The classifier_model.py file includes an optional section for a DistilBERT model which would require transformers and torch, but the pipeline is currently configured to run the baseline Logistic Regression model.)

3. API Key Configuration
The summarization_gemini.py script requires your Gemini API Key.

Create a file named .env in the project root directory.

Add your API key in the following format:

```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```
The script uses python-dotenv to automatically load this key into the environment variables.

## 🖥️ How to Run the Pipeline
The entire project is orchestrated through the main.ipynb Jupyter Notebook.

Open main.ipynb in your preferred notebook environment (JupyterLab, VS Code, etc.).

Execute the cells sequentially:

Cell 1: Loads necessary modules from the project files.

Cell 2: Calls the data loading and cleaning function, producing the final df (60,664 rows).

Cell 3 (Classifier Model): Runs the sentiment analysis pipeline, trains the Logistic Regression model, and saves the full dataset with predicted sentiments to ./results/sentiment_analysis_linear_regression.csv.

Cell 4 (Clustering): Performs product clustering, generates the ranking report (top_products_report.csv), and saves the clustered data (summarization_data_clustered.csv).

Cell 5 (Summarization): Calls the Gemini API function to generate the final recommendation article, saving it to ./results/final_recommendation_article.txt.

## ⚙️ Core Logic Breakdown
Data Cleaning (data_cleaner.py)
The script iterates through the three input dataframes, renames columns to a standard format (defined in COLUMN_MAPPING), merges them, and applies cleansing steps:

Fills missing product_name values using the mode of product_name for a given id.

Drops duplicate reviews based on the MUST_HAVE_COLUMNS subset.

Drops any row with missing data in the MUST_HAVE_COLUMNS.

Converts review_rating to numeric.

Sentiment Classification (classifier_model.py)
Sentiment Mapping: Review ratings are mapped to sentiment: (1-2) negative, (3) neutral, (4-5) positive.

Model: A Logistic Regression classifier is used as the robust baseline model.

Feature Engineering: TF-IDF (Term Frequency-Inverse Document Frequency) is used to vectorize the combined review title and text.

Imbalance Handling: class_weight='balanced' is used in the Logistic Regression to mitigate the dataset's heavy imbalance towards positive reviews (91.6%).

Output: The script adds predicted_sentiment and positive_proba columns to the main DataFrame, crucial for the subsequent ranking.

Product Clustering (product_analyzer.py)
Clustering Target: K-Means is applied to the unique products (grouped by id), not every review row.

Feature: TF-IDF features are extracted from the categories text string.

Reduction: PCA is used to reduce the high-dimensional TF-IDF features to 50 components, improving K-Means performance.

Metacategory Mapping: The 4 clusters are manually labeled into logical categories:

Cluster 0: "Smart Speakers & Home Control"

Cluster 1: "Tablets & E-Readers"

Cluster 2: "Video & Streaming Devices"

Cluster 3: "Digital Media & Apps"

Report Generation: The get_top_products_by_category function ranks products within each metacategory based on the mean positive_proba and saves the top 3 to top_products_report.csv.

AI Summarization (summarization_gemini.py)
Input Preparation: The script iterates through the top-ranked products, samples up to 20 positive and 20 negative review texts for each, and compiles this raw data along with the calculated sentiment score into a structured prompt.

Gemini Call: The compiled prompt is sent to the gemini-2.5-flash model.

Instruction: The model is instructed to act as a market analyst, synthesize the raw data, and generate a professional, structured recommendation article with a catchy title, methodology, category sections, and a conclusion.

Output: The final generated text is saved to final_recommendation_article.txt.

## Data
The data was uploaded in Huggingface so you don't need to import anything into the project you can check it in the load_data.

## 💡 Final Thoughts & Ironhack Context
This project, developed by Group 2 for the Ironhack Bootcamp, successfully integrates a full-stack data science pipeline, demonstrating proficiency across several key areas of the curriculum.

### 🏆 Project HighlightsEnd-to-End Pipeline: 
- We successfully executed the entire data science lifecycle, starting with data extraction from external sources (load_dataset_as_dataframe), through cleaning, transformation, modeling, and final deployment via an API.
- Addressing Data Imbalance: The sentiment classifier explicitly addresses the high class imbalance (over 91% positive reviews) by using stratified sampling and the class_weight='balanced' parameter in the Logistic Regression model, leading to a respectable macro F1-score (0.7110 macro avg).
- Feature Engineering & Dimensionality Reduction: We applied advanced NLP techniques like TF-IDF for text vectorization and PCA for dimensionality reduction on the category features, which was crucial for effective K-Means clustering of unique products.
- Advanced Metric Integration: We implemented and reported on key clustering evaluation metrics like the Silhouette Score (0.1827), Davies-Bouldin Index (2.1795), and Calinski-Harabasz Index (11.4705), demonstrating a quantitative approach to model selection.
- LLM Integration: The final step showcases modern AI practices by utilizing the Gemini API for sophisticated text generation, synthesizing complex analytical results into a professional, actionable report.

### 🚧 Future Improvements (Next Steps)
If given more time, Group 2 would focus on the following enhancements:
- Transformer Model Integration: Fully implement the DistilBERT fine-tuning pipeline (currently included but commented out in classifier_model.py) to compare performance against the TF-IDF baseline. This is expected to significantly boost the F1-score for the minority classes (negative/neutral).
- Dynamic Clustering (Elbow/Silhouette): Implement an automated mechanism to find the optimal number of clusters ($k$) instead of hardcoding k=4.
- UI/Deployment: Containerize the pipeline using Docker and deploy the final analysis report via a web framework like Streamlit or Flask to make the results easily accessible and interactive.