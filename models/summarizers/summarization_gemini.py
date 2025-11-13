import pandas as pd
import os
import sys
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv # Importa a biblioteca para carregar o .env
from typing import Dict

# Define the number of reviews to sample for the prompt
MAX_REVIEWS_PER_PRODUCT = 20

def generate_recommendation_article_from_clusters(results_dir: str = './results') -> str:
    """
    Loads detailed product data (from the clustering output), aggregates review text samples, 
    and generates a single recommendation article using the Google Gemini API.
    
    The function loads the GEMINI_API_KEY from a .env file or environment variables.
    
    Args:
        results_dir: Directory where input data and final output will be saved.
        
    Returns:
        The generated article text (string), or an error message if the API call fails.
    """
    print("\n" + "=" * 80)
    print("STARTING ARTICLE GENERATION (FROM CLUSTERED DATA)")
    print("=" * 80)

    # 1. 🌟 NEW: Load environment variables from .env file
    load_dotenv()
    
    # Check if the API Key is available
    if 'GEMINI_API_KEY' not in os.environ:
        print("ERROR: GEMINI_API_KEY not found in environment variables or .env file.")
        return "ERROR: Missing API Key."

    # 2. Load the clustered review data and the top products report
    reviews_path = os.path.join(results_dir, 'summarization_data_clustered.csv')
    report_path = os.path.join(results_dir, 'top_products_report.csv')
    
    try:
        df_clustered = pd.read_csv(reviews_path)
        df_report = pd.read_csv(report_path)
        print(f"Loaded {df_clustered.shape[0]} detailed reviews.")
    except FileNotFoundError:
        print(f"Error: Missing input files in {results_dir}. Cannot generate article.")
        return "ERROR: Missing input data."

    # 3. Format data into a structured prompt input string (logic remains the same)
    article_sections = []
    
    # Iterate through the products listed in the Top Products Report
    for index, row in df_report.iterrows():
        metacategory = row['metacategory']
        product_name = row['product_name']
        avg_proba = row['positive_proba_mean']
        
        # Filter the full review data for the specific top product
        product_reviews = df_clustered[
            (df_clustered['product_name'] == product_name) & 
            (df_clustered['metacategory'] == metacategory)
        ].copy()
        
        if product_reviews.empty:
            continue
            
        # Determine overall sentiment based on average probability
        if avg_proba >= 0.7:
            overall_sentiment = "Strongly Positive"
        elif avg_proba >= 0.55:
            overall_sentiment = "Positive"
        elif avg_proba >= 0.45:
            overall_sentiment = "Neutral"
        else:
            overall_sentiment = "Negative"

        # 4. Aggregate Sample Review Text for the LLM Prompt
        positive_samples = product_reviews[product_reviews['predicted_sentiment'] == 'positive']['review_text'].head(MAX_REVIEWS_PER_PRODUCT).tolist()
        negative_samples = product_reviews[product_reviews['predicted_sentiment'] == 'negative']['review_text'].head(MAX_REVIEWS_PER_PRODUCT).tolist()
        
        review_text_samples = (
            "POSITIVE REVIEWS: " + " | ".join(positive_samples) + 
            "\nNEGATIVE REVIEWS: " + " | ".join(negative_samples)
        )
        
        # Build the final prompt section for this product
        section_text = f"\n*** METACATEGORY: {metacategory} ***\n"
        section_text += f"- Product Name: {product_name}\n"
        section_text += f"  - Calculated Sentiment: {overall_sentiment} (Avg Positive Score: {avg_proba:.3f})\n"
        section_text += f"  - Raw Review Samples (Summarize These): {review_text_samples}\n"
        
        article_sections.append(section_text)

    # 5. Construct the final Prompt for Gemini (logic remains the same)
    full_review_text = "\n".join(article_sections)
    
    prompt = f"""
Act as a professional market analyst writing an authoritative, persuasive article for a major tech blog. 
Your goal is to synthesize the raw review samples and calculated sentiment scores provided below into a single, cohesive recommendation article.
The article must cover the top recommended products for *each* metacategory listed.

Structure the article strictly in English and use clear, professional formatting (Markdown headings):
1.  A catchy, professional title.
2.  A brief, engaging introduction explaining the analysis methodology (ML Clustering, Sentiment Analysis, and AI synthesis).
3.  Separate, clearly titled sections for each Metacategory.
4.  For each product, write a concise paragraph summarizing its key strengths (based on POSITIVE REVIEWS), weaknesses (based on NEGATIVE REVIEWS), and the overall buying recommendation.
5.  A strong concluding statement on the overall market trends.

--- RAW DATA FOR SYNTHESIS ---
{full_review_text}
---
"""
    
    # 6. Initialize Gemini Client and Generate the Article
    try:
        # A API Key é lida automaticamente do os.environ pelo cliente
        client = genai.Client() 
        
        print("\nCalling Gemini API to generate the article (Model: gemini-2.5-flash)...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        
        final_article = response.text
        
    except APIError as e:
        print(f"\nAPI Error: Could not connect to Gemini. Check your GEMINI_API_KEY and network.")
        return "ERROR: Gemini API call failed."
    except Exception as e:
        print(f"\nError: An unexpected error occurred during API call: {e}")
        return "ERROR: Unknown API issue."

    # 7. Save the final article (logic remains the same)
    article_path = os.path.join(results_dir, 'final_recommendation_article.txt')
    
    try:
        with open(article_path, 'w', encoding='utf-8') as f:
            f.write(final_article)
        print(f"\nFinal article saved successfully to: {article_path}")
    except Exception as e:
        print(f"Error saving final article: {e}")
        
    return final_article