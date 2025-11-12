import pandas as pd
import numpy as np # Importado para pd.NA (se necessário, mas já é nativo)

# --- Configuration ---
COLUMN_MAPPING = {
    # (MUST) Columns
    'id': 'id',
    'name': 'product_name',
    'categories': 'categories',
    'reviews.rating': 'review_rating',
    'reviews.text': 'review_text',
    'reviews.title': 'review_title',
    
    # Optional Columns
    'imageURLs': 'image_urls',
    'reviews.date': 'review_date',
    'reviews.doRecommend': 'review_recommend',
    'reviews.numHelpful': 'review_helpful_count',
    'sourceURLs': 'source_urls',
    'reviews.username': 'review_username',
    'username': 'review_username',
}

MUST_HAVE_COLUMNS = [
    'id',
    'product_name',
    'categories',
    'review_rating',
    'review_text',
    'review_title',
]


def handle_df(df, source_name):
    # Lógica inalterada
    print(f"Processing DataFrame from '{source_name}'...")
    original_cols_to_keep = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
    
    if not original_cols_to_keep:
         print(f"Warning: No matching columns found in '{source_name}'. Skipping.")
         return pd.DataFrame()

    df_cleaned = df[original_cols_to_keep].copy()
    df_cleaned = df_cleaned.rename(columns=COLUMN_MAPPING)
    df_cleaned['data_source'] = source_name

    print(f"Finished processing '{source_name}'. Found {len(df_cleaned)} rows.")
    return df_cleaned

def data_cleaner(df_list, source_names):
    all_cleaned_dfs = []
    
    if len(df_list) != len(source_names):
        print("Error: The list of DataFrames and source names must be the same length.")
        return pd.DataFrame()
        
    # 1. Clean each file individually using the helper function
    for df, name in zip(df_list, source_names):
        cleaned_df = handle_df(df, name)
        if not cleaned_df.empty:
            all_cleaned_dfs.append(cleaned_df)
            
    if not all_cleaned_dfs:
        print("Error: No data was successfully cleaned. Returning empty DataFrame.")
        return pd.DataFrame()

    # 2. Concatenate (stack) all the cleaned DataFrames
    print("\nMerging all datasets...")
    merged_df = pd.concat(all_cleaned_dfs, ignore_index=True)
    print(f"Total rows after merge: {len(merged_df)}")
    
    if 'product_name' in merged_df.columns:
        initial_nan_count = merged_df['product_name'].isna().sum()
        
        merged_df['product_name'] = merged_df.groupby('id')['product_name'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else pd.NA)
        )
        
        final_nan_count = merged_df['product_name'].isna().sum()
        
        if final_nan_count > 0:
            merged_df['product_name'] = merged_df['product_name'].fillna('Unknown Product')
            print(f"Filled {initial_nan_count - final_nan_count} product names based on ID.")
            print(f"Filled remaining {final_nan_count} NA names with 'Unknown Product'.")
        else:
            print(f"Filled {initial_nan_count} product names based on ID.")

    else:
        print("Warning: 'product_name' column not found after merge. Skipping name filling.")
        
    # 3. Remove duplicate reviews (Lógica inalterada)
    original_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=MUST_HAVE_COLUMNS, keep='first')
    new_count = len(merged_df)
    print(f"Dropped {original_count - new_count} duplicate reviews.")
    
    # 4. Final cleanup: Drop rows where (MUST) data is missing (Lógica inalterada)
    original_count = len(merged_df)
    
    if 'review_rating' in merged_df.columns:
        merged_df['review_rating'] = pd.to_numeric(merged_df['review_rating'], errors='coerce')
        
    merged_df = merged_df.dropna(subset=MUST_HAVE_COLUMNS)
    new_count = len(merged_df)
    
    print(f"Dropped {original_count - new_count} rows due to missing (MUST HAVE) data.")
    print(f"Final merged dataset has {new_count} rows.")
    
    # 5. Ensure all columns from the mapping are present (Lógica inalterada)
    all_final_cols = list(COLUMN_MAPPING.values())
    for new_col_name in all_final_cols:
        if new_col_name not in merged_df.columns:
            merged_df[new_col_name] = pd.NA
            
    # 6. Re-order columns for consistency (Lógica inalterada)
    final_columns_order = [col for col in all_final_cols if col in merged_df.columns]
    final_columns_order.append('data_source')
    
    merged_df = merged_df[final_columns_order]

    return merged_df