import pandas as pd

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
    print(f"Processing DataFrame from '{source_name}'...")
    # 1. Find which of our target columns *actually exist* in this file
    original_cols_to_keep = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
    
    if not original_cols_to_keep:
         print(f"Warning: No matching columns found in '{source_name}'. Skipping.")
         return pd.DataFrame()

    # 2. Select *only* those columns and create a copy
    df_cleaned = df[original_cols_to_keep].copy()

    # 3. Rename the columns to our clean schema
    df_cleaned = df_cleaned.rename(columns=COLUMN_MAPPING)
    
    # 4. Add the source identifier
    df_cleaned['data_source'] = source_name

    print(f"Finished processing '{source_name}'. Found {len(df_cleaned)} rows.")
    return df_cleaned

def fill_missing_product_names(df, id_col='id', name_col='product_name'):
    """
    Fills missing product names in a DataFrame using the product_id as a unique key.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'product_id' and 'product_name' columns.
        id_col (str): Name of the product ID column.
        name_col (str): Name of the product name column.
    
    Returns:
        pd.DataFrame: DataFrame with missing product names filled.
    """
    # Create a mapping of unique IDs to known names (excluding nulls)
    id_to_name = (
        df.dropna(subset=[name_col])
          .drop_duplicates(subset=[id_col])
          .set_index(id_col)[name_col]
          .to_dict()
    )
    
    # Fill missing product names using the mapping
    df[name_col] = df.apply(
        lambda row: id_to_name.get(row[id_col], row[name_col]), axis=1
    )
    
    return df

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

    # 3. Remove duplicate reviews
    # We define a duplicate as a review with the same product, rating, title, AND text.
    # We use the 'MUST_HAVE_COLUMNS' as they are the most reliable identifiers.
    original_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=MUST_HAVE_COLUMNS, keep='first')
    new_count = len(merged_df)
    print(f"Dropped {original_count - new_count} duplicate reviews.")
    
    # 4. Final cleanup: Drop rows where (MUST) data is missing
    original_count = len(merged_df)
    fill_missing_product_names(merged_df)
    #merged_df = merged_df.dropna(subset=MUST_HAVE_COLUMNS)
    new_count = len(merged_df)
    
    print(f"Dropped {original_count - new_count} rows due to missing (MUST HAVE) data.")
    print(f"Final merged dataset has {new_count} rows.")
    
    # 5. Ensure all columns from the mapping are present
    all_final_cols = list(COLUMN_MAPPING.values())
    for new_col_name in all_final_cols:
        if new_col_name not in merged_df.columns:
            merged_df[new_col_name] = pd.NA
            
    # 6. Re-order columns for consistency
    final_columns_order = [col for col in all_final_cols if col in merged_df.columns]
    final_columns_order.append('data_source')
    
    merged_df = merged_df[final_columns_order]

    return merged_df